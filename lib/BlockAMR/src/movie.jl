# movie.jl — first-class movie output: derived fields rasterized on the GPU onto a
# uniform image at a chosen AMR level, stored as 16-bit-quantized values packed
# losslessly into RGB PNGs, plus a manifest.json the browser player reads
# (tools/movie/player.html: adjustable min/max, lin/log/symlog, colormaps, speed).
#
# Encoding.  Each (field, frame) image holds codes q ∈ 0:65535 in R (high byte) and
# G (low byte); B = 0.  q = 0 means "no data" (uncovered, non-finite, or ≤ 0 for a
# log field).  q ∈ 1:65535 maps linearly onto [lo, hi] of the stored quantity:
# log₁₀(value) for positive fields (enc = "log"), the value itself for signed ones
# (enc = "lin").  lo/hi are per frame, in the manifest.  65535 levels over the
# frame's range: ≲ 3e-4 dex for a 20-dex log field — far below what a colormap
# shows, so the viewer can re-range and re-scale freely.  PNG is a standard image
# type: it ships inside claude.ai Artifacts and decodes natively in browsers.
#
# The manifest is rewritten (atomically) after every frame, so the player can be
# pointed at a run that is still going.  2-D hierarchies (3-D: mid-plane slices
# are the next step).

import CodecZlib, Printf

# ── field catalogue ───────────────────────────────────────────────────────────
# key => (label, enc, default colormap, default display scale, needs)
const MOVIE_CATALOG = Dict{Symbol,NTuple{5,Any}}(
    :rho     => ("density",                 "log", "viridis", "log",    :none),
    :p       => ("thermal pressure",        "log", "magma",   "log",    :none),
    :T       => ("temperature",             "log", "inferno", "log",    :none),
    :B2      => ("magnetic energy B²",      "log", "plasma",  "log",    :mhd),
    :absJ    => ("|∇×B| (current)",         "log", "cividis", "log",    :mhd),
    :Jz      => ("J_z = (∇×B)_z",           "lin", "RdBu",    "symlog", :mhd),
    :divv    => ("∇·v (compression < 0)",   "lin", "RdBu",    "symlog", :none),
    :absw    => ("|∇×v| (vorticity)",       "log", "cividis", "log",    :none),
    :wz      => ("ω_z = (∇×v)_z",           "lin", "RdBu",    "symlog", :none),
    :vmag    => ("speed |v|",               "log", "turbo",   "lin",    :none),
    :vx      => ("v_x",                     "lin", "RdBu",    "lin",    :none),
    :mach    => ("Mach number",             "log", "turbo",   "log",    :none),
    :beta    => ("plasma β",                "log", "RdBu",    "log",    :mhd),
    :entropy => ("entropy p/ρ^γ",           "log", "viridis", "log",    :none),
    :tcool   => ("cooling time |e/ė|",      "log", "turbo",   "log",    :cooling),
    :level   => ("AMR level",               "lin", "gray",    "lin",    :none))
const MOVIE_ORDER = (:rho, :p, :T, :B2, :absJ, :Jz, :divv, :absw, :wz, :vmag, :vx, :mach,
                     :beta, :entropy, :tcool, :level)
_mcode(k::Symbol) = Int32(findfirst(==(k), MOVIE_ORDER))

"""
    MovieWriter(dir, hier; fields = :default, level = finest now, window = nothing,
                cooling = nothing, tfactor = nothing, length_unit = ("code", 1.0),
                units = Dict(), time_labels = [("t", "code", 1.0)], title = "")

`fields`: a list of keys of `MOVIE_CATALOG` (`:default` = everything that applies:
magnetic fields only for MHD, `:tcool` only with `cooling`).  `level`: the AMR level
whose resolution the images have (coarser data is replicated, finer levels are not
needed — coarse cells hold their restriction).  `window = (x0, y0, w, h)` in code
length (default: the whole domain).  Temperature uses the cooling table's μ(n,T)
when `cooling` is given (Kelvin), else `tfactor·p/ρ` (K) when `tfactor` is given
(= μ m_H v_unit²/k_B), else p/ρ in code units.  `units[key] = (unit_label,
factor)` converts a field's code values for display; `length_unit = (label,
code→label factor)` for the axes; `time_labels = [(name, unit, code→unit factor),
…]` adds clocks to every frame.
"""
mutable struct MovieWriter
    dir       :: String
    fields    :: Vector{Symbol}
    level     :: Int
    window    :: NTuple{4,Float64}
    cooling   :: Any
    tfactor   :: Float64          # T = tfactor·p/ρ without a cooling table (0: report p/ρ)
    length_unit :: Tuple{String,Float64}
    units     :: Dict{Symbol,Tuple{String,Float64}}
    time_labels :: Vector{Tuple{String,String,Float64}}
    title     :: String
    frames    :: Vector{Any}      # per frame: (times, per-field (lo, hi, p01, p99))
    nx :: Int; ny :: Int
    meta      :: Dict{String,Any}
end

function MovieWriter(dir::AbstractString, hier::AMRHierarchy; fields = :default,
                     level::Integer = length(hier.levels) - 1, window = nothing, cooling = nothing,
                     tfactor = nothing, length_unit = ("code", 1.0), units = Dict{Symbol,Tuple{String,Float64}}(),
                     time_labels = [("t", "code", 1.0)], title::AbstractString = "",
                     meta = Dict{String,Any}())
    hier.dim == 2 || error("MovieWriter: 2-D hierarchies only (3-D slices: not yet)")
    mhd = hier.physics === :mhd
    fl = fields === :default ?
         [k for k in MOVIE_ORDER if (MOVIE_CATALOG[k][5] !== :mhd || mhd) &&
                                    (MOVIE_CATALOG[k][5] !== :cooling || cooling !== nothing)] :
         collect(Symbol, fields)
    for k in fl
        haskey(MOVIE_CATALOG, k) || error("unknown movie field $k (have $(MOVIE_ORDER))")
        MOVIE_CATALOG[k][5] === :mhd && !mhd && error("field $k needs physics = :mhd")
        MOVIE_CATALOG[k][5] === :cooling && cooling === nothing && error("field $k needs `cooling`")
    end
    Lx = hier.box; Ly = hier.box * hier.nbase[2] / hier.nbase[1]
    win = window === nothing ? (0.0, 0.0, Lx, Ly) : Tuple(Float64.(window))
    dxo = level_dx(hier, level)
    nx = round(Int, win[3] / dxo); ny = round(Int, win[4] / dxo)
    mkpath(dir); foreach(k -> mkpath(joinpath(dir, String(k))), fl)
    install_player(dir)
    MovieWriter(String(dir), fl, Int(level), win, cooling,
                tfactor === nothing ? 0.0 : Float64(tfactor),
                (String(length_unit[1]), Float64(length_unit[2])),
                Dict{Symbol,Tuple{String,Float64}}(Symbol(k) => (String(v[1]), Float64(v[2])) for (k, v) in units),
                [(String(a), String(b), Float64(c)) for (a, b, c) in time_labels], String(title),
                Any[], nx, ny, Dict{String,Any}(meta))
end

# ── the rasterizer: one thread per active cell of one level ───────────────────
@inline function _mprim(D, S1, S2, S3, E, Ge, Bx, By, Bz, idx, dsc, ssc, esc, gsc, γ::Float32, mhd::Bool,
                        de::Bool = false, eta1::Float32 = 0.0f0)
    @inbounds begin
        ρ = max(Float32(D[idx]) * dsc, 1.0f-30)
        vx = Float32(S1[idx]) * ssc / ρ; vy = Float32(S2[idx]) * ssc / ρ; vz = Float32(S3[idx]) * ssc / ρ
        bx = mhd ? Float32(Bx[idx]) : 0.0f0; by = mhd ? Float32(By[idx]) : 0.0f0; bz = mhd ? Float32(Bz[idx]) : 0.0f0
        p = if mhd                 # dual energy (mhd.jl): Ge where the total's thermal part is noise
            KE = 0.5f0 * ρ * (vx*vx + vy*vy + vz*vz); EB = 0.5f0 * (bx*bx + by*by + bz*bz)
            e5 = Float32(E[idx]) * esc
            (γ - 1.0f0) * ((de ? _de_total(e5, KE, EB, Float32(Ge[idx]) * gsc, eta1) : e5) - KE - EB)
        else
            (γ - 1.0f0) * Float32(Ge[idx]) * gsc
        end
        return ρ, vx, vy, vz, max(p, 1.0f-30), bx, by, bz
    end
end

@kernel function _movie_raster_k!(out, @Const(fids), nf::Int32, W::Int32, H::Int32,
                                  D, S1, S2, S3, E, Ge, Bx, By, Bz, @Const(live_d), @Const(org),
                                  @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                                  γ::Float32, mhd::Bool, invdx::Float32, lvl::Float32, r::Int32,
                                  i0::Int32, j0::Int32, @Const(tab), nn::Int, nt::Int, n0::TC, dn::TC,
                                  t0::TC, dT::TC, XH::TC, zfrac::TC, Zrel::TC,
                                  gm1utem::TC, Tfloor::TC, uρ::TC, uv2::TC,
                                  ut::TC, tmode::Int32, tfac::Float32, de::Bool, eta1::Float32,
                                  B::Int32, ng::Int32, nd::Int32, stride::Int32) where {TC}
    t = @index(Global)
    t0_ = Int32(t) - Int32(1)
    B2 = B * B
    bi = t0_ ÷ B2; c = t0_ % B2
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % B + ng; j = c ÷ B + ng
        dsc = Float32(Dsc[slot]); ssc = Float32(Ssc[slot]); esc = Float32(Esc[slot]); gsc = Float32(Gsc[slot])
        # global cell index (level cells) → pixel block [px0, px0+r) × [py0, py0+r)
        gi = org[2bi + Int32(1)] + (i - ng); gj = org[2bi + Int32(2)] + (j - ng)
        px0 = gi * r - i0; py0 = gj * r - j0
        if !(px0 + r <= Int32(0) || px0 >= W || py0 + r <= Int32(0) || py0 >= H)
            z0 = Int32(0); o1 = Int32(1)
            ρ, vx, vy, vz, p, bx, by, bz = _mprim(D, S1, S2, S3, E, Ge, Bx, By, Bz, base + _lidx(i, j, z0, nd),
                                                  dsc, ssc, esc, gsc, γ, mhd, de, eta1)
            xm = _mprim(D, S1, S2, S3, E, Ge, Bx, By, Bz, base + _lidx(i - o1, j, z0, nd), dsc, ssc, esc, gsc, γ, mhd, de, eta1)
            xp = _mprim(D, S1, S2, S3, E, Ge, Bx, By, Bz, base + _lidx(i + o1, j, z0, nd), dsc, ssc, esc, gsc, γ, mhd, de, eta1)
            ym = _mprim(D, S1, S2, S3, E, Ge, Bx, By, Bz, base + _lidx(i, j - o1, z0, nd), dsc, ssc, esc, gsc, γ, mhd, de, eta1)
            yp = _mprim(D, S1, S2, S3, E, Ge, Bx, By, Bz, base + _lidx(i, j + o1, z0, nd), dsc, ssc, esc, gsc, γ, mhd, de, eta1)
            h = 0.5f0 * invdx
            dvx_dx = h * (xp[2] - xm[2]); dvy_dy = h * (yp[3] - ym[3])
            dvy_dx = h * (xp[3] - xm[3]); dvx_dy = h * (yp[2] - ym[2])
            dvz_dx = h * (xp[4] - xm[4]); dvz_dy = h * (yp[4] - ym[4])
            dby_dx = h * (xp[7] - xm[7]); dbx_dy = h * (yp[6] - ym[6])
            dbz_dx = h * (xp[8] - xm[8]); dbz_dy = h * (yp[8] - ym[8])
            v2 = vx*vx + vy*vy + vz*vz; b2 = bx*bx + by*by + bz*bz
            cs = sqrt(γ * p / ρ)
            e = p / ((γ - 1.0f0) * ρ)                         # specific internal energy, code
            Tg, tcool = if tmode == Int32(2)                  # cooling table (K, code time)
                ec = TC(e) * uv2; ρc = TC(ρ) * uρ; nH = XH * ρc / TC(_MH_CGS); ln = log10(nH)
                Tk, _ = _cloudy_T(ec, ln, zfrac, tab, nn, nt, n0, dn, t0, dT, gm1utem, Tfloor)
                ed = _cloudy_edot_spec(nH, Tk, ln, Zrel, XH, tab, nn, nt, n0, dn, t0, dT)
                (Float32(Tk), Float32(abs(ec / ed) / ut))
            elseif tmode == Int32(1)
                (tfac * p / ρ, 0.0f0)
            else
                (p / ρ, 0.0f0)
            end
            wz = dvy_dx - dvx_dy; Jz = dby_dx - dbx_dy
            for f in Int32(1):nf
                k = fids[f]
                v = k == Int32(1) ? ρ :
                    k == Int32(2) ? p :
                    k == Int32(3) ? Tg :
                    k == Int32(4) ? b2 :
                    k == Int32(5) ? sqrt(Jz*Jz + dbz_dy*dbz_dy + dbz_dx*dbz_dx) :
                    k == Int32(6) ? Jz :
                    k == Int32(7) ? dvx_dx + dvy_dy :
                    k == Int32(8) ? sqrt(wz*wz + dvz_dy*dvz_dy + dvz_dx*dvz_dx) :
                    k == Int32(9) ? wz :
                    k == Int32(10) ? sqrt(v2) :
                    k == Int32(11) ? vx :
                    k == Int32(12) ? sqrt(v2) / cs :
                    k == Int32(13) ? 2.0f0 * p / max(b2, 1.0f-38) :
                    k == Int32(14) ? p / ρ^γ :
                    k == Int32(15) ? tcool : lvl
                ob = Int(f - Int32(1)) * Int(W) * Int(H)
                for b in max(py0, Int32(0)):min(py0 + r, H) - Int32(1), a in max(px0, Int32(0)):min(px0 + r, W) - Int32(1)
                    out[ob + Int(b) * Int(W) + Int(a) + 1] = v
                end
            end
        end
    end
end

# quantize: code 0 = no data; 1..65535 ↔ [lo, hi] of log10(v) (log) or v (lin)
@kernel function _movie_quant_k!(q, @Const(out), @Const(lohi), @Const(islog), npx::Int32)
    t = @index(Global)
    @inbounds begin
        f = (Int32(t) - Int32(1)) ÷ npx + Int32(1)
        v = out[t]; lg = islog[f]
        x = lg ? (v > 0.0f0 ? log10(v) : NaN32) : v
        lo = lohi[2f - Int32(1)]; hi = lohi[2f]
        q[t] = if !isfinite(x) || !(hi >= lo)
            UInt16(0)
        elseif hi == lo
            UInt16(1)
        else
            UInt16(1 + unsafe_trunc(Int32, clamp((x - lo) / (hi - lo), 0.0f0, 1.0f0) * 65534.0f0 + 0.5f0))
        end
    end
end

const _NODATA = Float32(NaN)

"""
    write_frame!(mw, hier, t)

Rasterize, encode and write one frame at code time `t` (fills every level's R
ghosts first).  Returns the frame index.
"""
function write_frame!(mw::MovieWriter, hier::AMRHierarchy, t::Real)
    mw.level <= hier.Lcap || error("movie level beyond Lcap")
    be = hier.levels[1].be
    nf = length(mw.fields); W, H = mw.nx, mw.ny; npx = W * H
    out = device_zeros(be, Float32, (nf * npx,)); fill!(out, _NODATA)
    fids = to_device(be, Int32[_mcode(k) for k in mw.fields], Int32)
    Lo = mw.level; dxo = level_dx(hier, Lo)
    i0 = round(Int32, mw.window[1] / dxo); j0 = round(Int32, mw.window[2] / dxo)
    c = mw.cooling
    tmode = c !== nothing ? Int32(2) : (mw.tfactor > 0 ? Int32(1) : Int32(0))
    T = Float32
    tab = c !== nothing ? c.tab : device_zeros(be, Float32, (5,))
    TC = eltype(tab)
    u = c !== nothing ? c.units : CoolingUnits(density = 1.0, length = 1.0, velocity = 1.0)
    gm1utem = TC((hier.gamma - 1) * _MH_CGS / _KB_CGS)
    for l in 0:min(Lo, length(hier.levels) - 1)
        lev = hier.levels[l + 1]
        isempty(lev.live) && continue
        fill_ghosts!(hier, l)
        mhd = ismhd(lev)
        Bs = mhd ? mfields(lev)[1:3] : (lev.D, lev.D, lev.D)
        org = to_device(be, Int32[x for s in lev.live for x in (Int32(lev.meta[s].origin[1]), Int32(lev.meta[s].origin[2]))], Int32)
        r = Int32(2^(Lo - l))
        _movie_raster_k!(be)(out, fids, Int32(nf), Int32(W), Int32(H), gasfields(lev)[1:6]..., Bs...,
                             lev.live_d, org, lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc,
                             T(hier.gamma), mhd, T(1 / level_dx(hier, l)), T(l), r, i0, j0,
                             tab, c === nothing ? 2 : c.nn, c === nothing ? 2 : c.nt,
                             TC(c === nothing ? 0 : c.n0), TC(c === nothing ? 1 : c.dn),
                             TC(c === nothing ? 0 : c.t0), TC(c === nothing ? 1 : c.dT),
                             TC(c === nothing ? 0.76 : c.XH), TC(c === nothing ? 0 : c.Zrel * c.Zsun),
                             TC(c === nothing ? 0 : c.Zrel), gm1utem, TC(c === nothing ? 1 : c.Tfloor),
                             TC(u.density), TC(u.velocity^2), TC(time_unit(u)), tmode, T(mw.tfactor),
                             mhd && hier.mhd.de, T(mhd ? hier.mhd.eta1 : 0.0),
                             Int32(lev.B), Int32(lev.ng), Int32(lev.nd), Int32(lev.stride);
                             ndrange = length(lev.live) * lev.B^2)
    end
    # per-field ranges (in the stored quantity), then quantize on the device
    islog = [MOVIE_CATALOG[k][2] == "log" for k in mw.fields]
    lohi = zeros(Float32, 2nf)
    for (f, k) in enumerate(mw.fields)
        v = view(out, (f - 1) * npx + 1:f * npx)
        if islog[f]
            lo = mapreduce(x -> x > 0 && isfinite(x) ? log10(x) : Inf32, min, v; init = Inf32)
            hi = mapreduce(x -> x > 0 && isfinite(x) ? log10(x) : -Inf32, max, v; init = -Inf32)
        else
            lo = mapreduce(x -> isfinite(x) ? x : Inf32, min, v; init = Inf32)
            hi = mapreduce(x -> isfinite(x) ? x : -Inf32, max, v; init = -Inf32)
        end
        lohi[2f - 1] = lo; lohi[2f] = hi
    end
    q = device_zeros(be, UInt16, (nf * npx,))
    _movie_quant_k!(be)(q, out, to_device(be, lohi, Float32), to_device(be, islog, Bool), Int32(npx);
                        ndrange = nf * npx)
    hq = Array(q)
    k = length(mw.frames)
    stats = Vector{Any}(undef, nf)
    Threads.@threads for f in 1:nf
        codes = view(hq, (f - 1) * npx + 1:f * npx)
        stats[f] = _code_percentiles(codes, lohi[2f - 1], lohi[2f])
        _write_png_codes(joinpath(mw.dir, String(mw.fields[f]), Printf.@sprintf("%05d.png", k)), codes, W, H)
    end
    times = [Float64(t) * fac for (_, _, fac) in mw.time_labels]
    push!(mw.frames, (times = times, stats = [(lohi[2f - 1], lohi[2f], stats[f]...) for f in 1:nf]))
    _write_manifest(mw)
    return k
end

# 0.1 / 99.9 percentiles of the stored quantity, from the codes (a cloud filling 1 %
# of the image must not saturate the default "frame" range)
function _code_percentiles(codes, lo, hi)
    hist = zeros(Int, 65536)
    for c in codes; hist[Int(c) + 1] += 1; end
    n = length(codes) - hist[1]
    n == 0 && return (NaN, NaN)
    val(c) = lo + (c - 1) / 65534 * (hi - lo)
    out = Float64[]
    for target in (0.001n, 0.999n)
        acc = 0
        for c in 1:65535
            acc += hist[c + 1]
            if acc >= target; push!(out, val(c)); break; end
        end
    end
    return (out[1], out[2])
end

# ── PNG (RGB8: R = code >> 8, G = code & 0xff, B = 0), adaptive row filters ───
const _CRC_TABLE = let t = zeros(UInt32, 256)
    for n in 0:255
        c = UInt32(n)
        for _ in 1:8
            c = (c & 1) != 0 ? 0xedb88320 ⊻ (c >> 1) : c >> 1
        end
        t[n + 1] = c
    end
    t
end
function _crc32(data::AbstractVector{UInt8}, crc::UInt32 = 0xffffffff)
    for b in data
        crc = _CRC_TABLE[((crc ⊻ b) & 0xff) + 1] ⊻ (crc >> 8)
    end
    return crc
end
_be32(x) = UInt8[(x >> 24) & 0xff, (x >> 16) & 0xff, (x >> 8) & 0xff, x & 0xff]
function _chunk(io, typ::String, data::Vector{UInt8})
    write(io, _be32(UInt32(length(data))))
    td = vcat(Vector{UInt8}(typ), data)
    write(io, td)
    write(io, _be32(_crc32(td) ⊻ 0xffffffff))
end

@inline _paeth(a::Int, b::Int, c::Int) = (p = a + b - c; pa = abs(p - a); pb = abs(p - b); pc = abs(p - c);
                                          pa <= pb && pa <= pc ? a : (pb <= pc ? b : c))

function _write_png_codes(path::AbstractString, codes::AbstractVector{UInt16}, W::Int, H::Int)
    rb = 3W
    raw = Vector{UInt8}(undef, H * (rb + 1))
    prev = zeros(UInt8, rb); cur = Vector{UInt8}(undef, rb)
    cand = [Vector{UInt8}(undef, rb) for _ in 1:4]
    for row in 0:H-1
        y = H - 1 - row                                   # PNG row 0 = top = largest y
        @inbounds for x in 0:W-1
            c = codes[y * W + x + 1]
            cur[3x + 1] = UInt8(c >> 8); cur[3x + 2] = UInt8(c & 0xff); cur[3x + 3] = 0x00
        end
        best = 0; bestsum = typemax(Int)
        @inbounds for (fi, ft) in enumerate((0, 1, 2, 4))  # None, Sub, Up, Paeth
            o = cand[fi]; s = 0
            for x in 1:rb
                a = x > 3 ? Int(cur[x - 3]) : 0; b = Int(prev[x]); cc = x > 3 ? Int(prev[x - 3]) : 0
                pred = ft == 0 ? 0 : ft == 1 ? a : ft == 2 ? b : _paeth(a, b, cc)
                v = (Int(cur[x]) - pred) & 0xff
                o[x] = UInt8(v); s += v < 128 ? v : 256 - v
            end
            s < bestsum && (bestsum = s; best = fi)
        end
        off = row * (rb + 1)
        raw[off + 1] = UInt8((0, 1, 2, 4)[best])
        copyto!(raw, off + 2, cand[best], 1, rb)
        prev, cur = cur, prev
    end
    z = transcode(CodecZlib.ZlibCompressor(level = 6), raw)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
        _chunk(io, "IHDR", vcat(_be32(UInt32(W)), _be32(UInt32(H)), UInt8[8, 2, 0, 0, 0]))
        _chunk(io, "IDAT", z)
        _chunk(io, "IEND", UInt8[])
    end
    mv(tmp, path; force = true)
end

# ── manifest.json (hand-written JSON; no dependency) ──────────────────────────
_js(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""
_js(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
_js(x::Integer) = string(x)
_js(x::Bool) = string(x)
_js(x::Symbol) = _js(String(x))
_js(v::AbstractVector) = "[" * join(map(_js, v), ",") * "]"
_js(d::AbstractDict) = "{" * join([_js(string(k)) * ":" * _js(v) for (k, v) in d], ",") * "}"
_js(x::Tuple) = _js(collect(x))
_js(::Nothing) = "null"

function _write_manifest(mw::MovieWriter)
    lf = mw.length_unit[2]
    x0, y0, w, h = mw.window
    fields = map(enumerate(mw.fields)) do (f, k)
        lab, enc, cmap, scale, _ = MOVIE_CATALOG[k]
        unit, fac = get(mw.units, k, (k === :T && (mw.cooling !== nothing || mw.tfactor > 0) ? "K" :
                                      k === :tcool && !isempty(mw.time_labels) ? mw.time_labels[end][2] : "code",
                                      k === :tcool && !isempty(mw.time_labels) ? mw.time_labels[end][3] : 1.0))
        Dict("key" => String(k), "label" => lab, "unit" => unit, "factor" => fac, "enc" => enc,
             "cmap" => cmap, "scale" => scale, "dir" => String(k),
             "lohi" => [[fr.stats[f][1], fr.stats[f][2]] for fr in mw.frames],
             "prange" => [[fr.stats[f][3], fr.stats[f][4]] for fr in mw.frames])
    end
    m = Dict{String,Any}("format" => "vespa-movie-1", "title" => mw.title, "nx" => mw.nx, "ny" => mw.ny,
                         "extent" => [x0 * lf, (x0 + w) * lf, y0 * lf, (y0 + h) * lf],
                         "length_unit" => mw.length_unit[1], "level" => mw.level,
                         "prange_pct" => [0.1, 99.9],
                         "time_labels" => [[a, b] for (a, b, _) in mw.time_labels],
                         "times" => [fr.times for fr in mw.frames], "nframes" => length(mw.frames),
                         "fields" => fields, "meta" => mw.meta,
                         "pack" => Dict("per" => 1, "cols" => 1))
    tmp = joinpath(mw.dir, "manifest.json.tmp")
    write(tmp, _js(m)); mv(tmp, joinpath(mw.dir, "manifest.json"); force = true)
end

# ── the browser player ────────────────────────────────────────────────────────
const PLAYER_HTML = joinpath(@__DIR__, "..", "tools", "movie", "player.html")

"""
    install_player(dir)

Write `dir/index.html`: the WebGL2 movie player (tools/movie/player.html) with a
document head, so `python3 -m http.server -d dir` serves a viewable movie.  (The
bare player.html is the claude.ai Artifact page: the publisher adds the head.)
"""
function install_player(dir::AbstractString)
    isfile(PLAYER_HTML) || return nothing
    write(joinpath(dir, "index.html"),
          "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" " *
          "content=\"width=device-width,initial-scale=1,viewport-fit=cover\"></head><body>\n" *
          read(PLAYER_HTML, String) * "\n</body></html>\n")
    return nothing
end

# ── reading our own PNGs back (8-bit RGB, non-interlaced) ─────────────────────
function _read_png_codes(path::AbstractString)
    b = read(path)
    b[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] || error("$path: not a PNG")
    p = 9; W = H = 0; idat = UInt8[]
    be32(i) = (UInt32(b[i]) << 24) | (UInt32(b[i + 1]) << 16) | (UInt32(b[i + 2]) << 8) | UInt32(b[i + 3])
    while p <= length(b)
        n = Int(be32(p)); typ = String(b[p + 4:p + 7]); d = view(b, p + 8:p + 7 + n)
        if typ == "IHDR"
            W = Int(be32(p + 8)); H = Int(be32(p + 12))
            (d[9] == 8 && d[10] == 2 && d[13] == 0) || error("$path: need 8-bit RGB, non-interlaced")
        elseif typ == "IDAT"
            append!(idat, d)
        elseif typ == "IEND"
            break
        end
        p += 12 + n
    end
    raw = transcode(CodecZlib.ZlibDecompressor, idat)
    rb = 3W; prev = zeros(UInt8, rb); cur = zeros(UInt8, rb)
    codes = Vector{UInt16}(undef, W * H)
    for row in 0:H-1
        off = row * (rb + 1); ft = raw[off + 1]
        @inbounds for x in 1:rb
            a = x > 3 ? Int(cur[x - 3]) : 0; up = Int(prev[x]); c = x > 3 ? Int(prev[x - 3]) : 0
            pred = ft == 0 ? 0 : ft == 1 ? a : ft == 2 ? up : ft == 3 ? (a + up) >> 1 : _paeth(a, up, c)
            cur[x] = UInt8((Int(raw[off + 1 + x]) + pred) & 0xff)
        end
        y = H - 1 - row
        @inbounds for x in 0:W-1
            codes[y * W + x + 1] = (UInt16(cur[3x + 1]) << 8) | UInt16(cur[3x + 2])
        end
        prev, cur = cur, prev
    end
    return codes, W, H
end

# ── minimal JSON reader (enough for manifest.json) ────────────────────────────
function _jparse(s::AbstractString)
    i = Ref(firstindex(s))
    ws() = while i[] <= lastindex(s) && isspace(s[i[]]); i[] = nextind(s, i[]); end
    function val()
        ws(); c = s[i[]]
        if c == '{'
            d = Dict{String,Any}(); i[] += 1; ws()
            s[i[]] == '}' && (i[] += 1; return d)
            while true
                ws(); k = val(); ws(); s[i[]] == ':' || error("json: ':' expected"); i[] += 1
                d[k] = val(); ws()
                s[i[]] == ',' ? (i[] += 1) : (s[i[]] == '}' ? (i[] += 1; return d) : error("json: bad object"))
            end
        elseif c == '['
            a = Any[]; i[] += 1; ws()
            s[i[]] == ']' && (i[] += 1; return a)
            while true
                push!(a, val()); ws()
                s[i[]] == ',' ? (i[] += 1) : (s[i[]] == ']' ? (i[] += 1; return a) : error("json: bad array"))
            end
        elseif c == '"'
            io = IOBuffer(); i[] = nextind(s, i[])
            while s[i[]] != '"'
                ch = s[i[]]
                if ch == '\\'
                    i[] = nextind(s, i[]); e = s[i[]]
                    if e == 'u'
                        write(io, Char(parse(UInt16, s[i[]+1:i[]+4]; base = 16))); i[] += 4
                    else
                        write(io, e == 'n' ? '\n' : e == 't' ? '\t' : e)
                    end
                else
                    write(io, ch)
                end
                i[] = nextind(s, i[])
            end
            i[] = nextind(s, i[]); return String(take!(io))
        elseif startswith(SubString(s, i[]), "true");  i[] += 4; return true
        elseif startswith(SubString(s, i[]), "false"); i[] += 5; return false
        elseif startswith(SubString(s, i[]), "null");  i[] += 4; return nothing
        else
            j = i[]; while j <= lastindex(s) && (s[j] in "+-.eE0123456789"); j += 1; end
            t = s[i[]:j-1]; i[] = j
            return occursin(r"[.eE]", t) ? parse(Float64, t) : parse(Int, t)
        end
    end
    return val()
end
read_movie_manifest(dir::AbstractString) = _jparse(read(joinpath(dir, "manifest.json"), String))

"""
    movie_bundle(src, dst; fields = all, every = 1, frames = all, downsample = 1,
                 maxfiles = 480, maxsheet = 4096, title = nothing)

Repack a movie directory for sharing (a claude.ai Artifact holds ≤ 511 files, each
≤ 15 MB): keep every `every`-th frame (or the frame indices `frames`, 0-based),
optionally only `fields`, average `downsample`² pixel blocks (exact in the stored
quantity; no-data pixels excluded), and tile several frames per PNG sheet so the
whole bundle is ≤ `maxfiles` images with sheets ≤ `maxsheet` px a side (4096: a sheet is a ≤ 64 MB
texture in the viewer).  Warns when a sheet exceeds 15 MB or the bundle 256 MB
(the Artifact limits) — then use `downsample`, `every` or fewer `fields`.  Writes
`dst/manifest.json`, `dst/<field>/NNNNN.png`, the bare player `dst/player.html`
(the Artifact page) and `dst/index.html` (for `python3 -m http.server -d dst`).
Returns `(; nfiles, bytes, per, cols)`.
"""
function movie_bundle(src::AbstractString, dst::AbstractString; fields = nothing, every::Int = 1,
                      frames = nothing, downsample::Int = 1, maxfiles::Int = 480, maxsheet::Int = 4096,
                      title = nothing)
    m = read_movie_manifest(src)
    nf0 = m["nframes"]; W0 = m["nx"]; H0 = m["ny"]
    per0 = get(get(m, "pack", Dict()), "per", 1)
    per0 == 1 || error("movie_bundle: source is already packed")
    keep = frames === nothing ? collect(0:every:nf0-1) : collect(Int, frames)
    fl = fields === nothing ? m["fields"] : [f for f in m["fields"] if Symbol(f["key"]) in Symbol.(fields)]
    isempty(fl) && error("movie_bundle: no fields selected")
    ds = downsample; W = W0 ÷ ds; H = H0 ÷ ds
    maxfiles >= length(fl) || error("movie_bundle: maxfiles < number of fields")
    per = 1                                                   # frames per sheet
    while cld(length(keep), per) * length(fl) > maxfiles; per += 1; end
    cols = max(1, min(per, maxsheet ÷ W)); rows = cld(per, cols)
    rows * H <= maxsheet || error("movie_bundle: $per frames of $(W)×$(H) do not fit a $(maxsheet)² sheet; " *
                                  "use downsample, every, fewer fields or larger maxfiles")
    nsheet = cld(length(keep), per)
    mkpath(dst)
    nbytes = Threads.Atomic{Int}(0); biggest = Threads.Atomic{Int}(0)
    for f in fl
        d = joinpath(dst, f["dir"]); mkpath(d)
        Threads.@threads for s in 0:nsheet-1
            ks = keep[s*per+1:min((s+1)*per, length(keep))]
            sw = W * (length(ks) == per ? cols : min(cols, length(ks)))
            sh = H * cld(length(ks), cols)
            sheet = zeros(UInt16, sw * sh)               # row 0 = bottom, as _write_png_codes expects
            for (t, k) in enumerate(ks)
                codes, w0, h0 = _read_png_codes(joinpath(src, f["dir"], Printf.@sprintf("%05d.png", k)))
                tc = (t - 1) % cols; tr = (t - 1) ÷ cols   # tile row counted from the sheet top
                for y in 0:H-1, x in 0:W-1
                    v = if ds == 1
                        codes[y * w0 + x + 1]
                    else
                        acc = 0; n = 0
                        for b in 0:ds-1, a in 0:ds-1
                            c = codes[(y * ds + b) * w0 + x * ds + a + 1]
                            c != 0 && (acc += c; n += 1)
                        end
                        n == 0 ? UInt16(0) : UInt16(round(Int, acc / n))
                    end
                    # tile (tc, tr) occupies x ∈ tc·W…, and (top-down) rows tr·H…; bottom-up y':
                    Y = sh - 1 - (tr * H + (H - 1 - y))
                    sheet[Y * sw + tc * W + x + 1] = v
                end
            end
            path = joinpath(d, Printf.@sprintf("%05d.png", s))
            _write_png_codes(path, sheet, sw, sh)
            Threads.atomic_add!(nbytes, filesize(path)); Threads.atomic_max!(biggest, filesize(path))
        end
    end
    sel(a) = [a[k + 1] for k in keep]
    out = Dict{String,Any}(m)
    out["nx"] = W; out["ny"] = H; out["nframes"] = length(keep)
    out["times"] = sel(m["times"])
    out["fields"] = [merge(f, Dict("lohi" => sel(f["lohi"]), "prange" => sel(f["prange"]))) for f in fl]
    out["pack"] = Dict("per" => per, "cols" => cols)
    title === nothing || (out["title"] = String(title))
    write(joinpath(dst, "manifest.json"), _js(out))
    cp(PLAYER_HTML, joinpath(dst, "player.html"); force = true)
    install_player(dst)
    nfiles = nsheet * length(fl)
    biggest[] > 15 * 2^20 && @warn "movie_bundle: largest sheet $(round(biggest[] / 2^20; digits = 1)) MB > 15 MB (Artifact file limit)"
    nbytes[] > 256 * 2^20 && @warn "movie_bundle: bundle $(round(nbytes[] / 2^20; digits = 1)) MB > 256 MB (Artifact version limit)"
    return (; nfiles, bytes = nbytes[], maxbytes = biggest[], per, cols)
end
