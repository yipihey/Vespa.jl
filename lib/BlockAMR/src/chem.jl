# chem.jl — the fast analytic H+H₂ chemistry (ChemistryKernels.evolve_cell_analytic)
# over a level's block batch.  Species live as UInt16 log₂ mass fractions in the
# block pools (the ChemistryKernels codec — scale-free); gas state decodes through
# the per-block scales.  Flow per level: gather active cells → contiguous f32/u16
# buffers → `solve_chem_analytic_device_u16!` in place (zero-copy on the device)
# → scatter e_int back into Ge (and the ΔGe into Tau, keeping the dual-energy
# pair consistent — the patchgrid convention) + species codes.
#
# nsp convention (analytic_h2): sp[1] = HII, sp[2] = H2I mass fractions.

import ChemistryKernels

@kernel function _chem_gather_k!(rho32, e32, h16, m16,
                                 @Const(D), @Const(Ge), @Const(sp1), @Const(sp2),
                                 @Const(cov), @Const(live_d), @Const(Dsc), @Const(Gsc),
                                 B::Int32, ng::Int32, nd::Int32, stride::Int32,
                                 hfloor::UInt16, ecap::Float32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        idx = base + _lidx(i, j, k, nd)
        d16 = Float32(D[idx])
        ρ = d16 * Dsc[slot]
        # HOT-CELL SKIP (ecap > 0, code specific energy): guard-frozen garbage
        # cells (velocity/energy caps keep-old at unphysical states, T near the
        # 1e9 K solver cap) have cooling times of SECONDS — evolve_cell_analytic
        # grinds its full itcap on them EVERY substep (the j256h "hang": hours
        # per root step at 100% compute).  Feed them ρ=0, e=0: the solver's
        # density clamp makes every rate ~0 → one-iteration exit; the scatter
        # re-derives the same gate and discards the result.
        ehot = ecap > 0.0f0 &&
               Float32(Ge[idx]) * Gsc[slot] > ecap * max(ρ, 1.0f-30)
        # covered cells (child-restriction targets) skip chem the same way —
        # their results are overwritten by restriction (leaf-only chem)
        (ehot | (cov[idx] != 0x00)) && (ρ = 0.0f0)
        rho32[Int32(t)] = ρ
        # Representability gate: a cell whose STORED density is below the f16
        # normal range (2^-14; 0, subnormal, or NaN) has ≤10 mantissa bits —
        # its Ge/D ratio is quantization garbage, and the gas_temperature cap
        # then "bounds" it at 1e9 K ⇒ cs ~ 3000× the real signal, re-seeded
        # every step (the bamr256L10 chem-powered blast).  Such cells enter
        # chem as COLD vacuum (e=0 → solver evolves from its tiny floor).
        e32[Int32(t)]   = d16 > 6.103515625f-5 ?
                          Float32(Ge[idx]) * Gsc[slot] / ρ : 0.0f0
        # x_HII floor (codec units): a cell can reach the solver with e = 0 (its
        # stored-f16 Ge underflows in a block whose Ge window is set by a hot
        # dense neighbor — the C/F skin geometry) AND its HII at the codec zero;
        # evolve_cell_analytic's closed-form x_HII then hits a 0/0 corner → NaN.
        # Flooring x_HII at ~1e-12 (six decades below the relic fraction) kills
        # the corner for every cell; the codec is monotonic in the u16 code.
        h16[Int32(t)]   = max(sp1[idx], hfloor)
        m16[Int32(t)]   = sp2[idx]
    end
end

@kernel function _chem_scatter_k!(Ge, Tau, sp1, sp2,
                                  @Const(rho32), @Const(e32), @Const(h16), @Const(m16),
                                  @Const(D), @Const(cov), @Const(live_d), @Const(Dsc), @Const(Esc), @Const(Gsc),
                                  B::Int32, ng::Int32, nd::Int32, stride::Int32,
                                  efloor::Float32, ecap::Float32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        idx = base + _lidx(i, j, k, nd)
        # Representability gate, same test as the gather: a stored-subnormal
        # density cell entered chem as e=0 "cold vacuum" — and with its HII
        # also quantized to the codec floor, evolve_cell_analytic hits a 0/0
        # x_e corner and returns NaN (haloA L2 blocks holding restricted core
        # gas next to voids: 1e8 in-block contrast).  Such cells SKIP chem:
        # write nothing, so solver garbage never leaves the scratch buffers.
        d16s = Float32(D[idx])
        ehot = ecap > 0.0f0 &&
               Float32(Ge[idx]) * Gsc[slot] > ecap * max(d16s * Dsc[slot], 1.0f-30)
        if d16s > 6.103515625f-5 && !ehot && cov[idx] == 0x00
            esc = Esc[slot]; gsc = Gsc[slot]     # Ge on its own scale, Tau on Esc
            ge_old = Float32(Ge[idx]) * gsc
            # CMB temperature floor on the specific internal energy: in a hypersonic
            # f16 collapse the deep core's Ge decays to the chem MIN_TEMPERATURE (1 K,
            # sub-CMB) — unphysical (gas can't Compton-cool below the CMB).  efloor is
            # the code specific-energy at T_CMB(z); 0 disables (library default).
            en = e32[Int32(t)]
            en < efloor && (en = efloor)
            ge_new = en * rho32[Int32(t)]
            Ge[idx]  = _narrow(eltype(Ge), ge_new / gsc)
            Tau[idx] = _narrow(eltype(Tau), (Float32(Tau[idx]) * esc + (ge_new - ge_old)) / esc)
            sp1[idx] = h16[Int32(t)]
            sp2[idx] = m16[Int32(t)]
        end
    end
end

"""
    chem_level!(hier, l, dt; a_value, density_units, length_units, time_units,
                hubble = 71, Om = 0.27, OL = 0.73, fh = 0.76)

Advance the fast analytic H+H₂ chemistry over every active cell of level `l`
(`dt` in code time; the `*_units` convert code → CGS exactly as in
`ChemistryKernels.solve_chem_analytic!`).  Needs `hier` built with `nsp ≥ 2`
(sp[1] = HII, sp[2] = H2I).  Updates Ge, Tau (ΔGe-consistent) and the species.
"""
# cached per-level chem staging buffers (cap·B³; reallocated only on pool
# growth — the previous 4×device_zeros per call was per-substep alloc churn)
function _chembufs(lev::Level{V,U,F,I}) where {V,U,F,I}
    nb = lev.cap * lev.B^3
    cb = get(lev.tabs, :chembuf, nothing)
    if cb === nothing || length((cb::Tuple{F,F,U,U})[1]) < nb
        cb = (device_zeros(lev.be, Float32, (nb,)),
              device_zeros(lev.be, Float32, (nb,)),
              device_zeros(lev.be, UInt16, (nb,)),
              device_zeros(lev.be, UInt16, (nb,)))
        lev.tabs[:chembuf] = cb
    end
    return cb::Tuple{F,F,U,U}
end

# f32 species MASS-DENSITY scratch (HII_m = ρ·X, H2I_m = 2·n_H2·mh) for the
# full-network solve_chem_device! path (mode = :full_f32).
function _chembufs_f32(lev::Level{V,U,F,I}) where {V,U,F,I}
    nb = lev.cap * lev.B^3
    cb = get(lev.tabs, :chembuf32, nothing)
    if cb === nothing || length((cb::Tuple{F,F})[1]) < nb
        cb = (device_zeros(lev.be, Float32, (nb,)), device_zeros(lev.be, Float32, (nb,)))
        lev.tabs[:chembuf32] = cb
    end
    return cb::Tuple{F,F}
end

# u16 FRACTION → f32 MASS DENSITY (ρ·decode) before the f32 full-network solve
@kernel function _sp_u16_to_massdens_k!(hf, mf, @Const(h16), @Const(m16), @Const(rho))
    t = @index(Global)
    @inbounds begin
        ρ = rho[t]
        hf[t] = decode_log2sp(Float32, h16[t]) * ρ
        mf[t] = decode_log2sp(Float32, m16[t]) * ρ
    end
end

# f32 MASS DENSITY → u16 FRACTION (encode(HII_m/ρ)) after the solve
@kernel function _sp_massdens_to_u16_k!(h16, m16, @Const(hf), @Const(mf), @Const(rho))
    t = @index(Global)
    @inbounds begin
        invρ = 1.0f0 / max(rho[t], 1.0f-30)
        h16[t] = encode_log2sp(max(hf[t] * invρ, 0.0f0))
        m16[t] = encode_log2sp(max(mf[t] * invρ, 0.0f0))
    end
end

# LEAF-ONLY chem (RAMSES cooling-on-leaves): cells covered by a child level
# are overwritten by restriction every substep — chem iterations there are
# discarded work, and at a stiff collapse core EVERY level except the deepest
# covers the core, making chem cost grow linearly with depth.  The coverage
# mask is painted from the CHILD level's :res (restriction-target) table.
function _cov_mask!(hier::AMRHierarchy, l::Int)
    lev = hier.levels[l + 1]
    cm = get(lev.tabs, :covmask, nothing)
    cm !== nothing && return cm
    cm = device_zeros(lev.be, UInt8, (lev.cap * lev.stride,))
    if l + 2 <= length(hier.levels) && !isempty(hier.levels[l + 2].live)
        clev = hier.levels[l + 2]
        res = _tabt(clev, :res)
        res.total > 0 &&
            _rect_paint_k!(lev.be)(cm, res.jobs, res.cellstart, Int32(res.njobs),
                                   Int32(lev.nd), Int32(lev.stride); ndrange = res.total)
    end
    lev.tabs[:covmask] = cm
    return cm
end

function chem_level!(hier::AMRHierarchy, l::Int, dt::Real;
                     a_value::Real, density_units::Real, length_units::Real,
                     time_units::Real, hubble::Real = 71.0, Om::Real = 0.27,
                     OL::Real = 0.73, fh::Real = 0.76,
                     dtfrac::Real = 0.1, itcap::Int = 5000, rho_switch::Real = 0.0,
                     tmax::Real = 0.0, rate_tables = nothing, cool_tables = nothing,
                     mode::Symbol = :full_u16)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    @assert lev.nsp >= 2 "chem_level! needs nsp ≥ 2 (HII, H2I)"
    n = length(lev.live) * lev.B^3
    bufs = _chembufs(lev)
    rho32 = view(bufs[1], 1:n)
    e32   = view(bufs[2], 1:n)
    h16   = view(bufs[3], 1:n)
    m16   = view(bufs[4], 1:n)
    hfloor = ChemistryKernels.encode_log2sp(1.0f-12)
    # hot-cell skip threshold: tmax [K] → code specific internal energy
    T2K  = (length_units / time_units)^2 * 1.21148e-8      # K per code e
    ecap = tmax > 0 ? Float32(tmax / ((hier.gamma - 1) * 1.22 * T2K)) : 0.0f0
    cov = _cov_mask!(hier, l)
    _chem_gather_k!(lev.be)(rho32, e32, h16, m16, lev.D, lev.Ge, lev.sp[1], lev.sp[2],
                            cov, lev.live_d, lev.Dsc, lev.Gsc, Int32(lev.B), Int32(lev.ng),
                            Int32(lev.nd), Int32(lev.stride), hfloor, ecap; ndrange = n)
    dbg = get(ENV, "BAM_CHEMDBG", "0") == "1"
    local rin, ein, hin, min_
    if dbg
        rin = Array(rho32); ein = Array(e32); hin = Array(h16); min_ = Array(m16)
    end
    # rho_switch > 0 (CGS g/cm³): per-cell handoff — cells denser than rho_switch
    # run the FULL primordial network (3-body H₂ + dissociation), the rest the
    # cheap analytic path.  The analytic path is our IC generator; the dense
    # collapsing core continues on the full network.  0 ⇒ analytic everywhere.
    # solver mode: the FULL primordial network (solve_chem_device*!, evolve_cell)
    # is now as fast as the reduced analytic path (ChemistryKernels Jul-2026), so
    # it is the default.  :full_f32 gathers species as f32 MASS DENSITIES and runs
    # the f32 device solver (the clean-precision validation path); :full_u16 runs
    # the same network directly on the u16 fraction codes (production storage);
    # :analytic_u16 keeps the old reduced path; :hybrid the per-cell density switch.
    ckw = (; a_value, dt, density_units, length_units, time_units,
           hubble, Om, OL, fh, dtfrac, itcap, rate_tables, cool_tables,
           backend = hier.besym, precision = Float32)
    if mode === :full_f32
        hf, mf = _chembufs_f32(lev); hf = view(hf, 1:n); mf = view(mf, 1:n)
        _sp_u16_to_massdens_k!(lev.be)(hf, mf, h16, m16, rho32; ndrange = n)
        ChemistryKernels.solve_chem_device!(rho32, e32, hf, mf; ckw...)
        _sp_massdens_to_u16_k!(lev.be)(h16, m16, hf, mf, rho32; ndrange = n)
    elseif mode === :full_u16
        ChemistryKernels.solve_chem_device_u16!(rho32, e32, h16, m16; ckw...)
    elseif rho_switch > 0 || mode === :hybrid
        ChemistryKernels.solve_chem_hybrid_device_u16!(rho32, e32, h16, m16;
            rho_switch = max(rho_switch, eps(Float64)), ckw...)
    else   # :analytic_u16
        ChemistryKernels.solve_chem_analytic_device_u16!(rho32, e32, h16, m16; ckw...)
    end
    if dbg
        eo = Array(e32)
        # Flag EVERY nonfinite output (a NaN input fails `> 0` comparisons, so
        # gating the check on ein > 0 hid the real culprits), plus finite-but-
        # EXTREME outputs (≥1e4× input) that overflow the f16 Ge encode at
        # scatter.  GATED cells (stored D subnormal) never scatter back — their
        # scratch NaN is benign; print them flagged but only ERROR on non-gated.
        bad = i -> !isfinite(eo[i]) || (ein[i] > 0 && eo[i] > 1.0f4 * ein[i])
        if any(bad, eachindex(eo))
            hd = Array(lev.D); live = Array(lev.live_d)
            B3 = lev.B^3
            gated = i -> begin
                bi = (i - 1) ÷ B3; c = (i - 1) % B3
                slot = Int(live[bi + 1])
                ci = c % lev.B + lev.ng; cj = (c ÷ lev.B) % lev.B + lev.ng
                ck = c ÷ (lev.B^2) + lev.ng
                idx = (slot - 1) * lev.stride + ((ck * lev.nd) + cj) * lev.nd + ci + 1
                Float32(hd[idx]) <= 6.103515625f-5
            end
            nbad = 0; shown = 0
            for i in eachindex(eo)
                bad(i) || continue
                g = gated(i)
                g || (nbad += 1)
                if shown < 12
                    println("CHEMDBG l=", l, " dt=", dt, " a=", a_value,
                            g ? " GATED" : " LIVE",
                            " OUT: e=", eo[i],
                            " IN: rho=", rin[i], " e=", ein[i],
                            " hii_code=", hin[i], " h2_code=", min_[i],
                            " xHII=", ChemistryKernels.decode_log2sp(Float64, hin[i]),
                            " xH2=", ChemistryKernels.decode_log2sp(Float64, min_[i]),
                            " du=", density_units, " tu=", time_units,
                            " vu2=", (length_units/time_units)^2)
                    shown += 1
                end
            end
            flush(stdout)
            nbad > 0 && error("chem NaN: $nbad live cells at level $l")
        end
    end
    # CMB specific-energy floor (code units) at the current redshift, μ=1.22
    # (neutral primordial).  T_CMB = 2.725·(1+z) = 2.725/a; T2 = (l/t)²·mH/kB.
    # Gated by BAM_TCMB_FLOOR (multiple of T_CMB; unset/0 ⇒ disabled).
    tcmb_mult = parse(Float64, get(ENV, "BAM_TCMB_FLOOR", "0"))
    efloor = 0.0f0
    if tcmb_mult > 0
        T_cmb = tcmb_mult * 2.725 / a_value
        T2 = (length_units / time_units)^2 * 1.21148e-8   # K per code specific-energy
        efloor = Float32(T_cmb / ((hier.gamma - 1) * 1.22 * T2))
    end
    _chem_scatter_k!(lev.be)(lev.Ge, lev.Tau, lev.sp[1], lev.sp[2],
                             rho32, e32, h16, m16, lev.D, cov, lev.live_d, lev.Dsc, lev.Esc, lev.Gsc,
                             Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                             Int32(lev.stride), efloor, ecap; ndrange = n)
    return nothing
end
