# Coarse–fine flux-correction (reflux) support for the KA hydro.
#
# `muscl_hancock_step_3d!`/`ppml_step_3d!` record grid-frame face fluxes into
# `frec[axis][field]` (flat, col-major, dims `gd`, `ng` ghosts) — `frec[a][c][p]` is the
# flux of conserved component `c` through the LO face of the cell at flat-position `p`,
# field order `(D,S1,S2,S3,E,Ge)`. `test_muscl_grid.jl` proves these reproduce the
# conservative update to round-off.
#
# To make that hydro conservative across an AMR coarse–fine boundary, an AMR substrate
# (Enzo) needs the per-(axis,side,cell) boundary fluxes and the per-(axis,LO-cell) interior
# fluxes. `boundary_flux_register` extracts exactly those from `frec` — on whatever backend
# `frec` lives on (CPU/CUDA/Metal), via KernelAbstractions gather kernels that touch only
# the (D−1) face planes, not the whole grid — and returns them as a `BoundaryFluxSet` whose
# keys / signs / units MATCH Vespa's native `_bflux_capture[_interior]!` (Vespa src/reflux.jl
# + driver.jl). So the validated `Vespa.bflux_plane` rasterizer and the EnzoLib `:julia`
# reflux harness consume it unchanged. (The keying reasoning is salvaged from the discarded
# one-off `gpu_reflux_writer.jl`; what was wrong there was the untested, frozen-only, Enzo-
# segfault-blocked integration — not this mapping.)
#
#   • Face f (1-based, f=1..act+1) is the LO edge of active cell f, between cells f−1 [lo]
#     and f [hi]; so fr-at-flatpos(face k) is the flux at face k, whose +axis LO cell is k−1.
#   • interior register: keyed by the +axis LO cell `c` (active 1..act−1); value = the flux
#     on cell c's HI side = face (c+1) = fr-at-flatpos(ng+c+1).  (An off-by-one here breaks
#     conservation SILENTLY — it does not crash; locked by `test_reflux_capture.jl`.)
#   • outer faces: lo boundary keyed by the first active cell (face 1, flatpos ng+1); hi by
#     the last active cell (face act+1, flatpos ng+act+1).
#   • value units = ∫F·area·dt = F·dt·dx²  (bflux_plane divides by Vcell=dx³ ⇒ F·dt/dx).
#
# Cell keys are 3-D active `CartesianIndex` (flux-dim = the LO/boundary cell, transverse =
# 1..act), exactly the EnzoGridMesh active handle the native `BoundaryFluxRegister` uses.

export boundary_flux_register, boundary_flux_register_ref, BoundaryFluxSet

"""
    BoundaryFluxSet{NV}

The reflux fluxes extracted from one grid's recorded `frec`, in Enzo-compatible keying:
`flux[(axis,side,cell)]` (outer-boundary faces) and `interior[(axis,lo_cell)]` (interior
coarse–fine faces), each an `NTuple{NV,Float64}` of `∫F·area·dt`. `act` is the active dims.
Field access mirrors `Vespa.BoundaryFluxRegister`, so `_write_fluxes!` consumes either.
"""
struct BoundaryFluxSet{NV}
    flux::Dict{Tuple{Int,Symbol,Any},NTuple{NV,Float64}}
    interior::Dict{Tuple{Int,Any},NTuple{NV,Float64}}
    act::NTuple{3,Int}
end

# the two non-`a` axes, increasing
@inline _trans3(a::Int) = a == 1 ? (2, 3) : a == 2 ? (1, 3) : (1, 2)

# 3-D active CartesianIndex with flux-dim `a` = k, transverse (b1,b2) = (u1,u2)
@inline _aidx(a, k, b1, b2, u1, u2) =
    CartesianIndex(ntuple(d -> d == a ? k : d == b1 ? u1 : u2, 3))

# ── KA gather kernels: pull the face planes out of the full flat frec array ──────────────
# Full-grid (col-major, dims gd, 1-based) linear index of cell (i1,i2,i3) is
#   1 + (i1-1)·s1 + (i2-1)·s2 + (i3-1)·s3,  s = (1, gd1, gd1·gd2).
# The caller passes the per-axis strides (sa,sb1,sb2) for the swept axis + its transverse,
# so the kernel needs no runtime axis branching.

# interior faces: out[(c) + (acta-1)·((u1-1) + actb1·(u2-1))] = F at flatpos (ng+c+1) along a
@kernel function _gather_interior_comp!(out, @Const(F), sa::Int, sb1::Int, sb2::Int,
                                        ng::Int, acta::Int, actb1::Int)
    c, u1, u2 = @index(Global, NTuple)
    lin  = 1 + (ng + c) * sa + (ng + u1 - 1) * sb1 + (ng + u2 - 1) * sb2
    oidx = c + (acta - 1) * ((u1 - 1) + actb1 * (u2 - 1))
    @inbounds out[oidx] = F[lin]
end

# one outer-boundary plane at flux flatpos `p` along a: out[u1 + actb1·(u2-1)] = F[lin]
@kernel function _gather_boundary_comp!(out, @Const(F), p::Int, sa::Int, sb1::Int, sb2::Int,
                                        ng::Int, actb1::Int)
    u1, u2 = @index(Global, NTuple)
    lin  = 1 + (p - 1) * sa + (ng + u1 - 1) * sb1 + (ng + u2 - 1) * sb2
    oidx = u1 + actb1 * (u2 - 1)
    @inbounds out[oidx] = F[lin]
end

"""
    boundary_flux_register(frec, gd, ng, dt, dx; nv=5) -> BoundaryFluxSet{nv}

Extract the coarse–fine reflux fluxes of one grid from its recorded `frec` (per-axis
`NTuple` of `nv`+ flat component arrays on any KA backend). `gd` = full dims incl. ghosts,
`ng` = ghost depth, `dt`/`dx` the step + cell width. KA gather kernels copy only the (D−1)
face planes to host; the host then assembles the keyed register.
"""
function boundary_flux_register(frec, gd::NTuple{3,Int}, ng::Int, dt::Real, dx::Real; nv::Int = 5)
    be  = KA.get_backend(frec[1][1])
    act = ntuple(d -> gd[d] - 2 * ng, 3)
    sc  = float(dt) * float(dx)^2                 # ∫F·area·dt = F·dt·dx²
    s   = (1, gd[1], gd[1] * gd[2])               # col-major strides
    flux     = Dict{Tuple{Int,Symbol,Any},NTuple{nv,Float64}}()
    interior = Dict{Tuple{Int,Any},NTuple{nv,Float64}}()
    for a in 1:3
        b1, b2 = _trans3(a); sa = s[a]; sb1 = s[b1]; sb2 = s[b2]
        ai, ab1, ab2 = act[a], act[b1], act[b2]
        Tf = eltype(frec[a][1])
        # gather (device) → host: interior + the two outer boundary planes, per component
        ibuf = ntuple(nv) do comp
            o = device_zeros(be, Tf, ((ai - 1) * ab1 * ab2,))
            ai > 1 && _gather_interior_comp!(be)(o, frec[a][comp], sa, sb1, sb2, ng, ai, ab1;
                                                 ndrange = (ai - 1, ab1, ab2))
            o
        end
        lobuf = ntuple(nv) do comp
            o = device_zeros(be, Tf, (ab1 * ab2,))
            _gather_boundary_comp!(be)(o, frec[a][comp], ng + 1, sa, sb1, sb2, ng, ab1;
                                       ndrange = (ab1, ab2)); o
        end
        hibuf = ntuple(nv) do comp
            o = device_zeros(be, Tf, (ab1 * ab2,))
            _gather_boundary_comp!(be)(o, frec[a][comp], ng + ai + 1, sa, sb1, sb2, ng, ab1;
                                       ndrange = (ab1, ab2)); o
        end
        KA.synchronize(be)
        ih  = map(to_host, ibuf); loh = map(to_host, lobuf); hih = map(to_host, hibuf)
        # interior faces (skip when act==1 on this axis: no interior face)
        for u2 in 1:ab2, u1 in 1:ab1, c in 1:ai-1
            oidx = c + (ai - 1) * ((u1 - 1) + ab1 * (u2 - 1))
            interior[(a, _aidx(a, c, b1, b2, u1, u2))] =
                ntuple(comp -> Float64(ih[comp][oidx]) * sc, nv)
        end
        # outer boundary faces
        for u2 in 1:ab2, u1 in 1:ab1
            oidx = u1 + ab1 * (u2 - 1)
            flux[(a, :lo, _aidx(a, 1,  b1, b2, u1, u2))] =
                ntuple(comp -> Float64(loh[comp][oidx]) * sc, nv)
            flux[(a, :hi, _aidx(a, ai, b1, b2, u1, u2))] =
                ntuple(comp -> Float64(hih[comp][oidx]) * sc, nv)
        end
    end
    return BoundaryFluxSet{nv}(flux, interior, act)
end

"""
    boundary_flux_register_ref(frec, gd, ng, dt, dx; nv=5) -> BoundaryFluxSet{nv}

Independent host-loop reference for [`boundary_flux_register`](@ref): copies the full `frec`
to host and indexes it directly. Same result, different code path — the oracle that locks
the KA gather's index mapping in `test_reflux_capture.jl`.
"""
function boundary_flux_register_ref(frec, gd::NTuple{3,Int}, ng::Int, dt::Real, dx::Real; nv::Int = 5)
    act = ntuple(d -> gd[d] - 2 * ng, 3)
    sc  = float(dt) * float(dx)^2
    frh = ntuple(a -> ntuple(c -> to_host(frec[a][c]), nv), 3)
    lin(c) = c[1] + gd[1] * ((c[2] - 1) + gd[2] * (c[3] - 1))      # 1-based col-major
    Fvec(a, p, b1, b2, u1, u2) = begin
        c = ntuple(d -> d == a ? p : d == b1 ? ng + u1 : ng + u2, 3)
        l = lin(c)
        ntuple(comp -> Float64(frh[a][comp][l]) * sc, nv)
    end
    flux     = Dict{Tuple{Int,Symbol,Any},NTuple{nv,Float64}}()
    interior = Dict{Tuple{Int,Any},NTuple{nv,Float64}}()
    for a in 1:3
        b1, b2 = _trans3(a)
        for u2 in 1:act[b2], u1 in 1:act[b1], c in 1:act[a]-1
            interior[(a, _aidx(a, c, b1, b2, u1, u2))] = Fvec(a, ng + c + 1, b1, b2, u1, u2)
        end
        for u2 in 1:act[b2], u1 in 1:act[b1]
            flux[(a, :lo, _aidx(a, 1, b1, b2, u1, u2))]      = Fvec(a, ng + 1,        b1, b2, u1, u2)
            flux[(a, :hi, _aidx(a, act[a], b1, b2, u1, u2))] = Fvec(a, ng + act[a] + 1, b1, b2, u1, u2)
        end
    end
    return BoundaryFluxSet{nv}(flux, interior, act)
end
