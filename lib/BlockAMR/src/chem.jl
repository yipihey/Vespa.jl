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
                                 @Const(live_d), @Const(Dsc), @Const(Esc),
                                 B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        idx = base + _lidx(i, j, k, nd)
        ρ = Float32(D[idx]) * Dsc[slot]
        rho32[Int32(t)] = ρ
        # f16-vacuumed cells (D underflows to EXACT stored zero, or NaN) must
        # enter chem as COLD vacuum — dividing Ge by the 1e-30 floor fabricates
        # e ~ 1e23 (T~1e30 K NaNs the rate fits — the bamr256L10 2-cell seed).
        # Unit-agnostic predicate: ρ > 0 (false for 0 AND NaN); tiny-but-real
        # densities divide honestly and the gas_temperature cap bounds them.
        e32[Int32(t)]   = ρ > 0.0f0 ? Float32(Ge[idx]) * Esc[slot] / ρ : 0.0f0
        h16[Int32(t)]   = sp1[idx]
        m16[Int32(t)]   = sp2[idx]
    end
end

@kernel function _chem_scatter_k!(Ge, Tau, sp1, sp2,
                                  @Const(rho32), @Const(e32), @Const(h16), @Const(m16),
                                  @Const(live_d), @Const(Esc),
                                  B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        idx = base + _lidx(i, j, k, nd)
        esc = Esc[slot]
        ge_old = Float32(Ge[idx]) * esc
        ge_new = e32[Int32(t)] * rho32[Int32(t)]
        Ge[idx]  = _narrow(eltype(Ge), ge_new / esc)
        Tau[idx] = _narrow(eltype(Tau), (Float32(Tau[idx]) * esc + (ge_new - ge_old)) / esc)
        sp1[idx] = h16[Int32(t)]
        sp2[idx] = m16[Int32(t)]
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

function chem_level!(hier::AMRHierarchy, l::Int, dt::Real;
                     a_value::Real, density_units::Real, length_units::Real,
                     time_units::Real, hubble::Real = 71.0, Om::Real = 0.27,
                     OL::Real = 0.73, fh::Real = 0.76)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    @assert lev.nsp >= 2 "chem_level! needs nsp ≥ 2 (HII, H2I)"
    n = length(lev.live) * lev.B^3
    bufs = _chembufs(lev)
    rho32 = view(bufs[1], 1:n)
    e32   = view(bufs[2], 1:n)
    h16   = view(bufs[3], 1:n)
    m16   = view(bufs[4], 1:n)
    _chem_gather_k!(lev.be)(rho32, e32, h16, m16, lev.D, lev.Ge, lev.sp[1], lev.sp[2],
                            lev.live_d, lev.Dsc, lev.Esc, Int32(lev.B), Int32(lev.ng),
                            Int32(lev.nd), Int32(lev.stride); ndrange = n)
    dbg = get(ENV, "BAM_CHEMDBG", "0") == "1"
    local rin, ein, hin, min_
    if dbg
        rin = Array(rho32); ein = Array(e32); hin = Array(h16); min_ = Array(m16)
    end
    ChemistryKernels.solve_chem_analytic_device_u16!(rho32, e32, h16, m16;
        a_value, dt, density_units, length_units, time_units,
        hubble, Om, OL, fh, backend = hier.besym, precision = Float32)
    if dbg
        nb = mapreduce(x -> !isfinite(x), +, e32)
        if nb > 0
            eo = Array(e32)
            shown = 0
            for i in eachindex(eo)
                if !isfinite(eo[i]) && shown < 6
                    println("CHEMDBG l=", l, " dt=", dt, " a=", a_value,
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
            error("chem NaN: $nb cells at level $l")
        end
    end
    _chem_scatter_k!(lev.be)(lev.Ge, lev.Tau, lev.sp[1], lev.sp[2],
                             rho32, e32, h16, m16, lev.live_d, lev.Esc,
                             Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                             Int32(lev.stride); ndrange = n)
    return nothing
end
