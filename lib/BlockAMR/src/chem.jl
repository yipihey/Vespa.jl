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
        e32[Int32(t)]   = Float32(Ge[idx]) * Esc[slot] / max(ρ, 1.0f-30)
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
function chem_level!(hier::AMRHierarchy, l::Int, dt::Real;
                     a_value::Real, density_units::Real, length_units::Real,
                     time_units::Real, hubble::Real = 71.0, Om::Real = 0.27,
                     OL::Real = 0.73, fh::Real = 0.76)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    @assert lev.nsp >= 2 "chem_level! needs nsp ≥ 2 (HII, H2I)"
    n = length(lev.live) * lev.B^3
    rho32 = device_zeros(lev.be, Float32, (n,))
    e32   = device_zeros(lev.be, Float32, (n,))
    h16   = device_zeros(lev.be, UInt16, (n,))
    m16   = device_zeros(lev.be, UInt16, (n,))
    _chem_gather_k!(lev.be)(rho32, e32, h16, m16, lev.D, lev.Ge, lev.sp[1], lev.sp[2],
                            lev.live_d, lev.Dsc, lev.Esc, Int32(lev.B), Int32(lev.ng),
                            Int32(lev.nd), Int32(lev.stride); ndrange = n)
    ChemistryKernels.solve_chem_analytic_device_u16!(rho32, e32, h16, m16;
        a_value, dt, density_units, length_units, time_units,
        hubble, Om, OL, fh, backend = hier.besym, precision = Float32)
    _chem_scatter_k!(lev.be)(lev.Ge, lev.Tau, lev.sp[1], lev.sp[2],
                             rho32, e32, h16, m16, lev.live_d, lev.Esc,
                             Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                             Int32(lev.stride); ndrange = n)
    return nothing
end
