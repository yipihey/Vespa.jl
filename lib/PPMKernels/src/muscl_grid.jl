# ── MUSCL 3-D unsplit SSP-RK2 driver (Enzo HydroMethod=3, HD_RK) ──────────────
# Assembles the certified 1-D PLM+HLL flux line (`muscl_flux_line!`) into Enzo's
# dimensionally-UNSPLIT Runge-Kutta Godunov step. Per stage: convert the conserved
# state (D,S1,S2,S3,τ) → primitives, then for each axis transpose the swept axis to
# the lead (cyclic velocity roles, exactly like the PPM grid), take the HLL flux
# line, and accumulate the flux divergence  dU −= (F[i+1]−F[i])·dt/dx  into ONE
# shared dU (all three axes summed — that is what "unsplit" means; cf.
# Grid_Hydro3D.C). The momentum-flux components rotate back with the same cyclic
# map as the velocities. SSP-RK2 (Grid_RungeKutta2_{1st,2nd}Step.C + UpdatePrim):
#   stage 1  UpdatePrim(dU,1,1):  U1   = U0 + dU(U0)
#   stage 2  UpdatePrim(dU,½,½):  Unew = ½(U0 + U1 + dU(U1))
#
# Conserved set (Enzo iD,iS1,iS2,iS3,iEtot): D=ρ, S=ρv, τ=ρ·etot (etot specific).
# Pure hydro, DualEnergyFormalism=0 (matches the certified flux-line config), so
# eint is reconstructed from τ: eint = τ/D − ½|v|².

export muscl_step_3d!, prim_to_cons!, cons_to_prim!, total_field

# cons (D,S1,S2,S3,τ) → prim (ρ,eint,vx,vy,vz) over the whole grid. `small` floors
# the internal energy: at high Mach eint = τ/D − ½|v|² is a small difference of large
# numbers and can dip negative (roundoff / strong shocks) ⇒ p<0 ⇒ √(γp/ρ) NaN; the
# floor keeps it positive (no-op in normal flows where eint ≫ small).
@kernel function _cons2prim_k!(rho, eint, vx, vy, vz,
                               @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(Tau), small)
    i = @index(Global, Linear)
    T = eltype(rho)
    @inbounds begin
        d = D[i]; u = S1[i] / d; v = S2[i] / d; w = S3[i] / d
        rho[i] = d; vx[i] = u; vy[i] = v; vz[i] = w
        eint[i] = max(Tau[i] / d - T(0.5) * (u * u + v * v + w * w), small)
    end
end

# ── Dual-energy formalism (Enzo hydro_rk style): Ge=ρe advected as a 6th field, ──
# eint selected per-cell between the total-energy-derived value and the separately
# evolved one. Avoids the catastrophic eint = τ/ρ − ½|v|² cancellation at high Mach.
#
# Selection (Grid_UpdatePrim.C): with eint_tot = τ/ρ−½v², eint_dual = Ge/ρ, use the
# total-energy value when pressure is dynamically significant — cs² > η₁·v² AND
# eint_tot > ½·eint_dual — else trust the advected dual energy. Floored to `small`.
@inline function _dual_eint(τ::T, d::T, ge::T, v2::T, gamma::T, gm1::T, eta1::T, small::T) where {T}
    eint_tot = τ / d - T(0.5) * v2
    eint_dual = ge / d
    cs2 = eint_tot > zero(T) ? gamma * gm1 * eint_tot : zero(T)     # γp/ρ
    use_tot = (cs2 > eta1 * v2) && (eint_tot > T(0.5) * eint_dual)
    return max(use_tot ? eint_tot : eint_dual, small)
end

# cons→prim with the dual-energy selection (reads the gas energy Ge too).
@kernel function _cons2prim_dual_k!(rho, eint, vx, vy, vz,
                                    @Const(D), @Const(S1), @Const(S2), @Const(S3),
                                    @Const(Tau), @Const(Ge), gamma, eta1, small)
    i = @index(Global, Linear)
    T = eltype(rho)
    @inbounds begin
        d = D[i]; u = S1[i] / d; v = S2[i] / d; w = S3[i] / d; v2 = u*u + v*v + w*w
        rho[i] = d; vx[i] = u; vy[i] = v; vz[i] = w
        eint[i] = _dual_eint(Tau[i], d, Ge[i], v2, gamma, gamma - one(T), eta1, small)
    end
end

# advective update of the gas-energy field from its interface flux (scalar — no
# momentum rotation, transposes like D/τ).
@kernel function _ge_update_k!(Ge, @Const(fge), na::Int, nfi::Int, nghost::Int, dtdx)
    gi, gj = @index(Global, NTuple)
    cl = (gj - 1) * na + nghost + gi
    fo = (gj - 1) * nfi + gi
    @inbounds Ge[cl] -= (fge[fo+1] - fge[fo]) * dtdx
end

# end-of-step sync (Grid_UpdatePrim.C reset): re-select eint and overwrite the gas
# energy Ge=ρ·eint and the total energy τ=ρ·(eint+½v²) so the two stay consistent
# (and so a hot region's purely-advected Ge does not drift). Whole-grid, per cell.
# This is where the DEF trades exact total-energy conservation for accurate eint in
# the cold/supersonic regions where τ−½v² is unreliable.
@kernel function _dual_sync_k!(@Const(D), @Const(S1), @Const(S2), @Const(S3),
                               Tau, Ge, gamma, eta1, small)
    i = @index(Global, Linear)
    T = eltype(Tau)
    @inbounds begin
        d = D[i]; u = S1[i] / d; v = S2[i] / d; w = S3[i] / d; v2 = u*u + v*v + w*w
        e = _dual_eint(Tau[i], d, Ge[i], v2, gamma, gamma - one(T), eta1, small)
        Ge[i] = d * e
        Tau[i] = d * (e + T(0.5) * v2)
    end
end

"`dual_energy_sync!(D,S1,S2,S3,Tau,Ge; gamma, eta1=1e-3, small_rho=1e-10)` — DEF reset."
function dual_energy_sync!(D, S1, S2, S3, Tau, Ge; gamma::Real, eta1::Real = 1e-3,
                           small_rho::Real = 1e-10)
    be = KA.get_backend(D); T = eltype(D)
    _dual_sync_k!(be)(D, S1, S2, S3, Tau, Ge, T(gamma), T(eta1), T(small_rho); ndrange = length(D))
    KA.synchronize(be)
    return nothing
end

# per-axis flux divergence, into axis-local (transposed) full-grid dU slabs.
# Local cell m=1..active of pencil gj lives at cl=(gj-1)*na + nghost + m; the
# interfaces straddling it are fo and fo+1 with fo=(gj-1)*nfi + m. Ghost cells
# keep the zero the scratch was filled with, so the untranspose-add is clean.
@kernel function _muscl_div_k!(dDl, dNl, dT1l, dT2l, dEl,
                               @Const(fd), @Const(fs1), @Const(fs2), @Const(fs3), @Const(fe),
                               na::Int, nfi::Int, nghost::Int, dtdx)
    gi, gj = @index(Global, NTuple)             # gi=1..active, gj=1..ntr
    cl = (gj - 1) * na + nghost + gi
    fo = (gj - 1) * nfi + gi
    @inbounds begin
        dDl[cl]  = -(fd[fo+1]  - fd[fo])  * dtdx
        dNl[cl]  = -(fs1[fo+1] - fs1[fo]) * dtdx     # normal momentum
        dT1l[cl] = -(fs2[fo+1] - fs2[fo]) * dtdx     # transverse 1
        dT2l[cl] = -(fs3[fo+1] - fs3[fo]) * dtdx     # transverse 2
        dEl[cl]  = -(fe[fo+1]  - fe[fo])  * dtdx
    end
end

# inverse transpose that ADDS into dst (the global dU lives across all 3 axes, so
# each axis's contribution is summed in, not overwritten — cf. _untranspose_into!).
@kernel function _gather_add3!(dst, @Const(src), m1::Int, m2::Int, sa::Int, sb::Int, sc::Int)
    g = @index(Global, Linear)
    a = (g - 1) % m1 + 1
    t = (g - 1) ÷ m1
    b = t % m2 + 1
    c = t ÷ m2 + 1
    @inbounds dst[g] += src[1 + (a - 1) * sa + (b - 1) * sb + (c - 1) * sc]
end

function _untranspose_add_into!(dst, slab, dims::NTuple{3,Int}, perm::NTuple{3,Int})
    invp = _invperm3(perm)
    md = (dims[perm[1]], dims[perm[2]], dims[perm[3]])      # slab dims
    mstr = (1, md[1], md[1] * md[2])
    be = KA.get_backend(dst)
    _gather_add3!(be)(dst, slab, dims[1], dims[2], mstr[invp[1]], mstr[invp[2]], mstr[invp[3]];
                      ndrange = length(dst))
    return dst
end

# additive flux divergence written DIRECTLY into the global dU (used for the
# contiguous x-axis, where the local frame already IS the global frame — no
# transpose, no local slab, no gather). Momentum order (S1,S2,S3) = (normal,t1,t2).
@kernel function _muscl_div_add_x!(dD, dS1, dS2, dS3, dE,
                                   @Const(fd), @Const(fs1), @Const(fs2), @Const(fs3), @Const(fe),
                                   na::Int, nfi::Int, nghost::Int, dtdx)
    gi, gj = @index(Global, NTuple)
    cl = (gj - 1) * na + nghost + gi
    fo = (gj - 1) * nfi + gi
    @inbounds begin
        dD[cl]  += -(fd[fo+1]  - fd[fo])  * dtdx
        dS1[cl] += -(fs1[fo+1] - fs1[fo]) * dtdx
        dS2[cl] += -(fs2[fo+1] - fs2[fo]) * dtdx
        dS3[cl] += -(fs3[fo+1] - fs3[fo]) * dtdx
        dE[cl]  += -(fe[fo+1]  - fe[fo])  * dtdx
    end
end

# dU(U) — the unsplit flux-divergence operator. Fills the five global conserved
# increments (dD,dS1,dS2,dS3,dE) from the conserved state (D,S1,S2,S3,Tau). `prim`
# is caller-provided scratch (ρ,eint,vx,vy,vz). ALL temporaries (these + the
# per-axis ones) come from the scratch pool and are NOT reset here — the single
# reset at the top of `muscl_step_3d!` recycles the whole step's working set across
# steps (no per-step allocation once the pool is warm; no intra-step reuse, so no
# dangling-buffer hazard).
# single-field flux divergence for the dual-energy gas energy (a scalar — no
# momentum rotation): add-in-place for the contiguous x-axis, into a local slab for
# y/z (then untranspose-add like the scalar density/energy increments).
@kernel function _scalar_div_add!(dF, @Const(fge), na::Int, nfi::Int, nghost::Int, dtdx)
    gi, gj = @index(Global, NTuple)
    cl = (gj - 1) * na + nghost + gi; fo = (gj - 1) * nfi + gi
    @inbounds dF[cl] += -(fge[fo+1] - fge[fo]) * dtdx
end
@kernel function _scalar_div_local!(dFl, @Const(fge), na::Int, nfi::Int, nghost::Int, dtdx)
    gi, gj = @index(Global, NTuple)
    cl = (gj - 1) * na + nghost + gi; fo = (gj - 1) * nfi + gi
    @inbounds dFl[cl] = -(fge[fo+1] - fge[fo]) * dtdx
end

function _muscl_L!(dD, dS1, dS2, dS3, dE, prim,
                   D, S1, S2, S3, Tau, dims::NTuple{3,Int}, ng::Int,
                   dt::Real, gamma::Real, theta::Real, dx::Real, small_rho::Real;
                   ge = nothing, dGe = nothing, eta1::Real = 1e-3)
    be = KA.get_backend(D); T = eltype(D); N = length(D)
    rho, eint, vx, vy, vz = prim; dual = ge !== nothing
    if dual
        _cons2prim_dual_k!(be)(rho, eint, vx, vy, vz, D, S1, S2, S3, Tau, ge,
                               T(gamma), T(eta1), T(small_rho); ndrange = N)
        fill!(dGe, zero(T))
    else
        _cons2prim_k!(be)(rho, eint, vx, vy, vz, D, S1, S2, S3, Tau, T(small_rho); ndrange = N)
    end
    fill!(dD, zero(T)); fill!(dS1, zero(T)); fill!(dS2, zero(T)); fill!(dS3, zero(T)); fill!(dE, zero(T))
    KA.synchronize(be)
    for axis in (1, 2, 3)
        na = dims[axis]; ntr = N ÷ na
        active = na - 2 * ng; nfi = active + 1
        dtdx = T(dt) / T(dx)
        # cyclic velocity roles (normal, transverse1, transverse2) — Enzo EulerSweeps
        vu, vv, vw = axis == 1 ? (vx, vy, vz) : axis == 2 ? (vy, vz, vx) : (vz, vx, vy)
        fd  = _scratch(D, nfi * ntr); fs1 = _scratch(D, nfi * ntr); fs2 = _scratch(D, nfi * ntr)
        fs3 = _scratch(D, nfi * ntr); fe  = _scratch(D, nfi * ntr)
        fge = dual ? _scratch(D, nfi * ntr) : nothing

        if axis == 1                          # contiguous — no transpose, add in place
            muscl_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vu, vv, vw;
                             ncells = na, nghost = ng, jdim = ntr, gamma = gamma,
                             theta = theta, small_rho = small_rho, fge = fge)
            _muscl_div_add_x!(be)(dD, dS1, dS2, dS3, dE, fd, fs1, fs2, fs3, fe,
                                  na, nfi, ng, dtdx; ndrange = (active, ntr))
            dual && _scalar_div_add!(be)(dGe, fge, na, nfi, ng, dtdx; ndrange = (active, ntr))
        else
            perm = _axis_perm(axis)
            rhoT = transpose3(rho, dims, perm); eintT = transpose3(eint, dims, perm)
            vuT  = transpose3(vu, dims, perm);  vvT = transpose3(vv, dims, perm); vwT = transpose3(vw, dims, perm)
            muscl_flux_line!(fd, fs1, fs2, fs3, fe, rhoT, eintT, vuT, vvT, vwT;
                             ncells = na, nghost = ng, jdim = ntr, gamma = gamma,
                             theta = theta, small_rho = small_rho, fge = fge)
            dDl = _scratch(D, N); dNl = _scratch(D, N); dT1l = _scratch(D, N)
            dT2l = _scratch(D, N); dEl = _scratch(D, N)
            _muscl_div_k!(be)(dDl, dNl, dT1l, dT2l, dEl, fd, fs1, fs2, fs3, fe,
                              na, nfi, ng, dtdx; ndrange = (active, ntr))
            # untranspose-add: density/energy are scalars; momentum rotates back with
            # the SAME cyclic map as the velocities.
            sN, sT1, sT2 = axis == 2 ? (dS2, dS3, dS1) : (dS3, dS1, dS2)
            _untranspose_add_into!(dD,  dDl,  dims, perm)
            _untranspose_add_into!(dE,  dEl,  dims, perm)
            _untranspose_add_into!(sN,  dNl,  dims, perm)
            _untranspose_add_into!(sT1, dT1l, dims, perm)
            _untranspose_add_into!(sT2, dT2l, dims, perm)
            if dual
                dGel = _scratch(D, N)
                _scalar_div_local!(be)(dGel, fge, na, nfi, ng, dtdx; ndrange = (active, ntr))
                _untranspose_add_into!(dGe, dGel, dims, perm)
            end
        end
        KA.synchronize(be)
    end
    return nothing
end

"""
    muscl_step_3d!(D, S1, S2, S3, Tau, dims, ng; dt, gamma, theta=1.5, dx=1.0,
                   small_rho=1e-10)

One unsplit SSP-RK2 MUSCL timestep on the conserved state (`D`=ρ, `S1,S2,S3`=ρv,
`Tau`=ρ·etot), mutated in place. `dims=(nx,ny,nz)` flat column-major, `ng` ghost
zones each side; the active region per axis is `ng+1 .. dims[axis]-ng`. This is
Enzo's HydroMethod=3 default (HLL Riemann + PLM minmod-θ reconstruction, RK2).

Wrap an evolution/benchmark loop in [`with_pool`](@ref) for the GPU-allocation win
(the per-axis temporaries recycle across steps); the conserved/prim arrays the
driver itself holds live outside the pool so they survive the per-axis resets.

Passing `ge` (gas-energy `ρ·eint`) turns on the dual-energy formalism (see
[`muscl_hancock_step_3d!`](@ref)); `ge` is advected through both RK stages and
synced at step end.
"""
function muscl_step_3d!(D, S1, S2, S3, Tau, dims::NTuple{3,Int}, ng::Int;
                        dt::Real, gamma::Real, theta::Real = 1.5, dx::Real = 1.0,
                        small_rho::Real = 1e-10, bc! = nothing, ge = nothing, eta1::Real = 1e-3)
    be = KA.get_backend(D); T = eltype(D); N = length(D); dual = ge !== nothing
    half = T(0.5)
    bcfill!() = bc! === nothing ? nothing :
        (dual ? bc!(D, S1, S2, S3, Tau, ge) : bc!(D, S1, S2, S3, Tau))
    # ONE reset recycles the previous step's whole working set; every array below
    # (and inside _muscl_L!) is then drawn from the pool with no intra-step reset,
    # so there is no allocation after warm-up and no dangling-buffer hazard.
    KA.synchronize(be); _pool_reset!()
    sc() = _scratch(D, N; zero = false)
    D0  = sc(); S10 = sc(); S20 = sc(); S30 = sc(); Tau0 = sc()
    dD  = sc(); dS1 = sc(); dS2 = sc(); dS3 = sc(); dE   = sc()
    prim = (sc(), sc(), sc(), sc(), sc())
    Ge0 = dual ? sc() : nothing; dGe = dual ? sc() : nothing
    copyto!(D0, D); copyto!(S10, S1); copyto!(S20, S2); copyto!(S30, S3); copyto!(Tau0, Tau)
    dual && copyto!(Ge0, ge)

    L!(D_, S1_, S2_, S3_, Tau_) =
        _muscl_L!(dD, dS1, dS2, dS3, dE, prim, D_, S1_, S2_, S3_, Tau_,
                  dims, ng, dt, gamma, theta, dx, small_rho; ge = ge, dGe = dGe, eta1 = eta1)

    # stage 1: U1 = U0 + dU(U0)   (in place; D…Tau currently hold U0)
    bcfill!()                                  # ghost zones for U0
    L!(D, S1, S2, S3, Tau)
    @. D   = D0   + dD
    @. S1  = S10  + dS1
    @. S2  = S20  + dS2
    @. S3  = S30  + dS3
    @. Tau = Tau0 + dE
    dual && (@. ge = Ge0 + dGe)
    KA.synchronize(be)

    # stage 2: Unew = ½(U0 + U1 + dU(U1))   (D…Tau currently hold U1)
    bcfill!()                                  # ghost zones for U1
    L!(D, S1, S2, S3, Tau)
    @. D   = half * (D0   + D   + dD)
    @. S1  = half * (S10  + S1  + dS1)
    @. S2  = half * (S20  + S2  + dS2)
    @. S3  = half * (S30  + S3  + dS3)
    @. Tau = half * (Tau0 + Tau + dE)
    dual && (@. ge = half * (Ge0 + ge + dGe))
    # DEF reset (once per step)
    dual && dual_energy_sync!(D, S1, S2, S3, Tau, ge; gamma = gamma, eta1 = eta1, small_rho = small_rho)
    KA.synchronize(be)
    return nothing
end

# ══ MUSCL-Hancock: dimensionally-split, 2nd-order, 3 sweeps/step (≈ PPM cost) ══
# Same conserved set as muscl_step_3d!, but instead of unsplit RK2 (2 stages × 3
# axes = 6 sweeps) this is a Strang directional split (3 sweeps): each axis does
# ONE MUSCL-Hancock flux solve and updates the conserved state in place, exactly
# like the PPM grid sweep. The Hancock ½-step predictor (folded into the flux
# kernel) supplies the 2nd-order-in-time accuracy that the unsplit version got from
# the RK2 outer loop — so this matches PPM's sweep count and GPU memory traffic.

export muscl_hancock_step_3d!

# conservative in-place update of one axis-local conserved slab from its interface
# fluxes:  U[cell] −= (F[i+1] − F[i])·dt/dx  over the active region (Sn = normal mom).
@kernel function _cons_update_k!(D, Sn, St1, St2, Tau,
                                 @Const(fd), @Const(fs1), @Const(fs2), @Const(fs3), @Const(fe),
                                 na::Int, nfi::Int, nghost::Int, dtdx)
    gi, gj = @index(Global, NTuple)
    cl = (gj - 1) * na + nghost + gi
    fo = (gj - 1) * nfi + gi
    @inbounds begin
        D[cl]   -= (fd[fo+1]  - fd[fo])  * dtdx
        Sn[cl]  -= (fs1[fo+1] - fs1[fo]) * dtdx
        St1[cl] -= (fs2[fo+1] - fs2[fo]) * dtdx
        St2[cl] -= (fs3[fo+1] - fs3[fo]) * dtdx
        Tau[cl] -= (fe[fo+1]  - fe[fo])  * dtdx
    end
end

# flux RECORDING (for the AMR reflux): scatter a sweep's interface-flux line into a
# transposed slab — Gt[axis-cell ng+gi] = F[interface gi] (the flux through that
# cell's −axis face) — which `_untranspose_into!` then lands in the grid frame. The
# slab must be pre-zeroed (cells outside ng+1 … ng+nfi carry no recorded face).
@kernel function _flux_to_slab!(Gt, @Const(f), na::Int, nfi::Int, nghost::Int)
    gi, gj = @index(Global, NTuple)
    @inbounds Gt[(gj - 1) * na + nghost + gi] = f[(gj - 1) * nfi + gi]
end

# one Hancock sweep along `axis`, mutating the conserved state in place. Mirrors
# the PPM `sweep_axis!`: x is contiguous; y/z transpose the swept axis to the lead
# (conserved momenta rotated to (normal,t1,t2) order, like the velocities), solve,
# update in place, untranspose back.
function _hancock_sweep_axis!(D, S1, S2, S3, Tau, dims::NTuple{3,Int}, ng::Int, axis::Int;
                              dt::Real, gamma::Real, theta::Real, dx::Real, small_rho::Real,
                              recon::Symbol = :plm, coeffs = nothing, ge = nothing, eta1::Real = 1e-3,
                              frec = nothing)
    be = KA.get_backend(D); T = eltype(D); N = length(D)
    na = dims[axis]; ntr = N ÷ na; active = na - 2 * ng; nfi = active + 1
    dtdx = T(dt) / T(dx); cpred = T(dt) / (2 * T(dx)); dual = ge !== nothing
    perm = _axis_perm(axis)
    # conserved momenta in cyclic (normal, t1, t2) role for this axis
    Sn, St1, St2 = axis == 1 ? (S1, S2, S3) : axis == 2 ? (S2, S3, S1) : (S3, S1, S2)
    # record this sweep's grid-frame face fluxes into frec[axis] (6 fields D,S1,S2,S3,
    # E,Ge in GRID order): the momentum fluxes rotate back like the momenta.
    function record_fluxes!()
        frec === nothing && return
        fa = frec[axis]; nrm = axis; t1 = axis % 3 + 1; t2 = t1 % 3 + 1
        slab = _scratch(D, N)
        rec(comp, tgt) = begin
            fill!(slab, zero(T))
            _flux_to_slab!(be)(slab, comp, na, nfi, ng; ndrange = (nfi, ntr))
            _untranspose_into!(tgt, slab, dims, perm)
        end
        rec(fd, fa[1]); rec(fs1, fa[1+nrm]); rec(fs2, fa[1+t1]); rec(fs3, fa[1+t2]); rec(fe, fa[5])
        dual && rec(fge, fa[6])
    end
    rho = _scratch(D, N; zero = false); eint = _scratch(D, N; zero = false)
    vx = _scratch(D, N; zero = false); vy = _scratch(D, N; zero = false); vz = _scratch(D, N; zero = false)
    fd = _scratch(D, nfi * ntr); fs1 = _scratch(D, nfi * ntr); fs2 = _scratch(D, nfi * ntr)
    fs3 = _scratch(D, nfi * ntr); fe = _scratch(D, nfi * ntr)
    fge = dual ? _scratch(D, nfi * ntr) : nothing
    c2p(r, e, x, y, z, d_, sn, st1, st2, tau, g_) = dual ?
        _cons2prim_dual_k!(be)(r, e, x, y, z, d_, sn, st1, st2, tau, g_, T(gamma), T(eta1), T(small_rho); ndrange = N) :
        _cons2prim_k!(be)(r, e, x, y, z, d_, sn, st1, st2, tau, T(small_rho); ndrange = N)
    fl(rho, eint, vx, vy, vz) =
        muscl_hancock_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                                 ncells = na, nghost = ng, jdim = ntr, gamma = gamma,
                                 theta = theta, cpred = cpred, small_rho = small_rho,
                                 recon = recon, coeffs = coeffs, fge = fge)

    if axis == 1                                   # contiguous — work in place
        c2p(rho, eint, vx, vy, vz, D, Sn, St1, St2, Tau, ge)
        fl(rho, eint, vx, vy, vz)
        record_fluxes!()
        _cons_update_k!(be)(D, Sn, St1, St2, Tau, fd, fs1, fs2, fs3, fe,
                            na, nfi, ng, dtdx; ndrange = (active, ntr))
        dual && _ge_update_k!(be)(ge, fge, na, nfi, ng, dtdx; ndrange = (active, ntr))
    else
        DT = transpose3(D, dims, perm); TauT = transpose3(Tau, dims, perm)
        SnT = transpose3(Sn, dims, perm); St1T = transpose3(St1, dims, perm); St2T = transpose3(St2, dims, perm)
        GeT = dual ? transpose3(ge, dims, perm) : nothing
        c2p(rho, eint, vx, vy, vz, DT, SnT, St1T, St2T, TauT, GeT)
        fl(rho, eint, vx, vy, vz)
        record_fluxes!()
        _cons_update_k!(be)(DT, SnT, St1T, St2T, TauT, fd, fs1, fs2, fs3, fe,
                            na, nfi, ng, dtdx; ndrange = (active, ntr))
        dual && _ge_update_k!(be)(GeT, fge, na, nfi, ng, dtdx; ndrange = (active, ntr))
        # scatter the updated slabs back into the original-layout arrays (one pass each)
        _untranspose_into!(D, DT, dims, perm);    _untranspose_into!(Tau, TauT, dims, perm)
        _untranspose_into!(Sn, SnT, dims, perm);  _untranspose_into!(St1, St1T, dims, perm)
        _untranspose_into!(St2, St2T, dims, perm)
        dual && _untranspose_into!(ge, GeT, dims, perm)
    end
    return nothing
end

"""
    muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, ng;
                           dt, gamma, theta=1.5, dx=1.0, order=(1,2,3), small_rho=1e-10)

One 2nd-order MUSCL-Hancock timestep on the conserved state, dimensionally split:
three in-place directional sweeps in `order` (alternate `(1,2,3)`/`(3,2,1)` across
steps for Strang accuracy), each a single HLL flux solve with the Hancock ½-step
predictor. Same physics class as [`muscl_step_3d!`](@ref) (PLM+HLL) but **3
sweeps/step instead of 6**, to match the PPM grid driver's cost on the GPU.

`recon` selects the spatial reconstruction: `:plm` (default, minmod-θ piecewise
linear) or `:ppm` (monotonized piecewise-parabolic, via the certified
`_iv_recon_cell`; sharper, needs `ng ≥ 3`).

Passing `ge` (the gas-energy conserved field, `ρ·eint`) turns on the DUAL-ENERGY
FORMALISM (Enzo hydro_rk style): `ge` is advected alongside the state and the
internal energy is selected per cell (η₁ controls the switch), giving an accurate
pressure at high Mach where `τ/ρ − ½|v|²` cancels. Mutates the state (and `ge`) in
place; wrap a loop in [`with_pool`](@ref) for the allocation win.
"""
function muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims::NTuple{3,Int}, ng::Int;
                                dt::Real, gamma::Real, theta::Real = 1.5, dx::Real = 1.0,
                                order::NTuple{3,Int} = (1, 2, 3), small_rho::Real = 1e-10,
                                bc! = nothing, recon::Symbol = :plm, ge = nothing, eta1::Real = 1e-3,
                                fluxrec = nothing)
    be = KA.get_backend(D); T = eltype(D)
    recon === :ppm && ng < 3 && error("muscl_hancock_step_3d!: recon=:ppm needs ng ≥ 3 (got $ng)")
    # uniform-grid PPM coefficients (the dx-independent limit of _ie_geom1!/_ie_geom2!:
    # c1=c2=c3=c4=½, c5=1/6, c6=−1/6); constant ⇒ one set reused for all axes/steps.
    # NON-pooled (they must survive the per-sweep _pool_reset!).
    coeffs = nothing
    if recon === :ppm
        nmax = maximum(dims)
        mk(v) = (a = similar(D, nmax); fill!(a, T(v)); a)
        coeffs = (mk(0.5), mk(0.5), mk(0.5), mk(0.5), mk(1//6), mk(-1//6))
    end
    bcfill!() = bc! === nothing ? nothing :
        (ge === nothing ? bc!(D, S1, S2, S3, Tau) : bc!(D, S1, S2, S3, Tau, ge))
    KA.synchronize(be); _pool_reset!()
    for axis in order
        KA.synchronize(be); _pool_reset!()
        # directional split needs the ghost zones consistent with the state the
        # PREVIOUS sweep just updated, so refill BCs before each sweep.
        bcfill!()
        _hancock_sweep_axis!(D, S1, S2, S3, Tau, dims, ng, axis;
                             dt = dt, gamma = gamma, theta = theta, dx = dx, small_rho = small_rho,
                             recon = recon, coeffs = coeffs, ge = ge, eta1 = eta1, frec = fluxrec)
    end
    # DEF reset: re-sync the gas energy and total energy once per step.
    ge === nothing || dual_energy_sync!(D, S1, S2, S3, Tau, ge; gamma = gamma, eta1 = eta1, small_rho = small_rho)
    KA.synchronize(be)
    return nothing
end

# ── primitive ↔ conserved helpers (host-side IC staging / diagnostics) ────────
"`prim_to_cons!(D,S1,S2,S3,Tau, rho,vx,vy,vz,etot)` — conserved from primitives."
function prim_to_cons!(D, S1, S2, S3, Tau, rho, vx, vy, vz, etot)
    @. D   = rho
    @. S1  = rho * vx
    @. S2  = rho * vy
    @. S3  = rho * vz
    @. Tau = rho * etot
    return nothing
end

"`cons_to_prim!(rho,vx,vy,vz,eint, D,S1,S2,S3,Tau; gamma)` — primitives from conserved."
function cons_to_prim!(rho, vx, vy, vz, eint, D, S1, S2, S3, Tau)
    @. rho  = D
    @. vx   = S1 / D
    @. vy   = S2 / D
    @. vz   = S3 / D
    @. eint = Tau / D - oftype(D[1], 0.5) * ((S1 / D)^2 + (S2 / D)^2 + (S3 / D)^2)
    return nothing
end

"`total_field(f, dims, ng, dx)` — Σ f·dV over the active (non-ghost) interior."
function total_field(f, dims::NTuple{3,Int}, ng::Int, dx::Real)
    nx, ny, nz = dims
    h = to_host(f); T = eltype(h); dV = T(dx)^3; s = zero(T)
    @inbounds for k in (ng + 1):(nz - ng), j in (ng + 1):(ny - ng), i in (ng + 1):(nx - ng)
        s += h[i + nx * (j - 1) + nx * ny * (k - 1)]
    end
    return s * dV
end

# ── periodic boundary conditions + CFL timestep (for actual evolution runs) ────
export fill_periodic!, max_wavespeed

# Each ghost cell copies the periodic-image INTERIOR cell (the source is always
# interior — every out-of-range axis wraps inward by the interior width — so the
# in-place write is race-free: interior cells are never written here).
@kernel function _fill_periodic_k!(f, nx::Int, ny::Int, nz::Int, ng::Int)
    g = @index(Global, Linear)
    i = (g - 1) % nx + 1
    t = (g - 1) ÷ nx
    j = t % ny + 1
    k = t ÷ ny + 1
    nix = nx - 2ng; niy = ny - 2ng; niz = nz - 2ng
    ghost = i <= ng || i > nx - ng || j <= ng || j > ny - ng || k <= ng || k > nz - ng
    if ghost
        si = i <= ng ? i + nix : i > nx - ng ? i - nix : i
        sj = j <= ng ? j + niy : j > ny - ng ? j - niy : j
        sk = k <= ng ? k + niz : k > nz - ng ? k - niz : k
        @inbounds f[g] = f[si + nx * (sj - 1) + nx * ny * (sk - 1)]
    end
end

"`fill_periodic!(dims, ng, fields...)` — refill ghost zones of each field by periodic wrap."
function fill_periodic!(dims::NTuple{3,Int}, ng::Int, fields...)
    nx, ny, nz = dims
    for f in fields
        be = KA.get_backend(f)
        _fill_periodic_k!(be)(f, nx, ny, nz, ng; ndrange = length(f))
    end
    isempty(fields) || KA.synchronize(KA.get_backend(fields[1]))
    return nothing
end

@kernel function _wavespeed_k!(s, @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(Tau), gamma)
    i = @index(Global, Linear)
    T = eltype(s)
    @inbounds begin
        d = D[i]; u = S1[i] / d; v = S2[i] / d; w = S3[i] / d
        eint = Tau[i] / d - T(0.5) * (u * u + v * v + w * w)
        cs = sqrt(max(gamma * (gamma - one(T)) * eint, zero(T)))     # √(γ p/ρ), p=(γ−1)ρ·eint
        s[i] = cs + max(abs(u), max(abs(v), abs(w)))
    end
end

"""
    max_wavespeed(scratch, D, S1, S2, S3, Tau; gamma) -> Float

Maximum signal speed `|v|_∞ + c_s` over the grid (for a CFL timestep
`dt = Courant·dx / max_wavespeed`). `scratch` is a full-grid work array.
"""
function max_wavespeed(scratch, D, S1, S2, S3, Tau; gamma::Real)
    be = KA.get_backend(D); T = eltype(D)
    _wavespeed_k!(be)(scratch, D, S1, S2, S3, Tau, T(gamma); ndrange = length(D))
    KA.synchronize(be)
    return maximum(scratch)
end
