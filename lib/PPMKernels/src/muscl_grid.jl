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

# cons (D,S1,S2,S3,τ) → prim (ρ,eint,vx,vy,vz) over the whole grid
@kernel function _cons2prim_k!(rho, eint, vx, vy, vz,
                               @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(Tau))
    i = @index(Global, Linear)
    T = eltype(rho)
    @inbounds begin
        d = D[i]; u = S1[i] / d; v = S2[i] / d; w = S3[i] / d
        rho[i] = d; vx[i] = u; vy[i] = v; vz[i] = w
        eint[i] = Tau[i] / d - T(0.5) * (u * u + v * v + w * w)
    end
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
function _muscl_L!(dD, dS1, dS2, dS3, dE, prim,
                   D, S1, S2, S3, Tau, dims::NTuple{3,Int}, ng::Int,
                   dt::Real, gamma::Real, theta::Real, dx::Real, small_rho::Real)
    be = KA.get_backend(D); T = eltype(D); N = length(D)
    rho, eint, vx, vy, vz = prim
    _cons2prim_k!(be)(rho, eint, vx, vy, vz, D, S1, S2, S3, Tau; ndrange = N)
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

        if axis == 1                          # contiguous — no transpose, add in place
            muscl_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vu, vv, vw;
                             ncells = na, nghost = ng, jdim = ntr, gamma = gamma,
                             theta = theta, small_rho = small_rho)
            _muscl_div_add_x!(be)(dD, dS1, dS2, dS3, dE, fd, fs1, fs2, fs3, fe,
                                  na, nfi, ng, dtdx; ndrange = (active, ntr))
        else
            perm = _axis_perm(axis)
            rhoT = transpose3(rho, dims, perm); eintT = transpose3(eint, dims, perm)
            vuT  = transpose3(vu, dims, perm);  vvT = transpose3(vv, dims, perm); vwT = transpose3(vw, dims, perm)
            muscl_flux_line!(fd, fs1, fs2, fs3, fe, rhoT, eintT, vuT, vvT, vwT;
                             ncells = na, nghost = ng, jdim = ntr, gamma = gamma,
                             theta = theta, small_rho = small_rho)
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
"""
function muscl_step_3d!(D, S1, S2, S3, Tau, dims::NTuple{3,Int}, ng::Int;
                        dt::Real, gamma::Real, theta::Real = 1.5, dx::Real = 1.0,
                        small_rho::Real = 1e-10)
    be = KA.get_backend(D); T = eltype(D); N = length(D)
    half = T(0.5)
    # ONE reset recycles the previous step's whole working set; every array below
    # (and inside _muscl_L!) is then drawn from the pool with no intra-step reset,
    # so there is no allocation after warm-up and no dangling-buffer hazard.
    KA.synchronize(be); _pool_reset!()
    sc() = _scratch(D, N; zero = false)
    D0  = sc(); S10 = sc(); S20 = sc(); S30 = sc(); Tau0 = sc()
    dD  = sc(); dS1 = sc(); dS2 = sc(); dS3 = sc(); dE   = sc()
    prim = (sc(), sc(), sc(), sc(), sc())
    copyto!(D0, D); copyto!(S10, S1); copyto!(S20, S2); copyto!(S30, S3); copyto!(Tau0, Tau)

    L!(D_, S1_, S2_, S3_, Tau_) =
        _muscl_L!(dD, dS1, dS2, dS3, dE, prim, D_, S1_, S2_, S3_, Tau_,
                  dims, ng, dt, gamma, theta, dx, small_rho)

    # stage 1: U1 = U0 + dU(U0)   (in place; D…Tau currently hold U0)
    L!(D, S1, S2, S3, Tau)
    @. D   = D0   + dD
    @. S1  = S10  + dS1
    @. S2  = S20  + dS2
    @. S3  = S30  + dS3
    @. Tau = Tau0 + dE
    KA.synchronize(be)

    # stage 2: Unew = ½(U0 + U1 + dU(U1))   (D…Tau currently hold U1)
    L!(D, S1, S2, S3, Tau)
    @. D   = half * (D0   + D   + dD)
    @. S1  = half * (S10  + S1  + dS1)
    @. S2  = half * (S20  + S2  + dS2)
    @. S3  = half * (S30  + S3  + dS3)
    @. Tau = half * (Tau0 + Tau + dE)
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

# one Hancock sweep along `axis`, mutating the conserved state in place. Mirrors
# the PPM `sweep_axis!`: x is contiguous; y/z transpose the swept axis to the lead
# (conserved momenta rotated to (normal,t1,t2) order, like the velocities), solve,
# update in place, untranspose back.
function _hancock_sweep_axis!(D, S1, S2, S3, Tau, dims::NTuple{3,Int}, ng::Int, axis::Int;
                              dt::Real, gamma::Real, theta::Real, dx::Real, small_rho::Real)
    be = KA.get_backend(D); T = eltype(D); N = length(D)
    na = dims[axis]; ntr = N ÷ na; active = na - 2 * ng; nfi = active + 1
    dtdx = T(dt) / T(dx); cpred = T(dt) / (2 * T(dx))
    # conserved momenta in cyclic (normal, t1, t2) role for this axis
    Sn, St1, St2 = axis == 1 ? (S1, S2, S3) : axis == 2 ? (S2, S3, S1) : (S3, S1, S2)
    rho = _scratch(D, N; zero = false); eint = _scratch(D, N; zero = false)
    vx = _scratch(D, N; zero = false); vy = _scratch(D, N; zero = false); vz = _scratch(D, N; zero = false)
    fd = _scratch(D, nfi * ntr); fs1 = _scratch(D, nfi * ntr); fs2 = _scratch(D, nfi * ntr)
    fs3 = _scratch(D, nfi * ntr); fe = _scratch(D, nfi * ntr)

    if axis == 1                                   # contiguous — work in place
        _cons2prim_k!(be)(rho, eint, vx, vy, vz, D, Sn, St1, St2, Tau; ndrange = N)
        muscl_hancock_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                                 ncells = na, nghost = ng, jdim = ntr, gamma = gamma,
                                 theta = theta, cpred = cpred, small_rho = small_rho)
        _cons_update_k!(be)(D, Sn, St1, St2, Tau, fd, fs1, fs2, fs3, fe,
                            na, nfi, ng, dtdx; ndrange = (active, ntr))
    else
        perm = _axis_perm(axis)
        DT = transpose3(D, dims, perm); TauT = transpose3(Tau, dims, perm)
        SnT = transpose3(Sn, dims, perm); St1T = transpose3(St1, dims, perm); St2T = transpose3(St2, dims, perm)
        _cons2prim_k!(be)(rho, eint, vx, vy, vz, DT, SnT, St1T, St2T, TauT; ndrange = N)
        muscl_hancock_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                                 ncells = na, nghost = ng, jdim = ntr, gamma = gamma,
                                 theta = theta, cpred = cpred, small_rho = small_rho)
        _cons_update_k!(be)(DT, SnT, St1T, St2T, TauT, fd, fs1, fs2, fs3, fe,
                            na, nfi, ng, dtdx; ndrange = (active, ntr))
        # scatter the updated slabs back into the original-layout arrays (one pass each)
        _untranspose_into!(D, DT, dims, perm);    _untranspose_into!(Tau, TauT, dims, perm)
        _untranspose_into!(Sn, SnT, dims, perm);  _untranspose_into!(St1, St1T, dims, perm)
        _untranspose_into!(St2, St2T, dims, perm)
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
sweeps/step instead of 6**, to match the PPM grid driver's cost on the GPU. Mutates
the state in place; wrap a loop in [`with_pool`](@ref) for the allocation win.
"""
function muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims::NTuple{3,Int}, ng::Int;
                                dt::Real, gamma::Real, theta::Real = 1.5, dx::Real = 1.0,
                                order::NTuple{3,Int} = (1, 2, 3), small_rho::Real = 1e-10)
    be = KA.get_backend(D)
    KA.synchronize(be); _pool_reset!()
    for axis in order
        KA.synchronize(be); _pool_reset!()
        _hancock_sweep_axis!(D, S1, S2, S3, Tau, dims, ng, axis;
                             dt = dt, gamma = gamma, theta = theta, dx = dx, small_rho = small_rho)
    end
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
