# ── PPML 3-D Strang-split STATEFUL driver (Ustyugov+ 2009) ────────────────────
# Assembles the per-cell PPML primitives (ppml.jl) into a 3-D dimensionally-split
# step, mirroring the MUSCL-Hancock grid driver (muscl_grid.jl) for everything it can
# reuse (transpose/rotate machinery, conserved update, dual energy, fluxrec) and
# adding the one thing PPML needs: a PERSISTENT face-state pair (wL,wR) carried per
# cell per axis across steps. The pair is held in a `PpmlState`; each sweep reads it
# (predictor: RGK-limit → flatten → CW84 → characteristic trace), solves HLL with a
# star state, updates the conserved set, and re-derives the pair from the Riemann
# stars (corrector: RGK-limit → flatten). The face pair stores LAB-frame primitives
# (ρ,vx,vy,vz,p); per-axis sweeps rotate the velocity roles exactly like the momenta.
#
# Ghost face pairs are NOT transported — each sweep treats the one boundary-adjacent
# ghost as a DEGENERATE pair (wL=wR=⟨w⟩ ⇒ the trace returns the cell average, a
# locally first-order boundary face). The flux-form update stays conservative; only
# the stored pair at ACTIVE cells is stateful (re-limited against current averages
# each step, so cross-step staleness self-corrects). This is what lets the stateful
# solver run under ghost-based BCs and the live Enzo hierarchy.

export ppml_step_3d!, PpmlState, ppml_alloc_state, ppml_init_state!

# Per-axis persistent face-state pair. `wL[a]`/`wR[a]` each hold the 5 LAB-frame
# primitives (ρ,vx,vy,vz,p) as grid-shaped arrays — NOT pooled (must survive steps).
struct PpmlState{A}
    wL::NTuple{3,NTuple{5,A}}
    wR::NTuple{3,NTuple{5,A}}
    dims::NTuple{3,Int}
    ng::Int
end

"Allocate a `PpmlState` (30 grid-sized arrays shaped like `proto`) for `dims`/`ng`."
function ppml_alloc_state(proto, dims::NTuple{3,Int}, ng::Int)
    N = prod(dims)
    mk() = ntuple(_ -> ntuple(_ -> similar(proto, N), 5), 3)
    return PpmlState(mk(), mk(), dims, ng)
end

# ── cons → primitive (PRESSURE-based: PPML reconstructs p, not eint) ───────────
@kernel function _ppml_c2p_k!(rho, un, ut1, ut2, pr,
                              @Const(D), @Const(Sn), @Const(St1), @Const(St2), @Const(Tau), gm1, small)
    i = @index(Global, Linear); T = eltype(rho)
    @inbounds begin
        d = D[i]; a = Sn[i]/d; b = St1[i]/d; c = St2[i]/d
        rho[i] = d; un[i] = a; ut1[i] = b; ut2[i] = c
        pr[i] = gm1 * d * max(Tau[i]/d - T(0.5)*(a*a + b*b + c*c), small)
    end
end

@kernel function _ppml_c2p_dual_k!(rho, un, ut1, ut2, pr,
                                   @Const(D), @Const(Sn), @Const(St1), @Const(St2),
                                   @Const(Tau), @Const(Ge), gamma, eta1, small)
    i = @index(Global, Linear); T = eltype(rho)
    @inbounds begin
        d = D[i]; a = Sn[i]/d; b = St1[i]/d; c = St2[i]/d; v2 = a*a + b*b + c*c
        rho[i] = d; un[i] = a; ut1[i] = b; ut2[i] = c
        e = _dual_eint(Tau[i], d, Ge[i], v2, gamma, gamma - one(T), eta1, small)
        pr[i] = (gamma - one(T)) * d * e
    end
end

"`ppml_init_state!(st, D,S1,S2,S3,Tau; gamma, ge, …)` — degenerate pair wL=wR=⟨w⟩."
# Each axis's pair is stored in THAT AXIS's transposed frame (swept axis leading,
# velocities in normal/transverse roles) so the sweeps never transpose it. Init thus
# writes the cell-average primitives in each axis's frame (1 transpose set per axis).
function ppml_init_state!(st::PpmlState, D, S1, S2, S3, Tau; gamma::Real,
                          ge = nothing, eta1::Real = 1e-3, small_rho::Real = 1e-10)
    be = KA.get_backend(D); T = eltype(D); N = length(D); gm1 = T(gamma) - one(T)
    dims = st.dims
    for a in 1:3
        perm = _axis_perm(a)
        Sn, St1, St2 = a == 1 ? (S1, S2, S3) : a == 2 ? (S2, S3, S1) : (S3, S1, S2)
        if a == 1
            Dx, Snx, St1x, St2x, Taux, Gex = D, Sn, St1, St2, Tau, ge
        else
            Dx = transpose3(D, dims, perm); Taux = transpose3(Tau, dims, perm)
            Snx = transpose3(Sn, dims, perm); St1x = transpose3(St1, dims, perm); St2x = transpose3(St2, dims, perm)
            Gex = ge === nothing ? nothing : transpose3(ge, dims, perm)
        end
        w = st.wL[a]
        if ge === nothing
            _ppml_c2p_k!(be)(w[1], w[2], w[3], w[4], w[5], Dx, Snx, St1x, St2x, Taux, gm1, T(small_rho); ndrange = N)
        else
            _ppml_c2p_dual_k!(be)(w[1], w[2], w[3], w[4], w[5], Dx, Snx, St1x, St2x, Taux, Gex,
                                  T(gamma), T(eta1), T(small_rho); ndrange = N)
        end
        KA.synchronize(be)
        for k in 1:5
            copyto!(st.wR[a][k], w[k])
        end
    end
    KA.synchronize(be)
    return nothing
end

# ── predictor + characteristic trace (per cell, incl. one boundary ghost/side) ─
# `ic = 1 … active+2`: ic=1 and ic=active+2 are the boundary-adjacent ghosts (degenerate
# face = cell average); ic=2 … active+1 are active cells (full RGK→flatten→CW84→trace).
@kernel function _ppml_predict_k!(rfρ, rfu, rfv, rfw, rfp, lfρ, lfu, lfv, lfw, lfp,
                                  @Const(ρL), @Const(uL), @Const(vL), @Const(wL), @Const(pL),
                                  @Const(ρR), @Const(uR), @Const(vR), @Const(wR), @Const(pR),
                                  @Const(rho), @Const(un), @Const(ut1), @Const(ut2), @Const(pr),
                                  na::Int, nghost::Int, g, dt_dx, gdegen::Int)
    ic, gj = @index(Global, NTuple); T = eltype(rfρ)
    nf2 = na - 2 * nghost + 2                         # = active + 2
    cl = (gj - 1) * na + nghost + ic - 1
    fo = (gj - 1) * nf2 + ic
    @inbounds begin
        sρ = rho[cl]; su = un[cl]; sv = ut1[cl]; sw = ut2[cl]; sp = pr[cl]
        if gdegen == 1 && (ic == 1 || ic == nf2)         # stale ghost pair ⇒ degenerate face
            rfρ[fo] = sρ; rfu[fo] = su; rfv[fo] = sv; rfw[fo] = sw; rfp[fo] = sp
            lfρ[fo] = sρ; lfu[fo] = su; lfv[fo] = sv; lfw[fo] = sw; lfp[fo] = sp
        else
            # RGK characteristic limiter on the stored pair against the cell stencil
            (al, bl, cl_, dl, el, ar, br, cr, dr, er) = _ppml_rgk(
                rho[cl-1], un[cl-1], ut1[cl-1], ut2[cl-1], pr[cl-1],
                sρ, su, sv, sw, sp,
                rho[cl+1], un[cl+1], ut1[cl+1], ut2[cl+1], pr[cl+1],
                ρL[cl], uL[cl], vL[cl], wL[cl], pL[cl],
                ρR[cl], uR[cl], vR[cl], wR[cl], pR[cl], g)
            # CW84 shock flatten (5-pt: pressure + axial velocity)
            ω = _ppml_flatten_omega(pr[cl-2], un[cl-2], pr[cl-1], un[cl-1], sp, su,
                                    pr[cl+1], un[cl+1], pr[cl+2], un[cl+2])
            al = _ppml_flat1(al, sρ, ω); bl = _ppml_flat1(bl, su, ω); cl_ = _ppml_flat1(cl_, sv, ω)
            dl = _ppml_flat1(dl, sw, ω); el = _ppml_flat1(el, sp, ω)
            ar = _ppml_flat1(ar, sρ, ω); br = _ppml_flat1(br, su, ω); cr = _ppml_flat1(cr, sv, ω)
            dr = _ppml_flat1(dr, sw, ω); er = _ppml_flat1(er, sp, ω)
            # CW84 monotonize (+ ρ/p positivity guard)
            (al, bl, cl_, dl, el, ar, br, cr, dr, er) =
                _ppml_monotonize_all(al, bl, cl_, dl, el, sρ, su, sv, sw, sp, ar, br, cr, dr, er)
            # characteristic trace to t+dt/2 at the +axis (right) and −axis (left) faces
            (Rρ, Ru, Rv, Rw, Rp) = _ppml_face_right(al, bl, cl_, dl, el, sρ, su, sv, sw, sp, ar, br, cr, dr, er, dt_dx, g)
            (Lρ, Lu, Lv, Lw, Lp) = _ppml_face_left(al, bl, cl_, dl, el, sρ, su, sv, sw, sp, ar, br, cr, dr, er, dt_dx, g)
            rfρ[fo] = Rρ; rfu[fo] = Ru; rfv[fo] = Rv; rfw[fo] = Rw; rfp[fo] = Rp
            lfρ[fo] = Lρ; lfu[fo] = Lu; lfv[fo] = Lv; lfw[fo] = Lw; lfp[fo] = Lp
        end
    end
end

# ── HLL flux + star state per interface (gi = 1 … active+1) ────────────────────
@kernel function _ppml_riemann_k!(fd, fs1, fs2, fs3, fe, fge, sρ, su, sv, sw, sp,
                                  @Const(rfρ), @Const(rfu), @Const(rfv), @Const(rfw), @Const(rfp),
                                  @Const(lfρ), @Const(lfu), @Const(lfv), @Const(lfw), @Const(lfp),
                                  nfi::Int, nf2::Int, g, gm1, idual::Int)
    gi, gj = @index(Global, NTuple)
    fo = (gj - 1) * nfi + gi
    foL = (gj - 1) * nf2 + gi          # +face of the left cell (ic = gi)
    foR = foL + 1                      # −face of the right cell (ic = gi+1)
    @inbounds begin
        (F1, F2, F3, F4, F5, F6, ρs, us, vs, ws, ps) = _ppml_hll(
            rfρ[foL], rfu[foL], rfv[foL], rfw[foL], rfp[foL],
            lfρ[foR], lfu[foR], lfv[foR], lfw[foR], lfp[foR], g, gm1)
        fd[fo] = F1; fs1[fo] = F2; fs2[fo] = F3; fs3[fo] = F4; fe[fo] = F5
        idual == 1 && (fge[fo] = F6)
        sρ[fo] = ρs; su[fo] = us; sv[fo] = vs; sw[fo] = ws; sp[fo] = ps
    end
end

# ── corrector: re-derive the persistent face pair from the Riemann stars ───────
# per ACTIVE cell (ica = 1 … active ⇒ cl = ng+ica): seed (wL,wR) from the bracketing
# star states, RGK-limit against the POST-update averages, flatten, store.
@kernel function _ppml_correct_k!(ρL, uL, vL, wL, pL, ρR, uR, vR, wR, pR,
                                  @Const(rho), @Const(un), @Const(ut1), @Const(ut2), @Const(pr),
                                  @Const(sρ), @Const(su), @Const(sv), @Const(sw), @Const(sp),
                                  na::Int, nfi::Int, nghost::Int, g)
    ica, gj = @index(Global, NTuple); T = eltype(ρL)
    cl = (gj - 1) * na + nghost + ica
    foM = (gj - 1) * nfi + ica          # −axis interface (between cl-1, cl)
    foP = foM + 1                       # +axis interface (between cl, cl+1)
    @inbounds begin
        sc_ρ = rho[cl]; sc_u = un[cl]; sc_v = ut1[cl]; sc_w = ut2[cl]; sc_p = pr[cl]
        (al, bl, cl_, dl, el, ar, br, cr, dr, er) = _ppml_rgk(
            rho[cl-1], un[cl-1], ut1[cl-1], ut2[cl-1], pr[cl-1],
            sc_ρ, sc_u, sc_v, sc_w, sc_p,
            rho[cl+1], un[cl+1], ut1[cl+1], ut2[cl+1], pr[cl+1],
            sρ[foM], su[foM], sv[foM], sw[foM], sp[foM],
            sρ[foP], su[foP], sv[foP], sw[foP], sp[foP], g)
        ω = _ppml_flatten_omega(pr[cl-2], un[cl-2], pr[cl-1], un[cl-1], sc_p, sc_u,
                                pr[cl+1], un[cl+1], pr[cl+2], un[cl+2])
        ρL[cl] = _ppml_flat1(al, sc_ρ, ω); uL[cl] = _ppml_flat1(bl, sc_u, ω)
        vL[cl] = _ppml_flat1(cl_, sc_v, ω); wL[cl] = _ppml_flat1(dl, sc_w, ω); pL[cl] = _ppml_flat1(el, sc_p, ω)
        ρR[cl] = _ppml_flat1(ar, sc_ρ, ω); uR[cl] = _ppml_flat1(br, sc_u, ω)
        vR[cl] = _ppml_flat1(cr, sc_v, ω); wR[cl] = _ppml_flat1(dr, sc_w, ω); pR[cl] = _ppml_flat1(er, sc_p, ω)
    end
end

# one PPML sweep along `axis`, mutating the conserved state + the axis's face pair.
function _ppml_sweep_axis!(D, S1, S2, S3, Tau, st::PpmlState, dims::NTuple{3,Int}, ng::Int, axis::Int;
                           dt::Real, gamma::Real, dx::Real, small_rho::Real,
                           ge = nothing, eta1::Real = 1e-3, frec = nothing, face_periodic::Bool = false)
    be = KA.get_backend(D); T = eltype(D); N = length(D)
    na = dims[axis]; ntr = N ÷ na; active = na - 2 * ng; nfi = active + 1; nf2 = active + 2
    dtdx = T(dt) / T(dx); g = T(gamma); gm1 = g - one(T); dual = ge !== nothing
    perm = _axis_perm(axis); pdims = (dims[perm[1]], dims[perm[2]], dims[perm[3]])
    Sn, St1, St2 = axis == 1 ? (S1, S2, S3) : axis == 2 ? (S2, S3, S1) : (S3, S1, S2)
    # the axis's face pair is stored ALREADY in this axis's transposed frame, as
    # (ρ, vn, vt1, vt2, p) — so the sweep never transposes/rotates it (the big save).
    fL = st.wL[axis]; fR = st.wR[axis]
    # periodic state ⇒ wrap the stored face pair into the ghosts (in the transposed
    # frame, pdims) so the seam reconstruction — and the two seam fluxes — is bit-
    # identical ⇒ conservative.
    face_periodic && fill_periodic!(pdims, ng, fL..., fR...)
    ρLx, uLx, vLx, wLx, pLx = fL[1], fL[2], fL[3], fL[4], fL[5]
    ρRx, uRx, vRx, wRx, pRx = fR[1], fR[2], fR[3], fR[4], fR[5]

    # scratch (transposed frame): cell-average prims, traced faces, fluxes, stars
    rho = _scratch(D, N; zero = false); un = _scratch(D, N; zero = false)
    ut1 = _scratch(D, N; zero = false); ut2 = _scratch(D, N; zero = false); pr = _scratch(D, N; zero = false)
    rfρ = _scratch(D, nf2*ntr); rfu = _scratch(D, nf2*ntr); rfv = _scratch(D, nf2*ntr); rfw = _scratch(D, nf2*ntr); rfp = _scratch(D, nf2*ntr)
    lfρ = _scratch(D, nf2*ntr); lfu = _scratch(D, nf2*ntr); lfv = _scratch(D, nf2*ntr); lfw = _scratch(D, nf2*ntr); lfp = _scratch(D, nf2*ntr)
    fd = _scratch(D, nfi*ntr); fs1 = _scratch(D, nfi*ntr); fs2 = _scratch(D, nfi*ntr); fs3 = _scratch(D, nfi*ntr); fe = _scratch(D, nfi*ntr)
    fge = dual ? _scratch(D, nfi*ntr) : nothing
    sρ = _scratch(D, nfi*ntr); su = _scratch(D, nfi*ntr); sv = _scratch(D, nfi*ntr); sw = _scratch(D, nfi*ntr); sp = _scratch(D, nfi*ntr)

    # bring the CONSERVED set into the swept-axis-leading frame (the face pair already
    # lives there — only the lab-frame conserved arrays need transposing).
    if axis == 1
        Dx, Snx, St1x, St2x, Taux, Gex = D, Sn, St1, St2, Tau, ge
    else
        Dx = transpose3(D, dims, perm); Taux = transpose3(Tau, dims, perm)
        Snx = transpose3(Sn, dims, perm); St1x = transpose3(St1, dims, perm); St2x = transpose3(St2, dims, perm)
        Gex = dual ? transpose3(ge, dims, perm) : nothing
    end

    # cons → primitive (pressure) on the pre-update state
    if dual
        _ppml_c2p_dual_k!(be)(rho, un, ut1, ut2, pr, Dx, Snx, St1x, St2x, Taux, Gex, g, T(eta1), T(small_rho); ndrange = N)
    else
        _ppml_c2p_k!(be)(rho, un, ut1, ut2, pr, Dx, Snx, St1x, St2x, Taux, gm1, T(small_rho); ndrange = N)
    end
    # predictor + trace, Riemann flux + star
    _ppml_predict_k!(be)(rfρ, rfu, rfv, rfw, rfp, lfρ, lfu, lfv, lfw, lfp,
                         ρLx, uLx, vLx, wLx, pLx, ρRx, uRx, vRx, wRx, pRx,
                         rho, un, ut1, ut2, pr, na, ng, g, dtdx,
                         face_periodic ? 0 : 1; ndrange = (nf2, ntr))
    _ppml_riemann_k!(be)(fd, fs1, fs2, fs3, fe, dual ? fge : fd, sρ, su, sv, sw, sp,
                         rfρ, rfu, rfv, rfw, rfp, lfρ, lfu, lfv, lfw, lfp,
                         nfi, nf2, g, gm1, dual ? 1 : 0; ndrange = (nfi, ntr))
    # reflux recording (grid-frame face fluxes) — mirrors the Hancock path
    if frec !== nothing
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
    # conservative update + dual-energy advection
    _cons_update_k!(be)(Dx, Snx, St1x, St2x, Taux, fd, fs1, fs2, fs3, fe, na, nfi, ng, dtdx; ndrange = (active, ntr))
    dual && _ge_update_k!(be)(Gex, fge, na, nfi, ng, dtdx; ndrange = (active, ntr))
    # corrector: post-update averages → re-derive the persistent pair from the stars
    if dual
        _ppml_c2p_dual_k!(be)(rho, un, ut1, ut2, pr, Dx, Snx, St1x, St2x, Taux, Gex, g, T(eta1), T(small_rho); ndrange = N)
    else
        _ppml_c2p_k!(be)(rho, un, ut1, ut2, pr, Dx, Snx, St1x, St2x, Taux, gm1, T(small_rho); ndrange = N)
    end
    _ppml_correct_k!(be)(ρLx, uLx, vLx, wLx, pLx, ρRx, uRx, vRx, wRx, pRx,
                         rho, un, ut1, ut2, pr, sρ, su, sv, sw, sp, na, nfi, ng, g; ndrange = (active, ntr))

    # scatter the CONSERVED set back to the original layout (the corrector wrote the
    # face pair in place, in its persistent transposed frame — nothing to untranspose).
    if axis != 1
        _untranspose_into!(D, Dx, dims, perm);    _untranspose_into!(Tau, Taux, dims, perm)
        _untranspose_into!(Sn, Snx, dims, perm);  _untranspose_into!(St1, St1x, dims, perm); _untranspose_into!(St2, St2x, dims, perm)
        dual && _untranspose_into!(ge, Gex, dims, perm)
    end
    return nothing
end

"""
    ppml_step_3d!(D, S1, S2, S3, Tau, dims, ng; state, dt, gamma, dx=1.0,
                  order=(1,2,3), small_rho=1e-10, bc!=nothing, ge=nothing,
                  eta1=1e-3, fluxrec=nothing)

One PPML (Piecewise-Parabolic Method on a Local stencil, Ustyugov+ 2009) timestep on
the conserved state, dimensionally split: three in-place directional sweeps in `order`.
Each sweep reads the persistent face pair in `state` (a [`PpmlState`](@ref), allocate
with [`ppml_alloc_state`](@ref) + initialise with [`ppml_init_state!`](@ref)), applies
the RGK characteristic limiter + CW84 monotonize + shock flatten + the
characteristic-traced half-step predictor, solves HLL with a star state, updates the
conserved set, and re-derives the pair from the Riemann stars.

`recon`-free (the parabola is anchored on the stored pair). Needs **`ng ≥ 3`** (5-point
flatten stencil). `ge` (ρ·eint) turns on the dual-energy formalism; `fluxrec` records
grid-frame interface fluxes for the AMR reflux (same convention as
[`muscl_hancock_step_3d!`](@ref)). Mutates the state, `ge`, and `state` in place.
"""
function ppml_step_3d!(D, S1, S2, S3, Tau, dims::NTuple{3,Int}, ng::Int;
                       state::PpmlState, dt::Real, gamma::Real, dx::Real = 1.0,
                       order::NTuple{3,Int} = (1, 2, 3), small_rho::Real = 1e-10,
                       bc! = nothing, ge = nothing, eta1::Real = 1e-3, fluxrec = nothing,
                       face_periodic::Bool = false)
    be = KA.get_backend(D)
    ng < 3 && error("ppml_step_3d!: needs ng ≥ 3 (5-point flatten stencil), got $ng")
    # `face_periodic` = periodic standalone mode: with no explicit `bc!`, also wrap the
    # CONSERVED ghosts each sweep (the seam flux is single-valued only when both the
    # cell averages AND the stored face pair are periodic-consistent ⇒ conservation).
    if face_periodic && bc! === nothing
        bc! = (fs...) -> fill_periodic!(dims, ng, fs...)
    end
    bcfill!() = bc! === nothing ? nothing :
        (ge === nothing ? bc!(D, S1, S2, S3, Tau) : bc!(D, S1, S2, S3, Tau, ge))
    KA.synchronize(be); _pool_reset!()
    for axis in order
        KA.synchronize(be); _pool_reset!()
        bcfill!()
        _ppml_sweep_axis!(D, S1, S2, S3, Tau, state, dims, ng, axis;
                          dt = dt, gamma = gamma, dx = dx, small_rho = small_rho,
                          ge = ge, eta1 = eta1, frec = fluxrec, face_periodic = face_periodic)
    end
    ge === nothing || dual_energy_sync!(D, S1, S2, S3, Tau, ge; gamma = gamma, eta1 = eta1, small_rho = small_rho)
    KA.synchronize(be)
    return nothing
end
