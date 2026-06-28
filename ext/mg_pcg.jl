# Geometric-multigrid-preconditioned CG for the composite Poisson solve (the
# RAMSES/Enzo-style fast multigrid). The OUTER solver is still CG on the exact
# composite operator A = −∇²·V (so conservation/parity are untouched); a cell-centered
# geometric V-cycle on the uniform BASE grid is the preconditioner, which collapses the
# CG iteration count (~58 → a handful) by killing the low-frequency / box-scale modes
# that unpreconditioned CG resolves only at ~√κ rate.
#
# Mapping: every composite leaf maps (by its center) to a base-grid cell (i,j,k). For a
# uniform mesh this is a bijection; under AMR many fine leaves share one base cell. The
# preconditioner restricts the composite residual to the base grid, runs ONE V-cycle of
# the cell-centered Laplacian (damped-Jacobi smooth + full-weighting restrict + trilinear
# prolong, periodic), and prolongs the correction back; a complementary diagonal term
# (1/diag(A)) keeps M SPD and handles the within-base-cell fine scale under AMR.
#
# This file is `include`d into VespaKAGravityExt, so it shares its `using`s + `const KA`.

using MeshInterface: cell_center, cell_width, domain, max_level

# ── cell-centered periodic V-cycle kernels (validated: ~0.27 residual/cycle) ──
# damped Jacobi for -∇²: φ ← (1-ω)φ + ω(Σφ_nb + h²·b)/6   (cubic n, spacing h)
@kernel function _mgk_jacobi!(phi, @Const(phiold), @Const(b), h2, ω, n)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ip = i==n ? 1 : i+1; im = i==1 ? n : i-1
        jp = j==n ? 1 : j+1; jm = j==1 ? n : j-1
        kp = k==n ? 1 : k+1; km = k==1 ? n : k-1
        s = phiold[im,j,k]+phiold[ip,j,k]+phiold[i,jm,k]+phiold[i,jp,k]+phiold[i,j,km]+phiold[i,j,kp]
        phi[i,j,k] = (1-ω)*phiold[i,j,k] + ω*(s + h2*b[i,j,k])/6
    end
end
# residual r = b - (-∇²)φ = b - (6φ - Σφ_nb)/h²
@kernel function _mgk_residual!(r, @Const(phi), @Const(b), invh2, n)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ip = i==n ? 1 : i+1; im = i==1 ? n : i-1
        jp = j==n ? 1 : j+1; jm = j==1 ? n : j-1
        kp = k==n ? 1 : k+1; km = k==1 ? n : k-1
        s = phi[im,j,k]+phi[ip,j,k]+phi[i,jm,k]+phi[i,jp,k]+phi[i,j,km]+phi[i,j,kp]
        r[i,j,k] = b[i,j,k] - (6*phi[i,j,k] - s)*invh2
    end
end
# full-weighting restriction (2:1, cell-centered): coarse = mean of 8 fine children
@kernel function _mgk_restrict!(bc, @Const(rf))
    I, J, K = @index(Global, NTuple)
    @inbounds begin
        i=2I-1; j=2J-1; k=2K-1
        bc[I,J,K] = 0.125*(rf[i,j,k]+rf[i+1,j,k]+rf[i,j+1,k]+rf[i+1,j+1,k]+
                           rf[i,j,k+1]+rf[i+1,j,k+1]+rf[i,j+1,k+1]+rf[i+1,j+1,k+1])
    end
end
# trilinear cell-centered prolongation, ADDED into the fine grid (periodic; weights 3/4,1/4)
@kernel function _mgk_prolong_add!(phif, @Const(phic), nc)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ic=(i+1)÷2; jc=(j+1)÷2; kc=(k+1)÷2
        si = isodd(i) ? -1 : 1; sj = isodd(j) ? -1 : 1; sk = isodd(k) ? -1 : 1
        i2=ic+si; i2 = i2<1 ? nc : (i2>nc ? 1 : i2)
        j2=jc+sj; j2 = j2<1 ? nc : (j2>nc ? 1 : j2)
        k2=kc+sk; k2 = k2<1 ? nc : (k2>nc ? 1 : k2)
        v = 0.0
        for (wi,ii) in ((0.75,ic),(0.25,i2)), (wj,jj) in ((0.75,jc),(0.25,j2)), (wk,kk) in ((0.75,kc),(0.25,k2))
            v += wi*wj*wk*phic[ii,jj,kk]
        end
        phif[i,j,k] += v
    end
end

# ── composite ↔ base transfer (linear base index per composite ordinal) ──────
@kernel function _mgk_scatter!(base, @Const(r), @Const(blin))   # base .= Σ over cells; (zero base first)
    i = @index(Global)
    @inbounds KA.@atomic base[blin[i]] += Float64(r[i])
end
@kernel function _mgk_gather!(z, @Const(base), @Const(blin), γ, @Const(rr), @Const(diag))
    i = @index(Global)
    @inbounds z[i] = oftype(z[i], base[blin[i]] + γ * Float64(rr[i]) / diag[i])   # V-cycle + diagonal fine-scale
end

# ── build the base-grid hierarchy + the ordinal→base map (once per mesh) ──────
function _build_mg_cache(sim, grav, st, be)
    b = sim.backend
    dom = domain(b)
    lo = ntuple(d -> Float64(dom[d][1]), 3)
    L  = ntuple(d -> Float64(dom[d][2] - dom[d][1]), 3)
    # base (coarsest) cell width = max leaf width; base resolution N per axis.
    wmax = zeros(3)
    @inbounds for h in st.handles
        w = cell_width(b, h)
        for d in 1:3; wmax[d] = max(wmax[d], Float64(w[d])); end
    end
    Nt = ntuple(d -> max(1, round(Int, L[d] / wmax[d])), 3)
    N = Nt[1]
    @assert all(==(N), Nt) "MG preconditioner assumes a cubic base grid (got $Nt)"
    hbase = L[1] / N
    Vbase = hbase^3
    # ordinal → base linear index (column-major, 1-based) from cell centers
    blin = Vector{Int32}(undef, st.n)
    @inbounds for i in 1:st.n
        c = cell_center(b, st.handles[i])
        ix = clamp(floor(Int, (Float64(c[1]) - lo[1]) / hbase), 0, N-1)
        iy = clamp(floor(Int, (Float64(c[2]) - lo[2]) / hbase), 0, N-1)
        iz = clamp(floor(Int, (Float64(c[3]) - lo[3]) / hbase), 0, N-1)
        blin[i] = Int32(ix + iy*N + iz*N*N + 1)
    end
    # diagonal of the composite operator A = −∇²·V  (row sums of the CSR), for the
    # complementary diagonal preconditioner term.
    diag = Vector{Float64}(undef, st.n)
    @inbounds for c in 1:st.n
        s = 0.0
        for k in st.rowptr[c]:(st.rowptr[c+1]-1); s += st.coefval[k]; end
        diag[c] = s == 0.0 ? 1.0 : s
    end
    # V-cycle level scratch (cubic; coarsen while even and > 4)
    levels = Int[]; nn = N
    while true
        push!(levels, nn)
        (nn > 4 && iseven(nn)) || break
        nn ÷= 2
    end
    lev = Dict{Int,Any}()
    for nl in levels
        lev[nl] = (b = KA.allocate(be, Float64, (nl,nl,nl)),
                   e = KA.allocate(be, Float64, (nl,nl,nl)),
                   r = KA.allocate(be, Float64, (nl,nl,nl)),
                   t = KA.allocate(be, Float64, (nl,nl,nl)))
    end
    return (N = N, hbase = hbase, Vbase = Vbase, levels = levels,
            d_blin = _to_dev(be, blin), d_diag = _to_dev(be, diag), lev = lev,
            uniform = (st.n == N^3))
end

# One V-cycle for (-∇²)y = rhs on a cubic n³ periodic grid (y in `lev[n].e`, rhs in `lev[n].b`).
function _mg_vcycle!(mg, n, h, be; pre=2, post=2)
    ω = 0.8; h2 = h*h; invh2 = 1/h2
    c = mg.lev[n]; phi = c.e; rhs = c.b; tmp = c.t
    for _ in 1:pre
        copyto!(tmp, phi); _mgk_jacobi!(be)(phi, tmp, rhs, h2, ω, n; ndrange=(n,n,n))
    end
    if !(n > 4 && iseven(n))
        for _ in 1:30
            copyto!(tmp, phi); _mgk_jacobi!(be)(phi, tmp, rhs, h2, ω, n; ndrange=(n,n,n))
        end
        return
    end
    _mgk_residual!(be)(c.r, phi, rhs, invh2, n; ndrange=(n,n,n))
    nc = n ÷ 2; cc = mg.lev[nc]
    _mgk_restrict!(be)(cc.b, c.r; ndrange=(nc,nc,nc))
    fill!(cc.e, 0.0)
    _mg_vcycle!(mg, nc, 2h, be; pre=pre, post=post)
    _mgk_prolong_add!(be)(phi, cc.e, nc; ndrange=(n,n,n))
    for _ in 1:post
        copyto!(tmp, phi); _mgk_jacobi!(be)(phi, tmp, rhs, h2, ω, n; ndrange=(n,n,n))
    end
    return
end


# ── composite-level smoothing on the REAL operator A = −∇²·V ──────────────────
# One weighted-Jacobi sweep: x ← x + ω·(rhs − A·x)/diag(A). Unlike the base V-cycle,
# this smoother acts on the actual composite leaf set (refined cells included), so it
# damps the fine-scale error the base grid can't see — the key to AMR robustness.
@kernel function _csr_jacobi!(x, @Const(rhs), @Const(Ax), @Const(diag), ω)
    i = @index(Global)
    @inbounds x[i] = oftype(x[i], Float64(x[i]) + ω*(Float64(rhs[i]) - Float64(Ax[i]))/diag[i])
end

# Composite-smoothed two-level cycle (the shared kernel of BOTH AMR approaches):
# ν1 composite pre-smooths → restrict composite residual to the base → base V-cycle
# (low-frequency / deep-well modes) → prolong correction → ν2 composite post-smooths.
# Improves `x` toward A·x = rhs in place. Coarse correction lands in `c.mc`; residual in
# `c.mr` — both distinct from `x` so it is safe to call with x === c.z (the PCG path).
function _amg_cycle!(x, rhs, c, mg, be; ν1 = 2, ν2 = 2, ω = 0.8)
    for _ in 1:ν1
        _matvec!(c.Ap, x, c, be)
        _csr_jacobi!(be)(x, rhs, c.Ap, mg.d_diag, ω; ndrange = length(x))
    end
    _matvec!(c.Ap, x, c, be); @. c.mr = rhs - c.Ap            # composite residual
    N = mg.N; top = mg.lev[N]
    fill!(top.b, 0.0)
    _mgk_scatter!(be)(vec(top.b), c.mr, mg.d_blin; ndrange = length(c.mr))   # R·r → base
    fill!(top.e, 0.0)
    _mg_vcycle!(mg, N, mg.hbase, be)                          # (−∇²)e = base rhs
    μ = sum(top.e) / length(top.e); @. top.e = (top.e - μ) / mg.Vbase
    _mgk_gather!(be)(c.mc, vec(top.e), mg.d_blin, 0.0, c.mr, mg.d_diag; ndrange = length(c.mc))
    @. x += c.mc                                              # P·(coarse correction)
    for _ in 1:ν2
        _matvec!(c.Ap, x, c, be)
        _csr_jacobi!(be)(x, rhs, c.Ap, mg.d_diag, ω; ndrange = length(x))
    end
    return x
end

# Build (once per mesh) the flat CSR cache + the multigrid hierarchy/transfer maps.
function _mg_ensure_caches!(sim, grav, be)
    T = eltype(grav.bv)
    if grav.ka_cache === nothing
        st = Vespa._build_poisson_ka_struct(sim, grav)
        z() = KA.allocate(be, T, (st.n,))
        grav.ka_cache = (st = st,
                         d_rowptr = _to_dev(be, st.rowptr), d_colidx = _to_dev(be, st.colidx),
                         d_coefval = _to_dev(be, st.coefval), d_vol = _to_dev(be, st.vol),
                         d_glo = _to_dev(be, st.glo), d_ghi = _to_dev(be, st.ghi),
                         d_gcoef = _to_dev(be, st.gcoef),
                         gx = KA.allocate(be, Float64, (st.n,)), gy = KA.allocate(be, Float64, (st.n,)),
                         gz = KA.allocate(be, Float64, (st.n,)), tmp64 = KA.allocate(be, Float64, (st.n,)),
                         phi = z(), r = z(), p = z(), Ap = z(), zz = z(), b = z(),
                         mr = z(), mc = z(),
                         b_host = zeros(T, st.n), phi_host = zeros(T, st.n))
        copyto!(grav.ka_cache.phi, Vespa._gather_field(st, grav.phiv))
    end
    grav.mg_cache === nothing && (grav.mg_cache = _build_mg_cache(sim, grav, grav.ka_cache.st, be))
    return grav.ka_cache, grav.ka_cache.st, grav.mg_cache
end

# ── Approach A: MG-preconditioned CG (Krylov-stabilized) ─────────────────────
# Outer CG on the exact composite A; preconditioner M⁻¹r = one composite-smoothed
# two-level cycle (x0 = 0). Same φ as plain CG, far fewer iterations, robust under AMR.
function _solve_poisson_mgpcg!(sim, grav)
    be = grav.ka; T = eltype(grav.bv)
    c, st, mg = _mg_ensure_caches!(sim, grav, be)
    Vespa._fill_poisson_rhs_flat!(c.b_host, sim, grav, st); copyto!(c.b, c.b_host)
    apply!(z, r) = (fill!(z, zero(T)); _amg_cycle!(z, r, c, mg, be))
    _matvec!(c.Ap, c.phi, c, be); @. c.r = c.b - c.Ap
    apply!(c.zz, c.r)
    copyto!(c.p, c.zz)
    rz = _dot!(c.tmp64, c.r, c.zz)
    bnorm = sqrt(_dot!(c.tmp64, c.b, c.b)); bnorm = bnorm == 0.0 ? 1.0 : bnorm
    iters = 0; relres = sqrt(_dot!(c.tmp64, c.r, c.r)) / bnorm
    for k in 1:grav.maxiter
        iters = k
        _matvec!(c.Ap, c.p, c, be)
        pAp = _dot!(c.tmp64, c.p, c.Ap); pAp <= 0.0 && break
        α = rz / pAp
        @. c.phi += T(α) * c.p
        @. c.r -= T(α) * c.Ap
        relres = sqrt(_dot!(c.tmp64, c.r, c.r)) / bnorm
        relres <= grav.tol && break
        apply!(c.zz, c.r)
        rznew = _dot!(c.tmp64, c.r, c.zz)
        β = rznew / rz
        @. c.p = c.zz + T(β) * c.p
        rz = rznew
    end
    if grav.project_mean
        μ = _dot!(c.tmp64, c.d_vol, c.phi) / sum(c.d_vol); @. c.phi -= μ
    end
    copyto!(c.phi_host, c.phi); Vespa._scatter_field!(grav.phiv, st, c.phi_host)
    return iters, relres
end

# ── Approach B: standalone multigrid V-cycle solver (RAMSES-style, no Krylov) ─
# Iterate the composite-smoothed two-level cycle on φ directly until the residual
# falls below tol. No CG ⇒ no dot-product/host syncs in the loop except the residual.
function _solve_poisson_mgvsolve!(sim, grav)
    be = grav.ka; T = eltype(grav.bv)
    c, st, mg = _mg_ensure_caches!(sim, grav, be)
    Vespa._fill_poisson_rhs_flat!(c.b_host, sim, grav, st); copyto!(c.b, c.b_host)
    bnorm = sqrt(_dot!(c.tmp64, c.b, c.b)); bnorm = bnorm == 0.0 ? 1.0 : bnorm
    iters = 0; relres = 1.0
    for k in 1:grav.maxiter
        iters = k
        _amg_cycle!(c.phi, c.b, c, mg, be)
        _matvec!(c.Ap, c.phi, c, be); @. c.r = c.b - c.Ap
        relres = sqrt(_dot!(c.tmp64, c.r, c.r)) / bnorm
        relres <= grav.tol && break
    end
    if grav.project_mean
        μ = _dot!(c.tmp64, c.d_vol, c.phi) / sum(c.d_vol); @. c.phi -= μ
    end
    copyto!(c.phi_host, c.phi); Vespa._scatter_field!(grav.phiv, st, c.phi_host)
    return iters, relres
end
