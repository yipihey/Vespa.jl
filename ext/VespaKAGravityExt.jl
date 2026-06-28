# KernelAbstractions composite Poisson solve (the GPU/CPU device path for
# `enable_gravity!(...; ka=backend)`). The matrix-free CG of `src/gravity.jl` is
# moved onto a KA backend: the leaves are flattened to ordinals and the operator
# A = −∇²·V is stored as CSR adjacency (`Vespa._build_poisson_ka_struct`), so the
# matvec is a per-cell GATHER kernel (no atomics) computing the SAME stencil the
# CPU `apply_laplacian!` does — including coarse↔fine sub-faces, so it stays
# across-level. The CG vector ops are device broadcasts/reductions. Result matches
# the CPU CG to solver tolerance; the operator is identical, so conservation/accuracy
# carry over. Device-agnostic (CPU()/CUDABackend()/MetalBackend()).

module VespaKAGravityExt

using Vespa
using KernelAbstractions
using MeshInterface: max_level
const KA = KernelAbstractions

# (A·x)[c] = Σ_{neighbours nb} coef·(x_c − x_nb)   — the −∇²·V operator, per cell.
# `x`/`out` are in the field precision (f32 ok); `coefval` is f64 geometry, so the
# accumulation `s` runs in f64 and is stored back at the field precision.
@kernel function _lap_csr_kernel!(out, @Const(x), @Const(rowptr), @Const(colidx), @Const(coefval))
    c = @index(Global)
    @inbounds begin
        xc = Float64(x[c])
        s = 0.0
        @fastmath for k in rowptr[c]:(rowptr[c + 1] - 1)   # FMA contraction; conservation is structural
            s += coefval[k] * (xc - Float64(x[colidx[k]]))
        end
        out[c] = s                                          # → eltype(out) (f32 storage)
    end
end

_to_dev(be, a) = (d = KA.allocate(be, eltype(a), size(a)); copyto!(d, a); d)
@inline _f64mul(a, b) = Float64(a) * Float64(b)
# f64-accumulated reduction into a reused scratch buffer (no per-call n-sized temp).
# `_dot!` returns the host scalar (a blocking device→host copy) — used for one-off
# reductions (bnorm, project_mean). `_dot_dev!` writes the result into a 1-element
# DEVICE array (`sum!`, no host copy) — used in the CG hot loop so α and the dot
# results stay on-device and the host does ONE read per iteration (the convergence
# check), not two, letting the matvec→dot→update→dot chain pipeline on the stream.
@inline _dot!(tmp, x, y) = (tmp .= _f64mul.(x, y); sum(tmp))
@inline _dot_dev!(out1, tmp, x, y) = (tmp .= _f64mul.(x, y); sum!(out1, tmp); out1)

# g = −∇φ per leaf, on the device, from the cached stencil (lo/hi neighbour ordinal
# + 1/(dlo+dhi) per axis) — replaces the host `_leaf_gravity` loop + its upload.
@kernel function _grav_kernel!(gx, gy, gz, @Const(phi), @Const(glo), @Const(ghi), @Const(gcoef))
    c = @index(Global)
    @inbounds begin
        gx[c] = -(Float64(phi[ghi[1, c]]) - Float64(phi[glo[1, c]])) * gcoef[1, c]
        gy[c] = -(Float64(phi[ghi[2, c]]) - Float64(phi[glo[2, c]])) * gcoef[2, c]
        gz[c] = -(Float64(phi[ghi[3, c]]) - Float64(phi[glo[3, c]])) * gcoef[3, c]
    end
end

# No explicit `KA.synchronize` here: the matvec and the dot/broadcast that consume `out`
# run on the same stream (ordered), and the following `_dot!`'s `sum` blocks anyway —
# an extra full-device sync per CG iteration was pure host-stall overhead.
function _matvec!(out, x, c, be)
    _lap_csr_kernel!(be)(out, x, c.d_rowptr, c.d_colidx, c.d_coefval; ndrange = length(x))
    return out
end

function Vespa._solve_poisson_ka!(sim, grav)
    # Composite-smoothed geometric multigrid (uniform AND AMR): a damped-Jacobi smoother
    # on the real composite operator + a base-grid V-cycle. Two outer-loop forms to pick:
    #   :mg  — MG-preconditioned CG (Krylov-stabilized)        [_solve_poisson_mgpcg!]
    #   :mgv — standalone V-cycle solver (RAMSES-style, no CG) [_solve_poisson_mgvsolve!]
    grav.precond === :mg  && return _solve_poisson_mgpcg!(sim, grav)
    grav.precond === :mgv && return _solve_poisson_mgvsolve!(sim, grav)
    be = grav.ka
    T = eltype(grav.bv)                                 # field precision (f32 ok); geom stays f64
    if grav.ka_cache === nothing
        st = Vespa._build_poisson_ka_struct(sim, grav)
        grav.ka_cache = (st = st,
                         d_rowptr = _to_dev(be, st.rowptr),
                         d_colidx = _to_dev(be, st.colidx),
                         d_coefval = _to_dev(be, st.coefval),   # f64 geometry coefficients
                         d_vol = _to_dev(be, st.vol),           # f64 cell volumes
                         d_glo = _to_dev(be, st.glo),           # gravity stencil (g = −∇φ)
                         d_ghi = _to_dev(be, st.ghi),
                         d_gcoef = _to_dev(be, st.gcoef),
                         gx = KA.allocate(be, Float64, (st.n,)),  # per-leaf g (device)
                         gy = KA.allocate(be, Float64, (st.n,)),
                         gz = KA.allocate(be, Float64, (st.n,)),
                         tmp64 = KA.allocate(be, Float64, (st.n,)),  # CG dot scratch
                         phi = KA.allocate(be, T, (st.n,)),     # CG vectors at field precision
                         r = KA.allocate(be, T, (st.n,)),
                         p = KA.allocate(be, T, (st.n,)),
                         Ap = KA.allocate(be, T, (st.n,)),
                         b = KA.allocate(be, T, (st.n,)),
                         b_host = zeros(T, st.n),               # host staging for the flat RHS
                         phi_host = zeros(T, st.n),             # host staging for the φ scatter (no Array() temp)
                         s_pAp = KA.allocate(be, Float64, (1,)),    # device CG scalars (no per-iter host read)
                         s_rsold = KA.allocate(be, Float64, (1,)),
                         s_rsnew = KA.allocate(be, Float64, (1,)),
                         s_alpha = KA.allocate(be, Float64, (1,)),
                         hbuf = Vector{Float64}(undef, 1))      # 1-elem host staging for the rsnew read
        copyto!(grav.ka_cache.phi, Vespa._gather_field(st, grav.phiv))  # warm start from host φ ONCE per mesh
    end
    c = grav.ka_cache; st = c.st

    Vespa._fill_poisson_rhs_flat!(c.b_host, sim, grav, st)   # flat host RHS (deposits DM particles)
    copyto!(c.b, c.b_host)                                    # upload RHS to device
    # c.phi persists on the device across solves as the warm start.
    _matvec!(c.Ap, c.phi, c, be)                    # r = b − A·φ0
    @. c.r = c.b - c.Ap
    copyto!(c.p, c.r)
    _dot_dev!(c.s_rsold, c.tmp64, c.r, c.r)         # rsold on device
    copyto!(c.hbuf, c.s_rsold); rsold = c.hbuf[1]   # one host read for the initial relres
    bnorm = sqrt(_dot!(c.tmp64, c.b, c.b)); bnorm = bnorm == 0.0 ? 1.0 : bnorm
    iters = 0; relres = sqrt(rsold) / bnorm
    # Device-scalar CG: pAp and α stay on the device (`α = rsold/pAp` via a 1-element
    # broadcast); only `rsnew` is read to the host each iteration — both for the
    # convergence test AND to form β = rsnew/rsold on the host (cheap, already paid).
    # This is exactly the same math as the host-scalar CG (bit-identical α/β values),
    # so it converges in the same iterations and matches the CPU path — it just halves
    # the per-iteration host↔device round-trips.
    for k in 1:grav.maxiter
        iters = k
        _matvec!(c.Ap, c.p, c, be)
        _dot_dev!(c.s_pAp, c.tmp64, c.p, c.Ap)      # pAp on device
        @. c.s_alpha = c.s_rsold / c.s_pAp          # α = rsold/pAp on device
        @. c.phi += c.s_alpha * c.p
        @. c.r -= c.s_alpha * c.Ap
        _dot_dev!(c.s_rsnew, c.tmp64, c.r, c.r)     # rsnew on device
        copyto!(c.hbuf, c.s_rsnew); rsnew = c.hbuf[1]   # the one host read per iteration
        relres = sqrt(rsnew) / bnorm
        relres <= grav.tol && break
        β = rsnew / rsold
        @. c.p = c.r + β * c.p
        copyto!(c.s_rsold, c.s_rsnew)               # rsold ← rsnew (device, for next α)
        rsold = rsnew
    end
    if grav.project_mean
        μ = _dot!(c.tmp64, c.d_vol, c.phi) / sum(c.d_vol)   # volume-weighted mean of φ
        @. c.phi -= μ
    end
    copyto!(c.phi_host, c.phi)                       # device→host into the reused buffer (no temp)
    Vespa._scatter_field!(grav.phiv, st, c.phi_host)
    return iters, relres
end

# g = −∇φ per leaf on the device, from the cached φ + gravity stencil (no host loop).
function Vespa._device_leaf_gravity!(grav, be)
    c = grav.ka_cache
    _grav_kernel!(be)(c.gx, c.gy, c.gz, c.phi, c.d_glo, c.d_ghi, c.d_gcoef; ndrange = c.st.n)
    KA.synchronize(be)
    return c.gx, c.gy, c.gz
end

include(joinpath(@__DIR__, "mg_pcg.jl"))   # geometric-multigrid-preconditioned CG (opt-in: grav.precond=:mg)

end # module
