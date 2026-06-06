# MultigridSolver — the V-cycle host driver.
# Port of src/enzo/MultigridSolver.C (PRE_SMOOTH=2, POST_SMOOTH=3, NUM_CYCLES=1,
# bottom smoothing = 3·PRE_SMOOTH, tolerance=2e-6, max_iter=100, start_depth=0).
#
# The driver orchestrates the certified per-level kernels (mg_relax!, mg_calc_defect!,
# mg_restrict!, mg_prolong!). start_depth=0 makes the mg_prolong2 RHS pre-pass a
# no-op, so it is omitted (add mg_prolong2 only for a non-zero start depth).

const _PRE_SMOOTH  = 2
const _POST_SMOOTH = 3

"""
    mg_dims_schedule(d0::NTuple{3,Int}) -> Vector{NTuple{3,Int}}

The V-cycle level dimensions starting from the top grid `d0`, halving each axis as
`(n+1) ÷ 2` (Enzo's `Dims[d][depth+1] = (Dims[d][depth]+1)/2`) until the smallest
next dimension would be `< 3`. Returns `[d0, d1, …, d_bottom]` (clean `2^k+1` grids
land exactly: 33 → 17 → 9 → 5 → 3).
"""
function mg_dims_schedule(d0::NTuple{3,Int})
    dims = [d0]
    while true
        nxt = ((dims[end][1] + 1) ÷ 2, (dims[end][2] + 1) ÷ 2, (dims[end][3] + 1) ÷ 2)
        minimum(nxt) < 3 && break
        push!(dims, nxt)
    end
    return dims
end

# One recursive μ-cycle from level `L` (μ=1 ⇒ V-cycle, μ=2 ⇒ W-cycle): pre-smooth →
# defect → restrict → recurse μ× on the coarse correction → prolong-add → post-smooth.
# At the bottom level just smooth. Pure device work — no host reductions or syncs
# (the down-leg defect skips its norm; everything is queue-ordered).
function _mu_cycle!(Sol, RHS, Def, L::Int, nlev::Int, pre::Int, post::Int, mu::Int)
    T = eltype(Sol[L])
    if L == nlev                          # coarsest grid: smooth
        for _ in 1:(3 * pre)
            mg_relax!(Sol[L], RHS[L])
        end
        return
    end
    for _ in 1:pre
        mg_relax!(Sol[L], RHS[L])
    end
    mg_calc_defect!(Def[L], Sol[L], RHS[L]; compute_norm = false)
    mg_restrict!(RHS[L+1], Def[L])
    fill!(Sol[L+1], zero(T))
    for _ in 1:mu
        _mu_cycle!(Sol, RHS, Def, L + 1, nlev, pre, post, mu)
    end
    mg_prolong!(Def[L], Sol[L+1])         # coarse correction → fine scratch
    Sol[L] .+= Def[L]                     # add correction
    for _ in 1:post
        mg_relax!(Sol[L], RHS[L])
    end
end

# Snapshot / re-impose the width-1 Dirichlet boundary ring of `A` (the 6 faces).
# Needed for a NON-ZERO Dirichlet boundary (e.g. a subgrid's parent-interpolated
# potential): mg_prolong! writes the FULL fine grid, so the V-cycle's prolong-add
# perturbs the boundary by O(interp error) each cycle — which on a non-zero
# boundary leaves a fixed ~1e-4 solution error (invisible when the boundary is 0,
# as in all the uniform-grid tests). Re-imposing the saved faces after each μ-cycle
# pins the boundary and restores round-off recovery (validated: composite_subgrid_test.jl).
# getindex with `:` already materializes a NEW array of A's own type (host Array or
# device MtlArray/CuArray) — so the faces stay ON THE SAME BACKEND. (Do NOT `collect`:
# that forces them to a host Matrix, and broadcasting a host array into a device slice
# in _impose_faces! is a non-bitstype GPU-kernel argument → Metal compile error.)
_save_faces(A::AbstractArray{T,3}) where {T} =
    (A[1, :, :], A[end, :, :], A[:, 1, :], A[:, end, :], A[:, :, 1], A[:, :, end])
function _impose_faces!(A::AbstractArray{T,3}, f) where {T}
    A[1, :, :] .= f[1]; A[end, :, :] .= f[2]; A[:, 1, :] .= f[3]
    A[:, end, :] .= f[4]; A[:, :, 1] .= f[5]; A[:, :, end] .= f[6]
    return A
end

"""
    vcycle_solve!(sol0, rhs0; rtol=1e-6, maxcycles=50, stagnation=0.98,
                  cycle=:V, pre=2, post=3, dirichlet=false) -> (sol0, norm, relresid)

Solve `L·sol = rhs` on a uniform grid with a multigrid μ-cycle, in place in `sol0`
(a 3-D device array; `rhs0` matching). Returns the solution, the final defect L2
`norm`, and the relative residual `relresid = norm/norm₀`.

`dirichlet=true` re-imposes `sol0`'s initial boundary ring after every μ-cycle —
**required for a non-zero Dirichlet boundary** (a subgrid solve whose boundary is
the parent-interpolated potential), because `mg_prolong!` writes the full grid and
the prolong-add would otherwise drift the boundary (round-off recovery is lost; see
`_save_faces`). Leave `false` (default) for the zero-boundary uniform-grid case.

  * `cycle` — `:V` (one coarse recursion) or `:W` (two). On a 256³ field the
    **W-cycle contracts ~0.09/cycle (≈6 cycles to 1e-6) vs the V-cycle's ~0.50
    (≈20 cycles)** — textbook multigrid. The default is `:V` because on the GPU
    each W-cycle visits the coarse grids twice and those tiny launch-latency-bound
    kernels currently cost more wall-time than the fewer cycles save; `:W` wins on
    the CPU and would win on the GPU with single-command-buffer batching. Pick `:W`
    for robustness / ill-conditioned problems.
  * `pre`/`post` — pre-/post-smoothing sweeps per level (default 2/3, matching Enzo).

Convergence uses a **relative residual** `‖r‖/‖r₀‖ ≤ rtol` — precision-agnostic,
unlike Enzo's `‖r‖/mean(|φ|)` which an f32 residual floor can never satisfy. It
also stops on **stagnation** (`norm ≥ stagnation·prev_norm`), which auto-detects
the f32 round-off floor instead of grinding to `maxcycles`. No plain-relaxation
tail; the down-leg skips the discarded defect-norm reduction; and there are NO
per-kernel syncs — the only host round-trip is one finest-grid norm per cycle.
"""
function vcycle_solve!(sol0::AbstractArray{T,3}, rhs0::AbstractArray{T,3};
                       rtol::Real = 1e-6, maxcycles::Integer = 50,
                       stagnation::Real = 0.98, cycle::Symbol = :V,
                       pre::Integer = 2, post::Integer = 3,
                       dirichlet::Bool = false) where {T}
    be = KA.get_backend(sol0)
    d0 = size(sol0)
    dims = mg_dims_schedule(d0)
    nlev = length(dims)                  # levels 1..nlev ↔ Enzo depth 0..bottom
    mu = cycle === :W ? 2 : 1
    faces = dirichlet ? _save_faces(sol0) : nothing   # pin a non-zero parent boundary

    # Per-level field stores. Level 1 aliases the caller's arrays.
    Sol = Vector{typeof(sol0)}(undef, nlev)
    RHS = Vector{typeof(sol0)}(undef, nlev)
    Def = Vector{typeof(sol0)}(undef, nlev)
    Sol[1] = sol0; RHS[1] = rhs0; Def[1] = device_zeros(be, T, d0)
    for L in 2:nlev
        Sol[L] = device_zeros(be, T, dims[L])
        RHS[L] = device_zeros(be, T, dims[L])
        Def[L] = device_zeros(be, T, dims[L])
    end

    # initial residual ‖r₀‖ (sol = 0 ⇒ defect = rhs); the relative-residual base.
    norm0 = mg_calc_defect!(Def[1], Sol[1], RHS[1])
    norm0 == zero(T) && (KA.synchronize(be); return sol0, zero(T), zero(T))
    rtolf = T(rtol); stag = T(stagnation)
    norm = norm0; prev = T(Inf); iter = 0

    while iter < maxcycles
        _mu_cycle!(Sol, RHS, Def, 1, nlev, Int(pre), Int(post), mu)
        faces === nothing || _impose_faces!(Sol[1], faces)   # re-pin parent Dirichlet boundary
        # ── ONE finest-grid residual norm per cycle (the only host reduction) ──
        norm = mg_calc_defect!(Def[1], Sol[1], RHS[1])
        iter += 1
        norm ≤ rtolf * norm0 && break          # converged (relative residual)
        norm ≥ stag * prev && break            # stagnated (e.g. f32 round-off floor)
        prev = norm
    end

    KA.synchronize(be)
    return sol0, norm, norm / norm0
end
