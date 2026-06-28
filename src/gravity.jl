# Self-gravity: solve the Poisson equation ∇²φ = 4πG ρ on the composite (leaf)
# mesh and (Phase 3b+) add the gravitational source g = −∇φ to the gas. On a
# composite AMR mesh the discrete finite-volume Laplacian is structurally
# identical to the hydro flux divergence (`accumulate_flux!`): summing the
# face-gradient fluxes into each cell gives ∇²φ·V, and the backend's hanging-node
# sub-face enumeration makes it conservative and level-consistent across coarse↔
# fine jumps for free — exactly as it does for hydro. So gravity needs no per-
# level grids and no coarse→fine BC interpolation: ONE global solve over all
# leaves couples the levels through the shared faces.
#
# Operator sign. We solve with the symmetric positive-(semi)definite operator
# `A = −∇²·V` (the standard discrete Poisson matrix: positive diagonal Σ area/d,
# negative symmetric off-diagonals −area/d), so plain Conjugate Gradient applies.
# The right-hand side is then `b = −4πG ρ V`, and the recovered φ satisfies
# −∇²φ·V = −4πGρ·V ⟺ ∇²φ = 4πG ρ.
#
# Phase 3a (this file initially): the solver + analytic oracles only — no hydro
# coupling, no Simulation wiring. `GravityField` is a standalone value;
# `solve_poisson!(sim, grav)` fills its φ store from the current gas density.

const FOURπ = 4.0 * π

"""
    GravityField

Holds the gravitational potential store `φ` plus Conjugate-Gradient scratch
(`r`, `p`, `Ap`, `b`) as 1-component cell-average field stores on the simulation's
backend, their cached views, and the solver configuration. `bcs` are φ's OWN
boundary conditions (independent of the gas BCs). `project_mean` removes the
constant null space for a fully periodic box (where Poisson is singular).
"""
mutable struct GravityField
    phi; r; p; Ap; b                # scalar field stores (one component each)
    phiv; rv; pv; Apv; bv           # cached handle-indexed views
    G::Float64
    bcs::BoundaryConditions
    maxiter::Int
    tol::Float64
    project_mean::Bool
    ka::Any                         # nothing (CPU CG), or a KA backend (GPU/CPU device CG)
    ka_cache::Any                   # flattened leaf/face structure for the KA solve (built once)
    cg_cache::Any                   # CPU CSR cache (struct + flat work vectors) for the CPU CG
    precond::Symbol                 # :none (plain CG) or :mg (geometric-multigrid-preconditioned CG, KA path)
    mg_cache::Any                   # base-grid multigrid hierarchy + transfer maps (built once per mesh)
end

# Overloaded by the `VespaKAGravityExt` package extension (load KernelAbstractions
# + a device package). Runs the composite CG on the KA backend `grav.ka`.
function _solve_poisson_ka! end

# Overloaded by `VespaKAGravityExt`: computes g = −∇φ for every leaf on the device
# from the cached φ (`grav.ka_cache.phi`) + the gravity stencil, returning the device
# (gx, gy, gz). Lets the particle interp gather g without a host neighbour loop.
function _device_leaf_gravity! end

"""
    enable_gravity!(sim; G=1.0, bcs=Periodic(), maxiter=500, tol=1e-8,
                    project_mean=(bcs isa Periodic)) -> GravityField

Allocate the potential + CG scratch stores on `sim`'s backend, attach the
resulting `GravityField` to `sim.grav` (so `evolve!`/`step!` apply self-gravity),
and return it. φ carries its own boundary conditions `bcs` (default periodic).
`project_mean` (default: periodic) projects out the constant null mode so the
singular all-periodic Poisson problem is solvable.
"""
function enable_gravity!(sim::Simulation; G::Real = 1.0,
                         bcs = Periodic(), maxiter::Integer = 500,
                         tol::Real = (_Tf(sim) === Float32 ? 1.0e-5 : 1.0e-8),  # precision-aware default
                         project_mean::Bool = (bcs isa Periodic),
                         ka = nothing, precond::Symbol = :none)
    b = sim.backend
    N = rank(b)
    gbcs = _as_bcs(bcs, N)
    mk(name) = allocate_fields(b, FieldSpec([name]); layout = sim.layout, eltype = _Tf(sim))
    phi = mk(:phi); rr = mk(:r); pp = mk(:p); ap = mk(:Ap); bb = mk(:b)
    v(store, name) = field_view(b, store, name)
    grav = GravityField(phi, rr, pp, ap, bb,
                        v(phi, :phi), v(rr, :r), v(pp, :p), v(ap, :Ap), v(bb, :b),
                        Float64(G), gbcs, Int(maxiter), Float64(tol), project_mean,
                        ka, nothing, nothing, precond, nothing)
    sim.grav = grav
    return grav
end

# ── matrix-free operator A = −∇²·V ───────────────────────────────────────────
# Center-to-center face distance along `axis`: ½(w_i+w_j). Uses widths, never
# cell-center differences, so it is correct for the periodic wrap (no coordinate
# jump) and for coarse↔fine sub-faces (asymmetric half-sum).
@inline function _face_distance(b, i, j, axis::Int)
    wi = cell_width(b, i)
    wj = cell_width(b, j)
    return 0.5 * (wi[axis] + wj[axis])
end

# out ← A·x  with  A = −∇²·V. Mirrors accumulate_flux!: zero the accumulator,
# then add the (negated) face-gradient flux into each side.
function apply_laplacian!(sim::Simulation, grav::GravityField, xv, outv)
    b = sim.backend
    z = zero(eltype(outv))
    for_each_cell(b) do c
        outv[c] = z
    end
    for_each_face(b; bcs = grav.bcs) do left, right, axis, area
        _lap_face!(sim, left, right, axis, area, xv, outv)
    end
    return nothing
end

# interior↔interior (incl. periodic wrap and coarse↔fine sub-faces): the −∇²
# contribution. gflux = (x_j − x_i)/d·area is the +∇ gradient flux i→j; for
# A=−∇²·V we subtract it from i and add it to j (giving +area/d on each diagonal,
# −area/d symmetric off-diagonal — the SPD Poisson stencil).
@inline function _lap_face!(sim::Simulation, left::Interior, right::Interior,
                            axis::Int, area::Real, xv, outv)
    i, j = left.cell, right.cell
    T = eltype(outv)
    d = _face_distance(sim.backend, i, j, axis)
    gflux = (xv[j] - xv[i]) / T(d) * T(area)        # field precision
    outv[i] -= gflux
    outv[j] += gflux
    return nothing
end

# Domain boundary: default to homogeneous Neumann (∂φ/∂n = 0) — zero gradient
# flux, the natural "isolated" default. (Periodic faces never reach here; they
# resolve as Interior. Dirichlet support is a later addition.)
@inline _lap_face!(::Simulation, ::Interior, ::DomainBoundary, ::Int, ::Real, xv, outv) = nothing
@inline _lap_face!(::Simulation, ::DomainBoundary, ::Interior, ::Int, ::Real, xv, outv) = nothing

# ── CG vector primitives over the leaf set (the conserved_totals reduction shape)
@inline function dot_cells(sim::Simulation, av, bv)
    s = zero(promote_type(eltype(av), Float64))      # reduction in ≥f64
    for_each_cell(sim.backend) do c
        s += av[c] * bv[c]
    end
    return s
end

@inline function axpy_cells!(sim::Simulation, α::Real, xv, yv)  # y += α x
    a = eltype(yv)(α)
    for_each_cell(sim.backend) do c
        yv[c] += a * xv[c]
    end
    return nothing
end

@inline function copy_cells!(sim::Simulation, dst, src)
    for_each_cell(sim.backend) do c
        dst[c] = src[c]
    end
    return nothing
end

# p ← x + β p  (CG direction update)
@inline function _xpby_cells!(sim::Simulation, xv, β::Real, pv)
    b = eltype(pv)(β)
    for_each_cell(sim.backend) do c
        pv[c] = xv[c] + b * pv[c]
    end
    return nothing
end

# Subtract the volume-weighted mean (the physical constant mode on the AMR mesh).
function project_zero_mean!(sim::Simulation, xv)
    b = sim.backend
    Tr = promote_type(eltype(xv), Float64)
    s = zero(Tr); vol = zero(Tr)
    for_each_cell(b) do c
        v = cell_volume(b, c)
        s += xv[c] * v; vol += v
    end
    μ = eltype(xv)(s / vol)
    for_each_cell(b) do c
        xv[c] -= μ
    end
    return μ
end

# ── Poisson solve ────────────────────────────────────────────────────────────
# RHS b[c] = −4πG (ρ[c] − ρ̄) V[c]  (the −∇²·V operator's right-hand side; density
# read from the gas state). For the periodic box the operator is singular: A is a
# graph Laplacian whose null space is the *unweighted* constant vector (row sums
# vanish), so the RHS must satisfy the discrete solvability condition Σ_c b[c] = 0
# — NOT the volume-weighted mean being zero. Subtracting the volume-weighted mean
# density ρ̄ gives Σ_c b[c] = −4πG(Σρ V − ρ̄ Σ V) = 0 exactly on any mesh, uniform
# or AMR. (On a uniform mesh this is identical to the old flat-constant projection,
# since every V is equal; they only diverge across coarse↔fine level jumps, where
# the old projection left b non-orthogonal to the null space and CG diverged.)
# Fill the FLAT, ordinal-indexed RHS vector `b_flat` (length st.n, for_each_cell order)
# directly — no `for_each_cell` closures, no per-cell `cell_volume` tree lookups (uses the
# cached `st.vol`), and no separate gather (the solve uploads/uses `b_flat` as-is). Gas
# density is read by the TYPED handle `dv[st.handles[i]]` (cheap getindex), DM density by
# the ordinal `rho_p[i]` (aligned: both `st` and the locator iterate the same for_each_cell
# order). This replaced the two `for_each_cell` passes + `_gather_field(bv)` that dominated
# the host solve at 64³ (~41 ms + 13 ms → a couple of flat loops).
function _fill_poisson_rhs_flat!(b_flat, sim::Simulation, grav::GravityField, st)
    di = density_index(sim.model)
    dv = sim.sv[di]                                       # density view, hoisted (one access)
    ps = sim.particles
    rho_p = nothing
    if ps !== nothing
        deposit_particle_density!(sim, ps)               # device when grav.ka set; updates ps.rho_p
        rho_p = ps.rho_p
    end
    # Comoving Poisson carries a 1/a factor: ∇²φ = (4πG/a)(ρ − ρ̄); a = aⁿ. Non-cosmo: a = 1.
    inv_a = sim.cosmo === nothing ? 1.0 : 1.0 / sim.cosmo.a
    _rhs_flat_kernel!(b_flat, st.handles, st.vol, dv, rho_p, grav.G, grav.project_mean, inv_a,
                      eltype(b_flat), st.n)
    return nothing
end

# Particle (DM) density at ordinal `i`, or 0 when there are no particles. Dispatch on
# the concrete `rho_p` type (Nothing vs Vector) keeps the flat loops type-stable.
@inline _pden(::Nothing, i) = 0.0
@inline _pden(rho_p::Vector, i) = @inbounds Float64(rho_p[i])

# Function barrier (concrete `handles`/`dv`/`rho_p`/`vol`): the discrete solvability mean
# ρ̄ = Σ(ρ_gas+ρ_DM)·vol / Σvol (unweighted-null-space condition), then
# b[i] = −4πG(ρ−ρ̄)·vol·inv_a. f64 accumulation; stored at the field precision T.
function _rhs_flat_kernel!(b_flat, handles, vol, dv, rho_p, G, project_mean, inv_a, ::Type{T}, n) where {T}
    ρ̄ = 0.0
    if project_mean
        s = 0.0; vsum = 0.0
        @inbounds for i in 1:n
            ρ = Float64(dv[handles[i]]) + _pden(rho_p, i)
            s += ρ * vol[i]; vsum += vol[i]
        end
        ρ̄ = s / vsum
    end
    @inbounds for i in 1:n
        ρ = Float64(dv[handles[i]]) + _pden(rho_p, i)
        b_flat[i] = T(-FOURπ * G * (ρ - ρ̄) * vol[i] * inv_a)
    end
    return nothing
end

"""
    solve_poisson!(sim, grav) -> (iters, relres)

Solve `∇²φ = 4πG ρ` for the current gas density into `grav`'s φ store via
matrix-free Conjugate Gradient (operator `A = −∇²·V`, SPD). Warm-starts from the
existing φ (cheap when φ evolves slowly). For a periodic box the constant null
mode is projected out of both the RHS and the solution. Returns the iteration
count and final relative residual.
"""
function solve_poisson!(sim::Simulation, grav::GravityField)
    grav.ka === nothing || return _solve_poisson_ka!(sim, grav)
    return _solve_poisson_csr_cpu!(sim, grav)
end

# CPU Conjugate Gradient on the CSR adjacency (the SAME flattened operator the GPU
# path uses). The CSR + flat work vectors are built once per mesh and cached on
# `grav.cg_cache` (invalidated on regrid), so the matvec is a per-cell GATHER over
# the neighbour list — NOT a `for_each_face` walk every iteration (the old path,
# which made AMR-CG O(n_iter·n_faces) and was the CPU bottleneck). f64 accumulation
# (matvec + dots) protects convergence at f32 field storage. Equivalent operator ⇒
# same φ to solver tolerance.
function _solve_poisson_csr_cpu!(sim::Simulation, grav::GravityField)
    T = eltype(grav.bv)
    if grav.cg_cache === nothing
        st = _build_poisson_ka_struct(sim, grav)
        z() = zeros(T, st.n)
        grav.cg_cache = (st = st, phi = z(), r = z(), p = z(), Ap = z(), b = z())
        _gather_into!(grav.cg_cache.phi, st, grav.phiv)  # warm start from host φ ONCE per mesh
    end
    c = grav.cg_cache
    _fill_poisson_rhs_flat!(c.b, sim, grav, c.st)       # flat RHS straight into c.b (no bv, no gather)
    # c.phi persists across solves as the warm start (kept in the cache between regrids);
    # only re-seeded from the host φ when the cache is (re)built above.
    # The hot CG loop runs behind a FUNCTION BARRIER: `grav.cg_cache::Any` makes c.st /
    # c.phi / c.r / … abstractly typed, so reading them inline would dynamic-dispatch
    # EVERY element access (the whole solve was ~100M allocs / multi-second). Passing them
    # as arguments specializes the barrier on the concrete `PoissonKAStruct` + `Vector{T}`.
    return _cg_csr_run!(c.st, grav.phiv, c.phi, c.r, c.p, c.Ap, c.b,
                        grav.maxiter, grav.tol, grav.project_mean, T)
end

# Concrete-typed CG iteration (the function barrier — see _solve_poisson_csr_cpu!). `st`
# is left unannotated (PoissonKAStruct is defined below): Julia specializes the barrier on
# the concrete argument types at the call regardless, which is the whole point.
function _cg_csr_run!(st, phiv, phi::AbstractVector{T}, r, p, Ap, b,
                      maxiter::Int, tol::Real, project_mean::Bool, ::Type{T}) where {T}
    _lap_csr_cpu!(Ap, phi, st)                           # r = b − A·φ0
    @inbounds for i in 1:st.n
        r[i] = b[i] - Ap[i]
    end
    copyto!(p, r)
    rsold = _dot64(r, r)
    bnorm = sqrt(_dot64(b, b)); bnorm = bnorm == 0.0 ? 1.0 : bnorm
    iters = 0; relres = sqrt(rsold) / bnorm
    for k in 1:maxiter
        iters = k
        _lap_csr_cpu!(Ap, p, st)
        pAp = _dot64(p, Ap)
        pAp <= 0.0 && break
        α = rsold / pAp
        @inbounds for i in 1:st.n
            phi[i] += T(α) * p[i]
            r[i] -= T(α) * Ap[i]
        end
        rsnew = _dot64(r, r)
        relres = sqrt(rsnew) / bnorm
        relres <= tol && break
        β = rsnew / rsold
        @inbounds for i in 1:st.n
            p[i] = r[i] + T(β) * p[i]
        end
        rsold = rsnew
    end
    if project_mean                                     # subtract the volume-weighted mean
        s = 0.0; vsum = 0.0
        @inbounds for i in 1:st.n
            s += st.vol[i] * Float64(phi[i]); vsum += st.vol[i]
        end
        μ = T(s / vsum)
        @inbounds for i in 1:st.n
            phi[i] -= μ
        end
    end
    _scatter_field!(phiv, st, phi)
    return iters, relres
end

# out = A·x with A = −∇²·V, via the CSR neighbour list. f64 accumulation, stored at
# the field precision. (x_c − x_nb) summed with coef = area/distance.
function _lap_csr_cpu!(out, x, st)
    @inbounds for c in 1:st.n
        xc = Float64(x[c]); s = 0.0
        for k in st.rowptr[c]:(st.rowptr[c + 1] - 1)
            s += st.coefval[k] * (xc - Float64(x[st.colidx[k]]))
        end
        out[c] = oftype(out[c], s)
    end
    return nothing
end

@inline function _dot64(x, y)                           # f64-accumulated inner product
    s = 0.0
    @inbounds for i in eachindex(x)
        s += Float64(x[i]) * Float64(y[i])
    end
    return s
end

@inline function _gather_into!(dst, st, view)
    @inbounds for i in 1:st.n
        dst[i] = view[st.handles[i]]
    end
    return nothing
end

# ── gravitational acceleration g = −∇φ, and the gas source term ──────────────
# Cell-centered g along each axis from the two face neighbours of the φ field,
# using the level-aware center distances. At a φ domain boundary (homogeneous
# Neumann default) the missing side contributes zero gradient. Returns an
# NTuple{3} (unused axes are 0).
# Convenience wrapper. `grav.phiv` is an `Any` field of GravityField; per-cell callers
# in hot loops should hoist it once and call `_grav_accel` directly (the typed barrier)
# so the φ-view access doesn't dynamic-dispatch every cell.
@inline gravity_accel(sim::Simulation, grav::GravityField, cell) =
    _grav_accel(sim.backend, grav.phiv, grav.bcs, cell)

# Cell-centered g = −∇φ from the two face neighbours per axis. `phiv` is passed
# explicitly (typed) so the whole gradient is allocation-free under a function barrier.
# One axis (`_gax`) computed at a time and assembled as a literal 3-tuple — avoids a
# `ntuple(_) do` closure that would box the captured φc/nbh/isbnd per cell.
@inline function _grav_accel(b, phiv, bcs::BoundaryConditions, cell)
    N = rank(b)
    T = eltype(phiv)
    φc = T(phiv[cell])
    nbh, isbnd = face_neighbor_handles(b, cell; bcs = bcs)   # one alloc-free read
    g1 = _gax(b, phiv, cell, φc, nbh, isbnd, 1, T)
    g2 = N >= 2 ? _gax(b, phiv, cell, φc, nbh, isbnd, 2, T) : zero(T)
    g3 = N >= 3 ? _gax(b, phiv, cell, φc, nbh, isbnd, 3, T) : zero(T)
    return (g1, g2, g3)
end

# Axis index (1..N) of conserved component `k` if it is a momentum component, else 0.
# Unrolled over the (small, static) momentum-index tuple ⇒ type-stable, alloc-free.
@inline function _mom_axis(mom, k::Int)
    a = 0
    @inbounds for d in eachindex(mom)
        mom[d] == k && (a = d)
    end
    return a
end

# Full-width gravity source contribution (to ADD into the subtracted accumulator):
# −ρ·g_d·V on momentum component d, −(ρv·g)·V on energy, 0 elsewhere. Built as a
# literal NTuple (Val length) so it stays allocation-free and writes via map/set_U!.
@inline function _grav_src_tuple(U, g, ρ::T, V::T, mom, ei::Int, ::Val{NV}) where {T,NV}
    es = zero(T)
    @inbounds for d in eachindex(mom)
        es += U[mom[d]] * g[d]                       # ρv·g  (U[mom[d]] = ρv_d)
    end
    esV = es * V
    return ntuple(Val(NV)) do k
        a = _mom_axis(mom, k)
        a != 0 ? -(ρ * g[a] * V) : (k == ei ? -esV : zero(T))
    end
end

@inline function _gax(b, phiv, cell, φc::T, nbh, isbnd, d::Int, ::Type{T}) where {T}
    lo = 2d - 1; hi = 2d                       # central diff: (φ_hi − φ_lo)/(d_lo + d_hi)
    wlo = isbnd[lo] ? φc : T(phiv[nbh[lo]])
    whi = isbnd[hi] ? φc : T(phiv[nbh[hi]])
    dlo = isbnd[lo] ? 0.5 * cell_width(b, cell)[d] : _face_distance(b, cell, nbh[lo], d)
    dhi = isbnd[hi] ? 0.5 * cell_width(b, cell)[d] : _face_distance(b, cell, nbh[hi], d)
    return -(whi - wlo) / T(dlo + dhi)         # g_d = −∂φ/∂x_d (field precision)
end

"""
    apply_gravity_source!(sim, grav; level=nothing)

Add the gravitational source for the current φ into the SAME `accv` accumulator
the flux uses (so the existing `_euler_apply!`/`_rk2_combine!` apply it as part of
`U −= dt·acc/V`). The source is `+ρg` on momentum and `+ρv·g` on energy; since
the accumulator is *subtracted*, we push `−ρg·V` and `−(ρv·g)·V`. `level`
restricts to one refinement level (matches the subcycling update). Call after each
`accumulate_flux!` so SSP-RK2 integrates the source to 2nd order; `g` is held
fixed across the step (φ solved once per root step).
"""
function apply_gravity_source!(sim::Simulation, grav::GravityField; level = nothing)
    # Hoist the Any-typed φ view once and run the per-cell loop behind a function
    # barrier (`grav.phiv` would otherwise dynamic-dispatch every cell: ~100 allocs/cell).
    _grav_source_loop!(sim.backend, sim.accv, sim.sv, grav.phiv, grav.bcs,
                       momentum_indices(sim.model), energy_index(sim.model),
                       density_index(sim.model), _Tf(sim), nvars_val(sim.model), level)
end

function _grav_source_loop!(b, av, sv, phiv, bcs::BoundaryConditions, mom, ei, di, Tf, nvV, level)
    for_each_cell(b; level = level) do c
        U = get_U(sv, c)
        ρ = U[di]
        V = Tf(cell_volume(b, c))                    # geometry → field precision
        g = _grav_accel(b, phiv, bcs, c)
        # Build the FULL per-component source as a tuple and add it via set_U!/map —
        # the accumulator is subtracted later, so we add −ρg·V (momentum) and −ρv·g·V
        # (energy). Avoids the dynamic `av[mi][c]` index into the heterogeneous view tuple.
        src = _grav_src_tuple(U, g, ρ, V, mom, ei, nvV)
        set_U!(av, c, map(+, get_U(av, c), src))
    end
    return nothing
end

# Free-fall timestep limiter, folded into the CFL reduction: dt ≲ η/√(4πGρ).
@inline _gravity_invdt(grav::GravityField, ρ::Real) =
    ρ > 0 ? sqrt(FOURπ * grav.G * ρ) / GRAV_DT_ETA : 0.0
const GRAV_DT_ETA = 0.5

# ── flattened leaf/face structure for the KA (GPU/CPU device) Poisson solve ───
# The composite operator A = −∇²·V is, per cell, (A·x)[c] = Σ_{neighbours nb of c}
# coef·(x_c − x_nb), with coef = area/distance — exactly what `apply_laplacian!`'s
# face scatter computes (each interior, incl. coarse↔fine sub-, face adds ±coef·
# (x_j−x_i) to its two cells), so the operator is identical and stays across-level.
# Storing it as CSR adjacency (per-cell neighbour list) makes the matvec a pure
# GATHER — no atomics — one device work-item per cell. The CG vector ops are device
# broadcasts/reductions. Built once per mesh (host, pure MeshInterface), invalidated
# on regrid. Domain-boundary faces contribute nothing (homogeneous Neumann).
struct PoissonKAStruct{H}
    handles::Vector{H}             # ordinal → leaf handle (gather/scatter back); typed ⇒ fast getindex
    n::Int
    rowptr::Vector{Int32}          # CSR row pointers (length n+1, 1-based)
    colidx::Vector{Int32}          # neighbour ordinal per adjacency (length nnz)
    coefval::Vector{Float64}       # area/distance per adjacency
    vol::Vector{Float64}           # per-leaf volume
    # gravity stencil (g = −∇φ per leaf): per axis (3) × cell (n), lo/hi neighbour
    # ordinal (self when that side is a boundary) and 1/(d_lo+d_hi). Lets g be
    # computed on the device from φ with NO host neighbour loop.
    glo::Matrix{Int32}             # (3, n)
    ghi::Matrix{Int32}             # (3, n)
    gcoef::Matrix{Float64}         # (3, n)
end

# handle → ordinal map. For integer handles (HGBackend) a DENSE Vector (O(1) lookup,
# no hashing) is ~5× faster than a Dict and is what the rebuild hot loops use; other
# handle types (RefMesh CartesianIndex) fall back to a Dict.
function _ordmap(handles::Vector{H}, n) where {H}
    if H <: Integer
        maxh = n == 0 ? 0 : Int(maximum(handles))
        omap = zeros(Int32, maxh)
        @inbounds for i in 1:n
            omap[handles[i]] = Int32(i)
        end
        return omap
    else
        d = Dict{H,Int}(); sizehint!(d, n)
        @inbounds for i in 1:n
            d[handles[i]] = i
        end
        return d
    end
end

# Rebuilt once per mesh (host) — the regrid hot path. The CSR adjacency AND the gravity
# stencil are filled in a SINGLE `for_each_face` walk (left=lo, right=hi per axis, so a
# face directly gives `right` = left's hi-neighbour and `left` = right's lo-neighbour).
# `neighbor()` (an HG tree traversal, 6/cell) is then needed ONLY for the few COARSE cells
# at refinement boundaries, where the face walk's representative fine neighbour may differ
# from `neighbor()`'s — the conforming/periodic majority is bit-identical to the old
# per-cell `neighbor()` build. This replaced the Vector-of-Vectors adjacency (push! growth)
# + the 6·n `neighbor()` calls that dominated the regrid (~8 s → ~2 s at 1 M leaves).
function _build_poisson_ka_struct(sim::Simulation, grav::GravityField)
    b = sim.backend
    handles_any = Any[]
    for_each_cell(b) do c
        push!(handles_any, c)
        return nothing
    end
    n = length(handles_any)
    H = n == 0 ? Any : typeof(handles_any[1])
    handles = convert(Vector{H}, handles_any)
    omap = _ordmap(handles, n)                    # dense (Int) or Dict — typed in the barrier
    return _assemble_poisson_struct(b, grav.bcs, handles, omap, n, Val(rank(b)))
end

function _assemble_poisson_struct(b, bcs, handles::Vector{H}, omap, n, ::Val{N}) where {H,N}
    vol = Vector{Float64}(undef, n); lev = Vector{Int32}(undef, n)
    @inbounds for i in 1:n
        vol[i] = Float64(cell_volume(b, handles[i]))
        lev[i] = Int32(level_of(b, handles[i]))
    end
    # stencil init: self-neighbour, zero distances (0 ⇒ "unfilled side" = domain boundary)
    glo = Matrix{Int32}(undef, 3, n); ghi = Matrix{Int32}(undef, 3, n)
    dlo = zeros(Float64, 3, n); dhi = zeros(Float64, 3, n)
    @inbounds for c in 1:n, d in 1:3
        glo[d, c] = Int32(c); ghi[d, c] = Int32(c)
    end
    marked = falses(n)                            # coarse cells at a level jump (need neighbor() recompute)
    ei = Int32[]; ej = Int32[]; ec = Float64[]    # flat edge list (one per interior face)
    sizehint!(ei, N * n); sizehint!(ej, N * n); sizehint!(ec, N * n)
    for_each_face(b; bcs = bcs) do left, right, axis, area
        if left isa Interior && right isa Interior
            i = Int(omap[left.cell]); j = Int(omap[right.cell])
            fd = Float64(_face_distance(b, left.cell, right.cell, axis))
            push!(ei, Int32(i)); push!(ej, Int32(j)); push!(ec, Float64(area) / fd)
            @inbounds begin
                ghi[axis, i] = Int32(j); dhi[axis, i] = fd      # right is i's hi-neighbour
                glo[axis, j] = Int32(i); dlo[axis, j] = fd      # left  is j's lo-neighbour
                li = lev[i]; lj = lev[j]
                li != lj && (li < lj ? (marked[i] = true) : (marked[j] = true))
            end
        end
        return nothing
    end
    # CSR (symmetric) from the flat edges via counting sort — no per-row Vectors.
    deg = zeros(Int32, n)
    @inbounds for k in eachindex(ei)
        deg[ei[k]] += Int32(1); deg[ej[k]] += Int32(1)
    end
    rowptr = Vector{Int32}(undef, n + 1); rowptr[1] = 1
    @inbounds for c in 1:n
        rowptr[c + 1] = rowptr[c] + deg[c]
    end
    nnz = Int(rowptr[n + 1]) - 1
    colidx = Vector{Int32}(undef, nnz); coefval = Vector{Float64}(undef, nnz)
    cur = copy(rowptr)
    @inbounds for k in eachindex(ei)
        i = ei[k]; j = ej[k]; cf = ec[k]
        p = cur[i]; colidx[p] = j; coefval[p] = cf; cur[i] = p + Int32(1)
        q = cur[j]; colidx[q] = i; coefval[q] = cf; cur[j] = q + Int32(1)
    end
    # gcoef for the conforming majority (from the accumulated face distances); a side left
    # at 0 is a domain boundary ⇒ homogeneous Neumann (self + half-width).
    gcoef = zeros(Float64, 3, n)
    @inbounds for c in 1:n
        marked[c] && continue
        for d in 1:N
            dl = dlo[d, c]; dh = dhi[d, c]
            if dl == 0.0 || dh == 0.0
                hw = 0.5 * Float64(cell_width(b, handles[c])[d])
                dl == 0.0 && (dl = hw); dh == 0.0 && (dh = hw)
            end
            gcoef[d, c] = 1.0 / (dl + dh)
        end
    end
    # exact recompute of the coarse refinement-boundary cells (matches the old build).
    # ONE cached-tuple read per marked cell via `face_neighbor_handles` (plain handles +
    # boundary mask) instead of 2N boxed `neighbor()` calls — bit-identical (same
    # representative source) but allocation-free, which collapses the regrid hot path.
    @inbounds for c in 1:n
        marked[c] || continue
        h = handles[c]
        nbh, isbnd = face_neighbor_handles(b, h; bcs = bcs)
        for d in 1:N
            lo = 2d - 1; hi = 2d
            if isbnd[lo]
                glo[d, c] = Int32(c); dl = 0.5 * Float64(cell_width(b, h)[d])
            else
                nb = nbh[lo]; glo[d, c] = Int32(omap[nb]); dl = Float64(_face_distance(b, h, nb, d))
            end
            if isbnd[hi]
                ghi[d, c] = Int32(c); dh = 0.5 * Float64(cell_width(b, h)[d])
            else
                nb = nbh[hi]; ghi[d, c] = Int32(omap[nb]); dh = Float64(_face_distance(b, h, nb, d))
            end
            gcoef[d, c] = 1.0 / (dl + dh)
        end
        for d in (N + 1):3
            glo[d, c] = Int32(c); ghi[d, c] = Int32(c); gcoef[d, c] = 0.0
        end
    end
    return PoissonKAStruct{H}(handles, n, rowptr, colidx, coefval, vol, glo, ghi, gcoef)
end

# field view → flat host vector in the FIELD precision (f32 when fields are f32),
# and back. Geometry (coef/vol) stays f64; only the field storage follows `eltype`.
_gather_field(st::PoissonKAStruct, view) = (T = eltype(view); T[view[h] for h in st.handles])
function _scatter_field!(view, st::PoissonKAStruct, x)
    T = eltype(view)
    @inbounds for k in 1:st.n
        view[st.handles[k]] = T(x[k])
    end
    return nothing
end
