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
end

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
                         tol::Real = 1e-8,
                         project_mean::Bool = (bcs isa Periodic))
    b = sim.backend
    N = rank(b)
    gbcs = _as_bcs(bcs, N)
    mk(name) = allocate_fields(b, FieldSpec([name]); layout = sim.layout)
    phi = mk(:phi); rr = mk(:r); pp = mk(:p); ap = mk(:Ap); bb = mk(:b)
    v(store, name) = field_view(b, store, name)
    grav = GravityField(phi, rr, pp, ap, bb,
                        v(phi, :phi), v(rr, :r), v(pp, :p), v(ap, :Ap), v(bb, :b),
                        Float64(G), gbcs, Int(maxiter), Float64(tol), project_mean)
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
    for_each_cell(b) do c
        outv[c] = 0.0
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
                            axis::Int, area::Float64, xv, outv)
    i, j = left.cell, right.cell
    d = _face_distance(sim.backend, i, j, axis)
    gflux = (xv[j] - xv[i]) / d * area
    outv[i] -= gflux
    outv[j] += gflux
    return nothing
end

# Domain boundary: default to homogeneous Neumann (∂φ/∂n = 0) — zero gradient
# flux, the natural "isolated" default. (Periodic faces never reach here; they
# resolve as Interior. Dirichlet support is a later addition.)
@inline _lap_face!(::Simulation, ::Interior, ::DomainBoundary, ::Int, ::Float64, xv, outv) = nothing
@inline _lap_face!(::Simulation, ::DomainBoundary, ::Interior, ::Int, ::Float64, xv, outv) = nothing

# ── CG vector primitives over the leaf set (the conserved_totals reduction shape)
@inline function dot_cells(sim::Simulation, av, bv)
    s = 0.0
    for_each_cell(sim.backend) do c
        s += av[c] * bv[c]
    end
    return s
end

@inline function axpy_cells!(sim::Simulation, α::Float64, xv, yv)  # y += α x
    for_each_cell(sim.backend) do c
        yv[c] += α * xv[c]
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
@inline function _xpby_cells!(sim::Simulation, xv, β::Float64, pv)
    for_each_cell(sim.backend) do c
        pv[c] = xv[c] + β * pv[c]
    end
    return nothing
end

# Subtract the volume-weighted mean (the physical constant mode on the AMR mesh).
function project_zero_mean!(sim::Simulation, xv)
    b = sim.backend
    s = 0.0; vol = 0.0
    for_each_cell(b) do c
        v = cell_volume(b, c)
        s += xv[c] * v; vol += v
    end
    μ = s / vol
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
function _fill_poisson_rhs!(sim::Simulation, grav::GravityField)
    b = sim.backend
    di = density_index(sim.model)
    ρ̄ = 0.0
    if grav.project_mean
        s = 0.0; vol = 0.0
        for_each_cell(b) do c
            V = cell_volume(b, c)
            s += get_U(sim.sv, c)[di] * V; vol += V
        end
        ρ̄ = s / vol
    end
    # Comoving Poisson carries a 1/a factor: ∇²φ = (4πG/a)(ρ − ρ̄). With the
    # cosmology G-normalization (4πG = 1) this is (1/a)(ρ − ρ̄); a = aⁿ (the cached
    # value at the current time, φ held over the step). Non-cosmological: a = 1.
    inv_a = sim.cosmo === nothing ? 1.0 : 1.0 / sim.cosmo.a
    for_each_cell(b) do c
        ρ = get_U(sim.sv, c)[di]
        grav.bv[c] = -FOURπ * grav.G * (ρ - ρ̄) * cell_volume(b, c) * inv_a
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
    _fill_poisson_rhs!(sim, grav)
    # r = b − A·φ0  (warm start from current φ)
    apply_laplacian!(sim, grav, grav.phiv, grav.Apv)
    for_each_cell(sim.backend) do c
        grav.rv[c] = grav.bv[c] - grav.Apv[c]
    end
    copy_cells!(sim, grav.pv, grav.rv)
    rsold = dot_cells(sim, grav.rv, grav.rv)
    bnorm = sqrt(dot_cells(sim, grav.bv, grav.bv))
    bnorm = bnorm == 0.0 ? 1.0 : bnorm
    iters = 0
    relres = sqrt(rsold) / bnorm
    for k in 1:grav.maxiter
        iters = k
        apply_laplacian!(sim, grav, grav.pv, grav.Apv)
        pAp = dot_cells(sim, grav.pv, grav.Apv)
        pAp <= 0.0 && break                        # lost SPD-ness numerically: stop
        α = rsold / pAp
        axpy_cells!(sim, α, grav.pv, grav.phiv)    # φ += α p
        axpy_cells!(sim, -α, grav.Apv, grav.rv)    # r -= α Ap
        rsnew = dot_cells(sim, grav.rv, grav.rv)
        relres = sqrt(rsnew) / bnorm
        relres <= grav.tol && break
        _xpby_cells!(sim, grav.rv, rsnew / rsold, grav.pv)  # p = r + β p
        rsold = rsnew
    end
    grav.project_mean && project_zero_mean!(sim, grav.phiv)
    return iters, relres
end

# ── gravitational acceleration g = −∇φ, and the gas source term ──────────────
# Cell-centered g along each axis from the two face neighbours of the φ field,
# using the level-aware center distances. At a φ domain boundary (homogeneous
# Neumann default) the missing side contributes zero gradient. Returns an
# NTuple{3} (unused axes are 0).
@inline function gravity_accel(sim::Simulation, grav::GravityField, cell)
    b = sim.backend
    N = rank(b)
    φc = grav.phiv[cell]
    return ntuple(3) do d
        d > N && return 0.0
        # central difference: (φ_hi − φ_lo) / (d_lo + d_hi)
        nlo = neighbor(b, cell, d, :lo; bcs = grav.bcs)
        nhi = neighbor(b, cell, d, :hi; bcs = grav.bcs)
        wlo = nlo isa Interior ? grav.phiv[nlo.cell] : φc
        whi = nhi isa Interior ? grav.phiv[nhi.cell] : φc
        dlo = nlo isa Interior ? _face_distance(b, cell, nlo.cell, d) : 0.5 * cell_width(b, cell)[d]
        dhi = nhi isa Interior ? _face_distance(b, cell, nhi.cell, d) : 0.5 * cell_width(b, cell)[d]
        -(whi - wlo) / (dlo + dhi)            # g_d = −∂φ/∂x_d
    end
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
    b = sim.backend
    av = sim.accv
    mom = momentum_indices(sim.model)
    ei = energy_index(sim.model)
    di = density_index(sim.model)
    for_each_cell(b; level = level) do c
        U = get_U(sim.sv, c)
        ρ = U[di]
        V = cell_volume(b, c)
        g = gravity_accel(sim, grav, c)
        @inbounds for d in 1:rank(b)
            mi = mom[d]
            av[mi][c] -= ρ * g[d] * V               # momentum source +ρg
            av[ei][c] -= U[mi] * g[d] * V            # energy source +ρv·g  (U[mi]=ρv_d)
        end
    end
    return nothing
end

# Free-fall timestep limiter, folded into the CFL reduction: dt ≲ η/√(4πGρ).
@inline _gravity_invdt(grav::GravityField, ρ::Float64) =
    ρ > 0 ? sqrt(FOURπ * grav.G * ρ) / GRAV_DT_ETA : 0.0
const GRAV_DT_ETA = 0.5
