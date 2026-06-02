# In-loop diagnostics and a minimal field extractor. These run over the live
# state through the seam (ADR P11) — no dump/reload, handle-based so they work on
# any backend regardless of cell ordering. Conservation tallies are the core
# acceptance check for the finite-volume scheme.

"Primitive state at a cell handle (the model's `cons2prim` of the conserved state)."
primitive_at(sim::Simulation, cell) = cons2prim(sim.model, get_U(sim.sv, cell))

"""
    conserved_totals(sim) -> NamedTuple

Volume-weighted sums of the conserved quantities over all cells (order-
independent). Constant in time to round-off under the conservative update with
non-leaky boundaries.
"""
function conserved_totals(sim::Simulation)
    b = sim.backend
    nv = nvars(sim.model)
    tot = zeros(Float64, nv)
    for_each_cell(b) do cell
        U = get_U(sim.sv, cell)
        v = cell_volume(b, cell)
        @inbounds for i in 1:nv
            tot[i] += U[i] * v
        end
    end
    mom = momentum_indices(sim.model)
    return (mass = tot[density_index(sim.model)],
            momentum_x = tot[mom[1]], momentum_y = tot[mom[2]], momentum_z = tot[mom[3]],
            energy = tot[energy_index(sim.model)])
end

"""
    cell_samples(sim) -> Vector{Tuple{NTuple{N,Float64},NTuple{5,Float64}}}

Per-cell `(center, primitive)` samples over the live state, in backend order.
Order-independent analyses (e.g. L1 error vs an analytic solution) consume this
directly; `dump_fields` sorts it for 1D plotting.
"""
function cell_samples(sim::Simulation)
    b = sim.backend
    N = rank(b)
    out = Tuple{NTuple{N,Float64},NTuple{nvars(sim.model),Float64}}[]
    for_each_cell(b) do cell
        push!(out, (cell_center(b, cell), primitive_at(sim, cell)))
    end
    return out
end

"""
    dump_fields(sim) -> NamedTuple

For a 1D simulation: interior primitive fields and the cell-center x-coordinate,
sorted by x. Convenient for plotting and error checks; read live from the running
state.
"""
function dump_fields(sim::Simulation)
    rank(sim.backend) == 1 ||
        throw(ArgumentError("dump_fields currently supports 1D; use cell_samples for ND"))
    s = cell_samples(sim)
    sort!(s; by = t -> t[1][1])
    x = [t[1][1] for t in s]
    ρ = [t[2][1] for t in s]
    vx = [t[2][2] for t in s]
    p = [t[2][5] for t in s]
    return (x = x, density = ρ, velocity_x = vx, pressure = p)
end
