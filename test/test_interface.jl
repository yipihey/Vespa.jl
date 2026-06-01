# Backend-contract conformance. Written against the interface, not against a
# concrete backend, so `run_interface_conformance(make_backend)` validates any
# backend. Invoked for RefMesh here and for HGBackend in test_hgbackend.jl.

using MeshInterface: Interior, DomainBoundary

"""
    run_interface_conformance(make; label, layouts)

`make(dims, domain; bcs)` must return a fresh backend over the given uniform
domain (dims are powers of two so any backend, incl. the HG tree, can build
them). `bcs` is the BoundaryConditions the neighbor tests should resolve.
`layouts` is the set of field layouts the backend claims to support — RefMesh
supports all three; HGBackend's cell-average model is currently SoA-only.
"""
function run_interface_conformance(make; label::String,
                                   layouts = (SoA(), AoS(), Blocked{4}()))
    @testset "interface conformance [$label]" begin
        outflow = BoundaryConditions(Outflow(), Val(1))
        periodic = BoundaryConditions(Periodic(), Val(1))

        @testset "topology & derived geometry" begin
            m = make((8,), ((0.0, 1.0),); bcs = outflow)
            @test rank(m) == 1
            @test n_cells(m) == 8
            @test domain(m) == ((0.0, 1.0),)
            # collect leaf handles via the iterator
            cells = Any[]
            for_each_cell(m) do c
                push!(cells, c)
            end
            @test length(cells) == 8
            # widths and volumes are derived and uniform
            @test all(c -> cell_width(m, c)[1] ≈ 1 / 8, cells)
            @test all(c -> cell_volume(m, c) ≈ 1 / 8, cells)
            # centers span (dx/2, 1-dx/2)
            xs = sort([cell_center(m, c)[1] for c in cells])
            @test xs[1] ≈ 1 / 16
            @test xs[end] ≈ 1 - 1 / 16
        end

        @testset "neighbor resolution & BCs" begin
            m = make((4,), ((0.0, 1.0),); bcs = outflow)
            # find the leftmost and a middle cell by coordinate
            cells = Any[]
            for_each_cell(m) do c; push!(cells, c); end
            sort!(cells; by = c -> cell_center(m, c)[1])
            left, second = cells[1], cells[2]

            # interior neighbor on the hi side of the leftmost cell is the 2nd cell
            nb = neighbor(m, left, 1, :hi; bcs = outflow)
            @test nb isa Interior
            @test cell_center(m, nb.cell)[1] ≈ cell_center(m, second)[1]

            # lo side of the leftmost cell is a domain boundary (Outflow)
            nb = neighbor(m, left, 1, :lo; bcs = outflow)
            @test nb isa DomainBoundary && nb.bc isa Outflow

            # under periodic BCs the lo side wraps to the rightmost interior cell
            nb = neighbor(m, left, 1, :lo; bcs = periodic)
            @test nb isa Interior
            @test cell_center(m, nb.cell)[1] ≈ cell_center(m, cells[end])[1]
        end

        @testset "field allocation & round-trip ($lay)" for lay in layouts
            m = make((8,), ((0.0, 1.0),); bcs = outflow)
            store = allocate_fields(m, FieldSpec([:q]); layout = lay)
            q = field_view(m, store, :q)
            for_each_cell(m) do c
                q[c] = cell_center(m, c)[1]
            end
            vals = Float64[]
            for_each_cell(m) do c
                push!(vals, q[c])
            end
            @test sort(vals) ≈ [(i - 0.5) / 8 for i in 1:8]
        end
    end
end

@testset "RefMesh interface conformance" begin
    make(dims, dom; bcs) = UniformMesh(dims, dom)
    run_interface_conformance(make; label = "RefMesh")
end

@testset "RefMesh conservative restrict!/prolong! (P5)" begin
    coarse = UniformMesh((4,), ((0.0, 1.0),))
    fine   = UniformMesh((8,), ((0.0, 1.0),))
    cs = allocate_fields(coarse, FieldSpec([:q]))
    fs = allocate_fields(fine,   FieldSpec([:q]))

    qf = field_view(fine, fs, :q)
    for_each_cell(fine) do c
        qf[c] = sin(cell_center(fine, c)[1] * 7)
    end

    total(mesh, store) = begin
        s = 0.0
        for_each_cell(mesh) do c
            s += field_view(mesh, store, :q)[c] * cell_volume(mesh, c)
        end
        s
    end
    restrict!(coarse, cs, fs)
    @test total(coarse, cs) ≈ total(fine, fs) rtol = 1e-14   # Σ value×volume preserved

    # prolong (injection) then restrict (mean) is the identity on coarse data.
    qc = field_view(coarse, cs, :q)
    for_each_cell(coarse) do c
        qc[c] = cos(cell_center(coarse, c)[1] * 3)
    end
    cs2 = allocate_fields(coarse, FieldSpec([:q]))
    prolong!(coarse, fs, cs)
    restrict!(coarse, cs2, fs)
    diff = 0.0
    for_each_cell(coarse) do c
        diff = max(diff, abs(field_view(coarse, cs2, :q)[c] - field_view(coarse, cs, :q)[c]))
    end
    @test diff < 1e-14
end
