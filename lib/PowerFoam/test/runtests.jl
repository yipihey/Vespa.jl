using Test
using PowerFoam
using KernelAbstractions

@testset "PowerFoam 2D prototype" begin
    @testset "zero-weight Voronoi split" begin
        pts = [0.25 0.5;
               0.75 0.5]
        mesh = power_diagram(PowerSites2D(pts))
        a = sort(cell_areas(mesh))
        @test a ≈ [0.5, 0.5]
        q = cell_quality(mesh)
        @test maximum(q.centroid_offset) < 1e-12
        ft = arepo_face_table(mesh)
        interior = findall(ft.c2 .> 0)
        @test length(interior) == 1
        f = interior[1]
        @test ft.area[f] ≈ 1.0
        @test ft.center[f, :] ≈ [0.5, 0.5]
    end

    @testset "power weight moves a bisector" begin
        pts = [0.25 0.5;
               0.75 0.5]
        mesh = power_diagram(PowerSites2D(pts; weights = [0.1, 0.0]))
        @test sort(cell_areas(mesh)) ≈ [0.4, 0.6]
        @test cell_areas(mesh)[1] ≈ 0.6
    end

    @testset "AREPO polygon import preserves face table" begin
        pts = [0.25 0.5;
               0.75 0.5]
        mesh = power_diagram(PowerSites2D(pts))
        imported = from_arepo_polygons(mesh.cells; generators = pts)
        @test cell_areas(imported) ≈ cell_areas(mesh)
        @test length(imported.faces.c1) == length(mesh.faces.c1)
        @test sort(imported.faces.area) ≈ sort(mesh.faces.area)
        @test imported.neighbors == mesh.neighbors
    end

    @testset "refine-patch metrics are finite" begin
        patch = refine_patch(6, 6; refine_radius = 0.2)
        pts = patch.points
        @test sum(patch.target_areas) ≈ 1.0
        @test refine_patch_points(6, 6; refine_radius = 0.2) == pts
        mesh = power_diagram(PowerSites2D(pts))
        q = mesh_quality(mesh)
        @test q.cells == size(pts, 1)
        @test q.volume ≈ 1.0
        @test isfinite(q.recon_cond_median)
        @test q.neighbor_count_max >= q.neighbor_count_min
    end

    @testset "weight relaxation reduces target area error" begin
        pts = [0.25 0.5;
               0.75 0.5]
        target = [0.6, 0.4]
        before = maximum(abs.(cell_areas(power_diagram(PowerSites2D(pts))) .- target))
        result = relax_weights(pts, target; steps = 8, gain = 0.7)
        after = maximum(abs.(cell_areas(result.mesh) .- target))
        @test after < before
        @test after < 0.02
    end

    @testset "smoothed face-aware relaxation controls tiny faces" begin
        patch = refine_patch(8, 8; refine_radius = 0.22)
        vor = power_diagram(PowerSites2D(patch.points))
        area_only = relax_weights(patch.points, patch.target_areas; steps = 20, gain = 0.35,
                                  small_face_weight = 0.0, compactness_weight = 0.0).mesh
        smoothed = relax_weights(patch.points, patch.target_areas; steps = 20, gain = 0.35,
                                 small_face_weight = 0.03, small_face_floor = 1e-4,
                                 compactness_weight = 0.02,
                                 smooth_strength = 0.5, smooth_passes = 1).mesh

        rms(mesh) = begin
            rel = (cell_areas(mesh) .- patch.target_areas) ./ patch.target_areas
            sqrt(sum(abs2, rel) / length(rel))
        end
        @test rms(smoothed) < rms(vor)
        @test mesh_quality(smoothed).small_face_p01 > mesh_quality(area_only).small_face_p01
    end

    @testset "velocity alignment metric prefers parallel or perpendicular faces" begin
        pts = [0.25 0.5;
               0.75 0.5]
        mesh = power_diagram(PowerSites2D(pts))
        parallel = face_velocity_alignment(mesh; velocity_field = _ -> (1.0, 0.0))
        perpendicular = face_velocity_alignment(mesh; velocity_field = _ -> (0.0, 1.0))
        diagonal = face_velocity_alignment(mesh; velocity_field = _ -> (1.0, 1.0))

        @test parallel.loss < 1e-12
        @test perpendicular.loss < 1e-12
        @test diagonal.loss > 0.4
        @test diagonal.middle_fraction > parallel.middle_fraction

        target = cell_areas(mesh)
        noflow = mesh_loss(mesh, target)
        flow = mesh_loss(mesh, target; velocity_alignment_weight = 0.5,
                         velocity_field = _ -> (1.0, 1.0))
        @test flow.velocity_alignment ≈ diagonal.loss
        @test flow.total > noflow.total
    end

    @testset "velocity alignment point relaxation reduces ambiguous faces" begin
        pts = Matrix{Float64}(undef, 25, 2)
        q = 1
        for j in 1:5, i in 1:5
            pts[q, 1] = (i - 0.5) / 5 + 0.015 * sin(17i + 3j)
            pts[q, 2] = (j - 0.5) / 5 + 0.015 * sin(5i + 19j)
            q += 1
        end
        before = power_diagram(PowerSites2D(pts))
        before_align = face_velocity_alignment(before; velocity_field = _ -> (1.0, 1.0))
        result = relax_points_velocity_alignment(pts;
            velocity_field = _ -> (1.0, 1.0),
            steps = 4, strength = 0.25, displacement_weight = 0.0)
        after_align = face_velocity_alignment(result.mesh; velocity_field = _ -> (1.0, 1.0))
        @test after_align.loss < before_align.loss
    end

    @testset "SVG writer emits an artifact" begin
        pts = [0.25 0.5;
               0.75 0.5]
        mesh = power_diagram(PowerSites2D(pts))
        path = tempname() * ".svg"
        write_svg(path, mesh; values = cell_areas(mesh))
        @test isfile(path)
        @test occursin("<svg", read(path, String))
    end

    @testset "AREPO face-table hydro step is conservative" begin
        pts = [0.25 0.5;
               0.75 0.5]
        mesh = power_diagram(PowerSites2D(pts))
        geom = arepo_mesh_arrays(mesh; T = Float64)

        uniform = euler_state_2d(mesh; rho = 1.0, vx = 0.0, vy = 0.0,
                                 pressure = 1.0, gamma = 1.4)
        uniform0 = total_conserved_2d(uniform, geom)
        finite_volume_step_2d!(uniform, geom; dt = 0.01, gamma = 1.4, riemann = :hll)
        uniform1 = total_conserved_2d(uniform, geom)
        @test uniform1.mass ≈ uniform0.mass
        @test uniform1.mx ≈ uniform0.mx
        @test uniform1.my ≈ uniform0.my
        @test uniform1.energy ≈ uniform0.energy
        @test all(conserved_to_primitive_2d(uniform; gamma = 1.4).pressure .> 0)

        contact = euler_state_2d(mesh; rho = [1.0, 2.0], vx = 0.0, vy = 0.0,
                                 pressure = 1.0, gamma = 1.4)
        total0 = total_conserved_2d(contact, geom)
        finite_volume_step_2d!(contact, geom; dt = 0.01, gamma = 1.4, riemann = :hll)
        total1 = total_conserved_2d(contact, geom)
        @test total1.mass ≈ total0.mass
        @test total1.mx ≈ total0.mx
        @test total1.my ≈ total0.my
        @test total1.energy ≈ total0.energy
        @test all(conserved_to_primitive_2d(contact; gamma = 1.4).rho .> 0)

        llf = euler_state_2d(mesh; rho = [1.0, 2.0], vx = 0.0, vy = 0.0,
                             pressure = 1.0, gamma = 1.4)
        finite_volume_step_2d!(llf, geom; dt = 0.01, gamma = 1.4, riemann = :llf)
        @test all(conserved_to_primitive_2d(llf; gamma = 1.4).rho .> 0)
    end

    @testset "AREPO hydro arrays stage through KA CPU backend" begin
        pts = [0.25 0.5;
               0.75 0.5]
        mesh = power_diagram(PowerSites2D(pts))
        geom = to_backend(KernelAbstractions.CPU(), arepo_mesh_arrays(mesh); T = Float32)
        state = to_backend(KernelAbstractions.CPU(),
                           euler_state_2d(mesh; rho = [1.0, 2.0], pressure = 1.0, gamma = 1.4);
                           T = Float32)
        work = hydro_work_2d(state, geom)
        finite_volume_step_2d!(state, geom; dt = 0.01f0, gamma = 1.4, riemann = :hll,
                               work)
        @test eltype(state.D) === Float32
        @test length(work.FD) == length(geom.c1)
        @test all(conserved_to_primitive_2d(state; gamma = 1.4).rho .> 0)
    end

    @testset "moving mesh ALE step rebuilds geometry and conserves integrals" begin
        pts = [0.25 0.5;
               0.75 0.5]
        mesh = power_diagram(PowerSites2D(pts))
        geom = arepo_mesh_arrays(mesh; T = Float64)

        static_state = euler_state_2d(mesh; rho = [1.0, 2.0], vx = 0.0, vy = 0.0,
                                      pressure = 1.0, gamma = 1.4)
        moving_state = euler_state_2d(mesh; rho = [1.0, 2.0], vx = 0.0, vy = 0.0,
                                      pressure = 1.0, gamma = 1.4)
        finite_volume_step_2d!(static_state, geom; dt = 0.01, gamma = 1.4, riemann = :hll)
        zero_v = zeros(2, 2)
        moved0 = moving_mesh_step_2d!(moving_state, mesh; dt = 0.01, gamma = 1.4,
                                      riemann = :hll, mesh_velocity = zero_v)
        @test moved0.mesh.generators ≈ mesh.generators
        @test moving_state.D ≈ static_state.D
        @test moving_state.Mx ≈ static_state.Mx
        @test moving_state.My ≈ static_state.My
        @test moving_state.E ≈ static_state.E

        state = euler_state_2d(mesh; rho = [1.0, 2.0], vx = 0.0, vy = 0.0,
                               pressure = 1.0, gamma = 1.4)
        total0 = total_conserved_2d(state, geom)
        vmesh = [0.1 0.0;
                 0.1 0.0]
        moved = moving_mesh_step_2d!(state, mesh; dt = 0.05, gamma = 1.4,
                                     riemann = :hll, mesh_velocity = vmesh)
        total1 = total_conserved_2d(state, moved.geom)
        @test moved.mesh.generators ≈ pts .+ 0.05 .* vmesh
        @test total1.mass ≈ total0.mass
        @test total1.mx ≈ total0.mx
        @test total1.my ≈ total0.my
        @test total1.energy ≈ total0.energy
        @test all(conserved_to_primitive_2d(state; gamma = 1.4).rho .> 0)
    end

    @testset "3D periodic Cartesian face-table hydro is conservative" begin
        geom = cartesian_periodic_mesh_arrays_3d(3; T = Float64)
        @test length(geom.volume) == 27
        @test length(geom.c1) == 81
        @test sum(geom.volume) ≈ 1.0
        @test all(diff(Int.(geom.cell_face_offsets)) .== 6)

        uniform = euler_state_3d(geom; rho = 1.0, vx = 0.1, vy = -0.2, vz = 0.05,
                                 pressure = 1.0, gamma = 1.4)
        total0 = total_conserved_3d(uniform, geom)
        finite_volume_step_3d!(uniform, geom; dt = 0.005, gamma = 1.4, riemann = :hll)
        total1 = total_conserved_3d(uniform, geom)
        @test total1.mass ≈ total0.mass
        @test total1.mx ≈ total0.mx
        @test total1.my ≈ total0.my
        @test total1.mz ≈ total0.mz
        @test total1.energy ≈ total0.energy
        @test all(conserved_to_primitive_3d(uniform; gamma = 1.4).pressure .> 0)

        rho = [i <= 13 ? 1.0 : 2.0 for i in eachindex(geom.volume)]
        contact = euler_state_3d(geom; rho, vx = 0.0, vy = 0.0, vz = 0.0,
                                 pressure = 1.0, gamma = 1.4)
        total0 = total_conserved_3d(contact, geom)
        finite_volume_step_3d!(contact, geom; dt = 0.005, gamma = 1.4, riemann = :llf)
        total1 = total_conserved_3d(contact, geom)
        @test total1.mass ≈ total0.mass
        @test total1.energy ≈ total0.energy
        @test all(conserved_to_primitive_3d(contact; gamma = 1.4).rho .> 0)
    end

    @testset "3D bounded Voronoi rebuilds a native face table" begin
        two = bounded_voronoi_mesh_arrays_3d([0.25 0.5 0.5;
                                              0.75 0.5 0.5]; T = Float64)
        @test two.volume ≈ [0.5, 0.5]
        @test sum(two.geom.volume) ≈ 1.0
        @test length(two.geom.c1) == 11
        @test count(>(0), two.geom.c2) == 1
        f = only(findall(two.geom.c2 .> 0))
        @test two.geom.face_area[f] ≈ 1.0
        @test two.face_center[f, :] ≈ [0.5, 0.5, 0.5]
        @test Int.(diff(two.geom.cell_face_offsets)) == [6, 6]

        pts = Matrix{Float64}(undef, 8, 3)
        q = 1
        for k in (0.25, 0.75), j in (0.25, 0.75), i in (0.25, 0.75)
            pts[q, :] .= (i, j, k)
            q += 1
        end
        mesh = bounded_voronoi_mesh_arrays_3d(pts; T = Float64)
        @test mesh.volume ≈ fill(1 / 8, 8)
        @test sum(mesh.geom.volume) ≈ 1.0
        @test length(mesh.geom.c1) == 36
        @test count(>(0), mesh.geom.c2) == 12
        @test all(diff(Int.(mesh.geom.cell_face_offsets)) .== 6)
    end

    @testset "3D periodic native Voronoi matches Cartesian lattice topology" begin
        pts = Matrix{Float64}(undef, 8, 3)
        q = 1
        for k in (0.25, 0.75), j in (0.25, 0.75), i in (0.25, 0.75)
            pts[q, :] .= (i, j, k)
            q += 1
        end
        mesh = periodic_voronoi_mesh_arrays_3d(pts; T = Float64)
        @test mesh.volume ≈ fill(1 / 8, 8)
        @test sum(mesh.geom.volume) ≈ 1.0
        @test length(mesh.geom.c1) == 24
        @test count(>(0), mesh.geom.c2) == 24
        @test all(diff(Int.(mesh.geom.cell_face_offsets)) .== 6)
        @test sort(mesh.geom.face_area) ≈ fill(0.25, 24)

        cart = cartesian_periodic_mesh_arrays_3d(2; T = Float64)
        @test sort(mesh.geom.face_area) ≈ sort(cart.face_area)
        @test sort(mesh.geom.volume) ≈ sort(cart.volume)

        local_mesh = local_periodic_voronoi_mesh_arrays_3d(pts; T = Float64,
                                                           bins_per_axis = 2,
                                                           search_radius = 1)
        @test local_mesh.volume ≈ fill(1 / 8, 8)
        @test sum(local_mesh.geom.volume) ≈ 1.0
        @test length(local_mesh.geom.c1) == 24
        @test count(>(0), local_mesh.geom.c2) == 24
        @test all(diff(Int.(local_mesh.geom.cell_face_offsets)) .== 6)
        @test sort(local_mesh.geom.face_area) ≈ sort(cart.face_area)
    end

    @testset "3D local periodic Voronoi scales past all-pairs gate" begin
        n = 4
        pts = Matrix{Float64}(undef, n^3, 3)
        q = 1
        for k in 1:n, j in 1:n, i in 1:n
            pts[q, 1] = (i - 0.5) / n + 0.01 * sin(17i + 3j + 5k) / n
            pts[q, 2] = (j - 0.5) / n + 0.01 * sin(7i + 19j + 2k) / n
            pts[q, 3] = (k - 0.5) / n + 0.01 * sin(11i + 13j + 23k) / n
            q += 1
        end
        mesh = local_periodic_voronoi_mesh_arrays_3d(pts; T = Float64,
                                                     bins_per_axis = n,
                                                     search_radius = 1)
        counts = diff(Int.(mesh.geom.cell_face_offsets))
        @test length(mesh.geom.volume) == n^3
        @test sum(mesh.geom.volume) ≈ 1.0 atol = 1e-8
        @test minimum(mesh.geom.volume) > 0
        @test minimum(counts) >= 6
        @test maximum(counts) < 32
    end

    @testset "3D native moving mesh rebuild preserves zero-velocity step" begin
        pts = [0.25 0.5 0.5;
               0.75 0.5 0.5]
        mesh = bounded_voronoi_mesh_arrays_3d(pts; T = Float64)
        geom = mesh.geom
        static_state = euler_state_3d(geom; rho = [1.0, 2.0],
                                      vx = 0.0, vy = 0.0, vz = 0.0,
                                      pressure = 1.0, gamma = 1.4)
        moving_state = euler_state_3d(geom; rho = [1.0, 2.0],
                                      vx = 0.0, vy = 0.0, vz = 0.0,
                                      pressure = 1.0, gamma = 1.4)
        finite_volume_step_3d!(static_state, geom; dt = 0.01, gamma = 1.4,
                               riemann = :hll)
        moved = moving_mesh_step_3d!(moving_state, pts; dt = 0.01, gamma = 1.4,
                                     riemann = :hll, mesh_velocity = zeros(2, 3))
        @test moved.points ≈ pts
        @test moving_state.D ≈ static_state.D
        @test moving_state.Mx ≈ static_state.Mx
        @test moving_state.My ≈ static_state.My
        @test moving_state.Mz ≈ static_state.Mz
        @test moving_state.E ≈ static_state.E
        @test total_conserved_3d(moving_state, moved.geom).mass ≈
              total_conserved_3d(static_state, geom).mass
    end

    @testset "3D periodic native moving mesh rebuild preserves zero-velocity step" begin
        pts = Matrix{Float64}(undef, 8, 3)
        q = 1
        for k in (0.25, 0.75), j in (0.25, 0.75), i in (0.25, 0.75)
            pts[q, :] .= (i, j, k)
            q += 1
        end
        mesh = periodic_voronoi_mesh_arrays_3d(pts; T = Float64)
        geom = mesh.geom
        rho = [i <= 4 ? 1.0 : 2.0 for i in 1:8]
        static_state = euler_state_3d(geom; rho, vx = 0.0, vy = 0.0, vz = 0.0,
                                      pressure = 1.0, gamma = 1.4)
        moving_state = euler_state_3d(geom; rho, vx = 0.0, vy = 0.0, vz = 0.0,
                                      pressure = 1.0, gamma = 1.4)
        finite_volume_step_3d!(static_state, geom; dt = 0.005, gamma = 1.4,
                               riemann = :hll)
        moved = moving_mesh_step_3d!(moving_state, pts; dt = 0.005, gamma = 1.4,
                                     boundary = :periodic, riemann = :hll,
                                     mesh_velocity = zeros(8, 3))
        @test moved.points ≈ pts
        @test length(moved.geom.c1) == length(geom.c1)
        @test moving_state.D ≈ static_state.D
        @test moving_state.Mx ≈ static_state.Mx
        @test moving_state.My ≈ static_state.My
        @test moving_state.Mz ≈ static_state.Mz
        @test moving_state.E ≈ static_state.E
    end

    @testset "3D hydro arrays stage through KA CPU backend" begin
        geom = cartesian_periodic_mesh_arrays_3d(2; T = Float64)
        state = euler_state_3d(geom; rho = 1.0, vx = 0.1, vy = 0.0, vz = 0.0,
                               pressure = 1.0, gamma = 1.4)
        dgeom = to_backend(KernelAbstractions.CPU(), geom; T = Float32)
        dstate = to_backend(KernelAbstractions.CPU(), state; T = Float32)
        work = hydro_work_3d(dstate, dgeom)
        finite_volume_step_3d!(dstate, dgeom; dt = 0.001f0, gamma = 1.4,
                               riemann = :hll, work)
        @test eltype(dstate.D) === Float32
        @test length(work.FE) == length(dgeom.c1)
        @test total_conserved_3d(dstate, dgeom).mass ≈ 1.0f0
    end

    @testset "3D face predictor reconstructs a linear field" begin
        n = 2
        geom = cartesian_periodic_mesh_arrays_3d(n; T = Float64)
        nc = length(geom.volume)
        nf = length(geom.c1)
        center = Matrix{Float64}(undef, nc, 3)
        q = 1
        for k in 1:n, j in 1:n, i in 1:n
            center[q, :] .= ((i - 0.5) / n, (j - 0.5) / n, (k - 0.5) / n)
            q += 1
        end
        face_center = Matrix{Float64}(undef, nf, 3)
        f = 1
        for k in 1:n, j in 1:n, i in 1:n
            face_center[f, :] .= (i / n, (j - 0.5) / n, (k - 0.5) / n); f += 1
            face_center[f, :] .= ((i - 0.5) / n, j / n, (k - 0.5) / n); f += 1
            face_center[f, :] .= ((i - 0.5) / n, (j - 0.5) / n, k / n); f += 1
        end
        ax, ay, az = 0.1, 0.2, -0.05
        rho = @. 1.0 + ax * center[:, 1] + ay * center[:, 2] + az * center[:, 3]
        zero_cell = zeros(nc)
        one_cell = ones(nc)
        gradients = HydroGradients3D(fill(ax, nc), fill(ay, nc), fill(az, nc),
                                     zero_cell, zero_cell, zero_cell,
                                     zero_cell, zero_cell, zero_cell,
                                     zero_cell, zero_cell, zero_cell,
                                     zero_cell, zero_cell, zero_cell)
        states = face_prediction_work_3d(geom)
        predict_face_states_3d!(states, geom, gradients, rho,
                                zero_cell, zero_cell, zero_cell, one_cell,
                                center, face_center; box_size = 1.0, gamma = 1.4)
        pred = face_states_to_arrays(states)
        expected_left = similar(rho, nf)
        expected_right = similar(rho, nf)
        for f in 1:nf
            i = Int(geom.c1[f])
            j = Int(geom.c2[f])
            dL = face_center[f, :] .- center[i, :]
            dR = face_center[f, :] .- center[j, :]
            dL .= ifelse.(dL .> 0.5, dL .- 1.0, ifelse.(dL .< -0.5, dL .+ 1.0, dL))
            dR .= ifelse.(dR .> 0.5, dR .- 1.0, ifelse.(dR .< -0.5, dR .+ 1.0, dR))
            expected_left[f] = rho[i] + ax * dL[1] + ay * dL[2] + az * dL[3]
            expected_right[f] = rho[j] + ax * dR[1] + ay * dR[2] + az * dR[3]
        end
        @test pred.left.rho ≈ expected_left
        @test pred.right.rho ≈ expected_right
        @test pred.left.vx ≈ zeros(nf)
        @test pred.right.pressure ≈ ones(nf)
    end

    @testset "3D primitive backend work matches host recovery" begin
        geom = cartesian_periodic_mesh_arrays_3d(2; T = Float64)
        nc = length(geom.volume)
        rho = range(0.9, 1.1; length = nc)
        vx = range(-0.02, 0.03; length = nc)
        vy = range(0.01, -0.04; length = nc)
        vz = range(0.04, 0.0; length = nc)
        pressure = range(0.7, 0.9; length = nc)
        state = euler_state_3d(geom; rho, vx, vy, vz, pressure, gamma = 1.4)
        expected = conserved_to_primitive_3d(state; gamma = 1.4)
        prim = primitive_work_3d(state)
        conserved_to_primitive_3d!(prim, state; gamma = 1.4)
        got = primitive_to_arrays_3d(prim)
        @test got.rho ≈ expected.rho
        @test got.vx ≈ expected.vx
        @test got.vy ≈ expected.vy
        @test got.vz ≈ expected.vz
        @test got.pressure ≈ expected.pressure
    end

    @testset "3D mesh-derived gradient connections recover smooth slopes" begin
        n = 4
        geom = cartesian_periodic_mesh_arrays_3d(n; T = Float64)
        nc = length(geom.volume)
        nf = length(geom.c1)
        center = Matrix{Float64}(undef, nc, 3)
        q = 1
        for k in 1:n, j in 1:n, i in 1:n
            center[q, :] .= ((i - 0.5) / n, (j - 0.5) / n, (k - 0.5) / n)
            q += 1
        end
        face_center = Matrix{Float64}(undef, nf, 3)
        f = 1
        for k in 1:n, j in 1:n, i in 1:n
            face_center[f, :] .= (i / n, (j - 0.5) / n, (k - 0.5) / n); f += 1
            face_center[f, :] .= ((i - 0.5) / n, j / n, (k - 0.5) / n); f += 1
            face_center[f, :] .= ((i - 0.5) / n, (j - 0.5) / n, k / n); f += 1
        end
        inside = [all(0.26 .< center[i, :] .< 0.74) for i in 1:nc]
        ax, ay, az = 0.1, -0.07, 0.04
        rho = @. 1.0 + ax * center[:, 1] + ay * center[:, 2] + az * center[:, 3]
        vx = @. 0.2 - 0.03 * center[:, 1]
        vy = @. -0.1 + 0.05 * center[:, 2]
        vz = @. 0.04 + 0.02 * center[:, 3]
        pressure = @. 0.7 + 0.08 * center[:, 1]
        state = euler_state_3d(geom; rho, vx, vy, vz, pressure, gamma = 1.4)
        prim = conserved_to_primitive_3d(state; gamma = 1.4)
        conn = gradient_connections_from_mesh_3d(geom, center, face_center,
                                                 prim.rho, prim.vx, prim.vy,
                                                 prim.vz, prim.pressure)
        gradients = hydro_gradient_work_3d(rho)
        calculate_gradients_3d!(gradients, conn, prim.rho, prim.vx, prim.vy,
                                prim.vz, prim.pressure, center;
                                box_size = 0.0, gamma = 1.4)
        g = hydro_gradients_to_arrays(gradients)
        pwork = primitive_work_3d(state)
        conserved_to_primitive_3d!(pwork, state; gamma = 1.4)
        direct = hydro_gradient_work_3d(pwork.rho)
        calculate_gradients_from_mesh_3d!(direct, geom, pwork, center,
                                          face_center; box_size = 0.0,
                                          gamma = 1.4)
        gd = hydro_gradients_to_arrays(direct)
        direct_cols = hydro_gradient_work_3d(pwork.rho)
        calculate_gradients_from_mesh_3d!(direct_cols, geom, pwork,
                                          collect(view(center, :, 1)),
                                          collect(view(center, :, 2)),
                                          collect(view(center, :, 3)),
                                          collect(view(face_center, :, 1)),
                                          collect(view(face_center, :, 2)),
                                          collect(view(face_center, :, 3));
                                          box_size = 0.0, gamma = 1.4)
        gdc = hydro_gradients_to_arrays(direct_cols)
        active_stride = maximum(Int.(diff(geom.cell_face_offsets)))
        active_counts = Int32.(diff(geom.cell_face_offsets))
        active_faces = zeros(Int32, active_stride * nc)
        active_signs = zeros(Int32, active_stride * nc)
        for i in 1:nc
            p0 = Int(geom.cell_face_offsets[i])
            p1 = Int(geom.cell_face_offsets[i + 1]) - 1
            for (q, p) in enumerate(p0:p1)
                active_faces[(i - 1) * active_stride + q] = Int32(geom.cell_faces[p])
                active_signs[(i - 1) * active_stride + q] = Int32(geom.cell_face_signs[p])
            end
        end
        direct_active = hydro_gradient_work_3d(pwork.rho)
        calculate_gradients_from_mesh_activecells_3d!(
            direct_active, geom, pwork,
            collect(view(center, :, 1)), collect(view(center, :, 2)),
            collect(view(center, :, 3)),
            collect(view(face_center, :, 1)), collect(view(face_center, :, 2)),
            collect(view(face_center, :, 3)),
            active_counts, active_faces, active_signs;
            active_stride, box_size = 0.0, gamma = 1.4)
        gda = hydro_gradients_to_arrays(direct_active)
        @test g.drho[inside, 1] ≈ fill(ax, count(inside)) atol = 1e-12
        @test g.drho[inside, 2] ≈ fill(ay, count(inside)) atol = 1e-12
        @test g.drho[inside, 3] ≈ fill(az, count(inside)) atol = 1e-12
        @test g.dpress[inside, 1] ≈ fill(0.08, count(inside)) atol = 1e-12
        @test gd.drho[inside, 1] ≈ fill(ax, count(inside)) atol = 1e-12
        @test gd.drho[inside, 2] ≈ fill(ay, count(inside)) atol = 1e-12
        @test gd.drho[inside, 3] ≈ fill(az, count(inside)) atol = 1e-12
        @test gd.dpress[inside, 1] ≈ fill(0.08, count(inside)) atol = 1e-12
        @test gdc.drho ≈ gd.drho
        @test gdc.dvel ≈ gd.dvel
        @test gdc.dpress ≈ gd.dpress
        @test gda.drho ≈ gd.drho
        @test gda.dvel ≈ gd.dvel
        @test gda.dpress ≈ gd.dpress
    end

    @testset "3D reconstructed hydro step preserves uniform flow" begin
        n = 2
        geom = cartesian_periodic_mesh_arrays_3d(n; T = Float64)
        nc = length(geom.volume)
        nf = length(geom.c1)
        center = Matrix{Float64}(undef, nc, 3)
        q = 1
        for k in 1:n, j in 1:n, i in 1:n
            center[q, :] .= ((i - 0.5) / n, (j - 0.5) / n, (k - 0.5) / n)
            q += 1
        end
        face_center = Matrix{Float64}(undef, nf, 3)
        f = 1
        for k in 1:n, j in 1:n, i in 1:n
            face_center[f, :] .= (i / n, (j - 0.5) / n, (k - 0.5) / n); f += 1
            face_center[f, :] .= ((i - 0.5) / n, j / n, (k - 0.5) / n); f += 1
            face_center[f, :] .= ((i - 0.5) / n, (j - 0.5) / n, k / n); f += 1
        end
        z = zeros(nc)
        gradients = HydroGradients3D((copy(z) for _ in 1:15)...)
        state = euler_state_3d(geom; rho = 1.0, vx = 0.07, vy = -0.03,
                               vz = 0.02, pressure = 1.0, gamma = 1.4)
        total0 = total_conserved_3d(state, geom)
        finite_volume_reconstructed_step_3d!(state, geom, gradients, center,
                                             face_center; dt = 0.002,
                                             gamma = 1.4, riemann = :hll)
        state_from_prim = euler_state_3d(geom; rho = 1.0, vx = 0.07, vy = -0.03,
                                         vz = 0.02, pressure = 1.0, gamma = 1.4)
        prim = primitive_work_3d(state_from_prim)
        conserved_to_primitive_3d!(prim, state_from_prim; gamma = 1.4)
        finite_volume_reconstructed_step_3d!(state_from_prim, geom, gradients,
                                             prim, center, face_center;
                                             dt = 0.002, gamma = 1.4,
                                             riemann = :hll)
        state_from_cols = euler_state_3d(geom; rho = 1.0, vx = 0.07,
                                         vy = -0.03, vz = 0.02,
                                         pressure = 1.0, gamma = 1.4)
        prim_cols = primitive_work_3d(state_from_cols)
        conserved_to_primitive_3d!(prim_cols, state_from_cols; gamma = 1.4)
        finite_volume_reconstructed_step_3d!(
            state_from_cols, geom, gradients, prim_cols,
            collect(view(center, :, 1)), collect(view(center, :, 2)),
            collect(view(center, :, 3)),
            collect(view(face_center, :, 1)), collect(view(face_center, :, 2)),
            collect(view(face_center, :, 3));
            dt = 0.002, gamma = 1.4, riemann = :hll)
        active_stride = maximum(Int.(diff(geom.cell_face_offsets)))
        active_counts = Int32.(diff(geom.cell_face_offsets))
        active_faces = zeros(Int32, active_stride * nc)
        active_signs = zeros(Int32, active_stride * nc)
        for i in 1:nc
            p0 = Int(geom.cell_face_offsets[i])
            p1 = Int(geom.cell_face_offsets[i + 1]) - 1
            for (q, p) in enumerate(p0:p1)
                active_faces[(i - 1) * active_stride + q] = Int32(geom.cell_faces[p])
                active_signs[(i - 1) * active_stride + q] = Int32(geom.cell_face_signs[p])
            end
        end
        state_from_active = euler_state_3d(geom; rho = 1.0, vx = 0.07,
                                          vy = -0.03, vz = 0.02,
                                          pressure = 1.0, gamma = 1.4)
        prim_active = primitive_work_3d(state_from_active)
        conserved_to_primitive_3d!(prim_active, state_from_active; gamma = 1.4)
        finite_volume_reconstructed_step_activecells_3d!(
            state_from_active, geom, gradients, prim_active,
            collect(view(center, :, 1)), collect(view(center, :, 2)),
            collect(view(center, :, 3)),
            collect(view(face_center, :, 1)), collect(view(face_center, :, 2)),
            collect(view(face_center, :, 3)),
            active_counts, active_faces, active_signs;
            active_stride, dt = 0.002, gamma = 1.4, riemann = :hll)
        total1 = total_conserved_3d(state, geom)
        prim = conserved_to_primitive_3d(state; gamma = 1.4)
        prim_from_buffer = conserved_to_primitive_3d(state_from_prim; gamma = 1.4)
        prim_from_cols = conserved_to_primitive_3d(state_from_cols; gamma = 1.4)
        prim_from_active = conserved_to_primitive_3d(state_from_active; gamma = 1.4)
        @test total1.mass ≈ total0.mass
        @test total1.energy ≈ total0.energy
        @test prim.rho ≈ ones(nc)
        @test prim.pressure ≈ ones(nc)
        @test prim_from_buffer.rho ≈ ones(nc)
        @test prim_from_buffer.pressure ≈ ones(nc)
        @test prim_from_cols.rho ≈ ones(nc)
        @test prim_from_cols.pressure ≈ ones(nc)
        @test prim_from_active.rho ≈ ones(nc)
        @test prim_from_active.pressure ≈ ones(nc)
    end

    @testset "AREPO hydro timestep helper follows Courant radius" begin
        volume = fill(1 / 8, 8)
        pressure = ones(8)
        rho = ones(8)
        dt = arepo_hydro_dt_3d(volume, pressure, rho; gamma = 1.4,
                               courant = 0.3, max_dt = 1.0, min_dt = 1e-6)
        radius = cbrt(3 * volume[1] / (4pi))
        @test dt ≈ fill(0.3 * radius / sqrt(1.4), 8)
        @test arepo_timebin_3d([0.125, 0.25, 0.5]; timebase_interval = 0.125) == [0, 1, 2]
    end
end
