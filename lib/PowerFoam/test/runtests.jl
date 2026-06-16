using Test
using PowerFoam
using KernelAbstractions

function tiny_snapshot_payload()
    return (
        header = (
            time = 0.125,
            box_size = 1.0,
            num_files = 1,
        ),
        gas = (
            density = [1.0, 0.9, 1.1, 1.05],
            masses = fill(0.25, 4),
            internal_energy = [2.4, 2.3, 2.5, 2.45],
            velocities = [
                0.10  0.00  0.00
                0.00  0.20  0.00
               -0.10  0.00  0.10
                0.05 -0.10  0.00
            ],
            Coordinates = [
                0.125 0.125 0.125
                0.375 0.125 0.125
                0.125 0.375 0.125
                0.375 0.375 0.125
            ],
            particle_ids = collect(1:4),
        ),
    )
end

@testset "PowerFoam 2D prototype" begin
    @testset "AREPO runtime scaffold normalizes problem requests" begin
        opts = ArepoRunOptions(start_time = 0.125, final_time = 0.25,
                               max_steps = 4, cfl = 0.3,
                               output_interval = 2)
        @test opts.start_time == 0.125
        @test opts.final_time == 0.25
        @test opts.max_steps == 4
        @test opts.cfl == 0.3
        @test_throws ErrorException ArepoRunOptions(final_time = -1)
        @test_throws ErrorException ArepoRunOptions(cfl = 0.0)

        spec = arepo_problem_spec(:kh2d; dimensionality = 2,
                                  domain = ((-0.5, 0.5), (0.0, 2.0)),
                                  periodic = (true, false),
                                  gas_cell_count = 64,
                                  physics = (hydro = true,
                                             tessellation = true,
                                             gravity = false),
                                  metadata = (solver = :hll,))
        @test spec.name == :kh2d
        @test spec.dimensionality == 2
        @test spec.domain == ((-0.5, 0.5), (0.0, 2.0), (0.0, 1.0))
        @test spec.periodic == (true, false, false)
        @test spec.metadata.solver == :hll

        state = arepo_run_scaffold(spec; backend = :ka, options = opts)
        @test state.status == :unsupported
        @test state.time == opts.start_time
        @test :runtime_loop in state.unsupported
        @test :tessellation in state.unsupported
        @test :hydro in state.unsupported
        @test !(:gravity in state.unsupported)
        @test any(contains("planning stub"), state.diagnostics)
        @test state.payload.ka_hydro_smoke isa ArepoHydroSmokeAssessment
        @test state.payload.ka_hydro_smoke.eligible
        @test state.payload.ka_hydro_smoke.status == :eligible_with_tessellation_adapter
        @test :tessellation_adapter in state.payload.ka_hydro_smoke.requirements

        prebuilt_spec = arepo_problem_spec(:prebuilt3d; dimensionality = 3,
                                           gas_cell_count = 8,
                                           physics = (hydro = true,
                                                      tessellation = false,
                                                      gravity = false))
        prebuilt_smoke = classify_ka_hydro_smoke(prebuilt_spec)
        @test prebuilt_smoke.eligible
        @test prebuilt_smoke.status == :eligible

        bridge_smoke = classify_ka_hydro_smoke(prebuilt_spec; backend = :bridge)
        @test !bridge_smoke.eligible
        @test bridge_smoke.status == :needs_ka_backend

        gravity_spec = arepo_problem_spec(:cosmo; particle_count = 8,
                                          physics = (hydro = false,
                                                     tessellation = false,
                                                     gravity = true))
        gravity_state = arepo_run_scaffold(gravity_spec)
        @test :gravity in gravity_state.unsupported
        @test !gravity_state.payload.ka_hydro_smoke.eligible

        noh2d_spec = arepo_problem_spec(:noh2d; dimensionality = 2,
                                        domain = ((-3.0, 3.0), (-3.0, 3.0)),
                                        periodic = (false, false),
                                        gas_cell_count = 576,
                                        physics = (hydro = true,
                                                   tessellation = true,
                                                   gravity = false),
                                        metadata = (n_side = 24, t_final = 0.2,
                                                    nbins = 24, cfl = 0.18,
                                                    domain_radius = 3.0,
                                                    riemann = :hll))
        noh2d_state = arepo_run_scaffold(
            noh2d_spec; backend = :ka,
            options = ArepoRunOptions(final_time = 0.2, max_steps = 10_000,
                                      cfl = 0.18))
        @test isempty(noh2d_state.unsupported)
        @test noh2d_state.status == :calibration_pending
        @test noh2d_state.payload.standard_problem.status == "calibration-PENDING"
        @test noh2d_state.payload.standard_problem.final_metric.t ≈ 0.2 atol = 1e-12
        @test noh2d_state.payload.standard_problem.final_metric.rho_min > 0

        soundwave2d_spec = arepo_problem_spec(:soundwave2d; dimensionality = 2,
                                              domain = ((0.0, 1.0), (0.0, 1.0)),
                                              periodic = (true, true),
                                              gas_cell_count = 32,
                                              physics = (hydro = true,
                                                         tessellation = true,
                                                         gravity = false),
                                              metadata = (nx = 8, ny = 4, t_final = 0.01,
                                                          cfl = 0.25,
                                                          amplitude = 1e-3,
                                                          riemann = :hll))
        soundwave2d_state = arepo_run_scaffold(
            soundwave2d_spec; backend = :ka,
            options = ArepoRunOptions(final_time = 0.01, max_steps = 10_000,
                                      cfl = 0.25))
        @test isempty(soundwave2d_state.unsupported)
        @test soundwave2d_state.status == :calibration_pending
        @test soundwave2d_state.payload.standard_problem.status == "calibration-PENDING"
        @test soundwave2d_state.payload.standard_problem.final_metric.t ≈ 0.01 atol = 1e-12
        @test soundwave2d_state.payload.standard_problem.final_metric.rho_min > 0
        @test soundwave2d_state.payload.standard_problem.final_metric.p_min > 0

        gresho2d_spec = arepo_problem_spec(:gresho2d; dimensionality = 2,
                                           domain = ((0.0, 1.0), (0.0, 1.0)),
                                           periodic = (true, true),
                                           gas_cell_count = 256,
                                           physics = (hydro = true,
                                                      tessellation = true,
                                                      gravity = false),
                                           metadata = (nx = 16, ny = 16, t_final = 0.01,
                                                       cfl = 0.18, nbins = 16,
                                                       center = (0.5, 0.5),
                                                       riemann = :hll))
        gresho2d_state = arepo_run_scaffold(
            gresho2d_spec; backend = :ka,
            options = ArepoRunOptions(final_time = 0.01, max_steps = 10_000,
                                      cfl = 0.18))
        @test isempty(gresho2d_state.unsupported)
        @test gresho2d_state.status == :calibration_pending
        @test gresho2d_state.payload.standard_problem.status == "calibration-PENDING"
        @test gresho2d_state.payload.standard_problem.final_metric.t ≈ 0.01 atol = 1e-12
        @test gresho2d_state.payload.standard_problem.final_metric.rho_min > 0
        @test gresho2d_state.payload.standard_problem.final_metric.p_min > 0
        @test isfinite(gresho2d_state.payload.standard_problem.final_metric.vt_l2)
    end

    @testset "AREPO snapshot hydro payload adapter preserves conserved views" begin
        snapshot = read_arepo_snapshot(tiny_snapshot_payload();
                                       root = "memory", snapshot_index = 7)
        payload3 = arepo_snapshot_hydro_payload(snapshot; dimensionality = 3,
                                                gamma = 5 / 3)
        payload2 = arepo_snapshot_hydro_payload(snapshot; dimensionality = 2,
                                                gamma = 5 / 3)

        @test payload3.dimensionality == 3
        @test payload3.conserved === payload3.conserved_3d
        @test payload2.conserved === payload2.conserved_2d
        @test payload3.source_derived.volume_derived
        @test payload3.source_derived.pressure_derived
        @test payload3.source_derived.center_derived
        @test payload3.volume_consistent
        @test payload3.locator == snapshot.locator
        @test payload3.header == snapshot.header
        @test payload3.center ≈ snapshot.gas.center
        @test payload3.primitive.rho ≈ snapshot.gas.density
        @test payload3.primitive.vx ≈ snapshot.gas.velocities[:, 1]
        @test payload3.primitive.vy ≈ snapshot.gas.velocities[:, 2]
        @test payload3.primitive.vz ≈ snapshot.gas.velocities[:, 3]
        @test payload3.primitive.pressure ≈ payload3.pressure
        @test payload3.mass_from_volume ≈ snapshot.gas.masses
        @test payload3.conserved_2d isa EulerState2D
        @test payload3.conserved_3d isa EulerState3D
        @test payload3.conserved_2d.D ≈ snapshot.gas.density
        @test payload3.conserved_2d.Mx ≈ snapshot.gas.density .* snapshot.gas.velocities[:, 1]
        @test payload3.conserved_2d.My ≈ snapshot.gas.density .* snapshot.gas.velocities[:, 2]
        @test payload3.conserved_3d.Mz ≈ snapshot.gas.density .* snapshot.gas.velocities[:, 3]
        @test payload3.conserved_2d.E ≈ payload2.conserved.E
        @test payload3.conserved_3d.E ≈
              payload3.pressure ./ (5 / 3 - 1) .+
              0.5 .* payload3.density .* (payload3.primitive.vx .^ 2 .+
                                          payload3.primitive.vy .^ 2 .+
                                          payload3.primitive.vz .^ 2)
        state2d = arepo_snapshot_hydro_state_2d(snapshot; gamma = 5 / 3)
        state3d = arepo_snapshot_hydro_state_3d(snapshot; gamma = 5 / 3)
        @test state2d.D ≈ payload2.conserved_2d.D
        @test state2d.Mx ≈ payload2.conserved_2d.Mx
        @test state2d.My ≈ payload2.conserved_2d.My
        @test state2d.E ≈ payload2.conserved_2d.E
        @test state3d.D ≈ payload3.conserved_3d.D
        @test state3d.Mx ≈ payload3.conserved_3d.Mx
        @test state3d.My ≈ payload3.conserved_3d.My
        @test state3d.Mz ≈ payload3.conserved_3d.Mz
        @test state3d.E ≈ payload3.conserved_3d.E
    end

    @testset "AREPO direct gravity oracle conserves pair symmetry" begin
        x = [0.0, 1.0, 0.0]
        y = [0.0, 0.0, 2.0]
        z = [0.0, 0.0, 0.0]
        m = [2.0, 3.0, 4.0]
        ax, ay, az = arepo_direct_gravity_accel(x, y, z, m)
        @test length(ax) == 3
        @test sum(m .* ax) ≈ 0.0
        @test sum(m .* ay) ≈ 0.0
        @test sum(m .* az) ≈ 0.0

        two = arepo_direct_gravity_oracle([0.0, 1.0],
                                          [0.0, 0.0],
                                          [0.0, 0.0],
                                          [2.0, 3.0])
        @test two.ax ≈ [3.0, -2.0]
        @test two.ay ≈ [0.0, 0.0]
        @test two.az ≈ [0.0, 0.0]
        @test two.potential_energy ≈ -6.0

        softened = arepo_direct_gravity_oracle([0.0, 1.0],
                                               [0.0, 0.0],
                                               [0.0, 0.0],
                                               [2.0, 3.0];
                                               softening = 1.0)
        @test abs(softened.ax[1]) < abs(two.ax[1])
        @test abs(softened.potential_energy) < abs(two.potential_energy)
        @test_throws ErrorException arepo_direct_gravity_accel(x, y, z, m;
                                                               periodic = true)
        @test_throws ErrorException arepo_direct_gravity_potential_energy(x, y, z, m;
                                                                          softening = -1.0)
    end

    @testset "AREPO parameter/config parser normalizes runtime fields" begin
        param_text = """
        InitCondFile ics.hdf5
        ICFormat 3
        OutputDir output
        SnapshotFileBase snap
        SnapFormat 3
        NumFilesPerSnapshot 1
        TimeBegin 0.0
        TimeMax 0.2
        CourantFac 0.4
        BoxSize 1.0
        PeriodicBoundariesOn 1
        ComovingIntegrationOn 0
        OutputListOn = yes
        OutputListFilename output_list.txt
        DesNumNgb 32
        SofteningComovingType0 0.01
        SofteningTypeOfPartType0 0
        ExtraKnob keep_me
        """
        config_text = """
        TWODIMS
        DOUBLEPRECISION=1
        HAVE_HDF5=0
        OUTPUT_PRESSURE
        RIEMANN_HLL
        """
        raw = parse_arepo_param_text(param_text)
        @test raw.InitCondFile == "ics.hdf5"
        @test raw.OutputListOn == "yes"
        @test raw.ExtraKnob == "keep_me"
        flags = parse_arepo_config_text(config_text)
        @test flags isa ArepoConfigFlags
        @test :TWODIMS in flags.enabled
        @test :DOUBLEPRECISION in flags.enabled
        @test !(:HAVE_HDF5 in flags.enabled)
        @test flags.values[:HAVE_HDF5] == "0"

        params = normalize_arepo_parameters(raw, flags)
        @test params isa ArepoParameterSet
        @test params.normalized.io.output_list_on === true
        @test params.normalized.time.time_max == 0.2
        @test params.normalized.domain.periodic_boundaries_on === true
        @test params.normalized.features.twodims
        @test params.normalized.features.double_precision
        @test !params.normalized.features.have_hdf5
        @test params.normalized.gravity.softening_comoving[1] == 0.01
        @test params.normalized.gravity.softening_type_of_part[1] == 0
        @test params.normalized.extras.ExtraKnob == "keep_me"
        validation = validate_arepo_parameters(params)
        @test validation isa ArepoParameterValidation
        @test validation.valid
        @test isempty(validation.errors)
        @test any(contains("partially specified"), validation.warnings)
        runtime_features = arepo_runtime_features(params)
        @test runtime_features isa ArepoRuntimeFeatureSet
        @test runtime_features.dimensionality == 2
        @test runtime_features.parameter_io
        @test runtime_features.package_hdf5 == arepo_snapshot_hdf5_available()
        @test runtime_features.snapshot_io == runtime_features.package_hdf5
        @test !runtime_features.config_hdf5
        @test runtime_features.hydro
        @test !runtime_features.cosmology
        @test runtime_features.riemann == :hll

        @test_throws ErrorException parse_arepo_param_text("BoxSize 1.0\nBoxSize 2.0\n")
        @test_throws ErrorException normalize_arepo_parameters((; OutputListOn = "maybe"))
        missing = validate_arepo_parameters(normalize_arepo_parameters((;
            InitCondFile = "ics.hdf5",
            OutputDir = "output",
            SnapshotFileBase = "snap",
            TimeBegin = "1.0",
            TimeMax = "0.5",
            CourantFac = "-0.1",
            BoxSize = "0.0",
        )))
        @test !missing.valid
        @test "TimeMax must be >= TimeBegin" in missing.errors
        @test "CourantFac must be positive" in missing.errors
        @test "BoxSize must be positive" in missing.errors
    end

    @testset "AREPO sound-wave 2D helpers stay executable" begin
        run = pf_soundwave2d_run(nx = 8, ny = 4, t_final = 0.01,
                                 amplitude = 1e-3, riemann = :hll)
        @test run.status == "calibration-PENDING"
        @test run.numerics_ok
        @test !isempty(run.history)
        @test !isempty(run.profile_rows)
        @test isfinite(run.final_metric.rho_l2)
        @test isfinite(run.final_metric.rho_mode_phase_error)
        @test run.final_metric.mass_rel_drift <= 1e-10
        @test run.final_metric.rho_min > 0
        @test run.final_metric.p_min > 0
    end

    @testset "AREPO Gresho 2D helpers stay executable" begin
        built = pf_gresho2d_initial_state(8, 8)
        prim = conserved_to_primitive_2d(built.state; gamma = PF_GRESHO2D_DEFAULT_GAMMA)
        @test length(prim.rho) == 64
        @test all(==(1.0), prim.rho)
        @test minimum(prim.pressure) > 0
        vt = pf_gresho2d_tangential_velocity(prim, built.centers)
        @test maximum(vt) > 0

        run = pf_gresho2d_run(; nx = 8, ny = 8, t_final = 0.01,
                              nbins = 8, cfl = 0.12, riemann = :hll,
                              max_steps = 256)
        @test !isempty(run.history)
        @test !isempty(run.profile_rows)
        @test run.history[end].t ≈ 0.01 atol = 1e-12
        @test run.history[end].rho_min > 0
        @test run.history[end].p_min > 0
        @test isfinite(run.history[end].vt_l2)
        @test isfinite(run.history[end].vt_peak_ratio)
        @test run.status in ("calibration-PENDING", "run-FAIL")
    end

    @testset "3D conflict face boundary rows mark unique cavity faces" begin
        be = KernelAbstractions.CPU()
        source = Int32[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        slot = Int32[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        tetra = Int32[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        local_face = Int32[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        face_v1 = Int32[1, 1, 2, 1, 1, 1, 2, 1, 7, 7, 8, 7]
        face_v2 = Int32[2, 2, 3, 3, 2, 2, 3, 3, 8, 8, 9, 9]
        face_v3 = Int32[3, 4, 4, 4, 3, 5, 5, 5, 9, 10, 10, 10]
        active = Int32[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        conflict = CandidateConflictFaceRows3D(source, slot, tetra, local_face,
                                               face_v1, face_v2, face_v3,
                                               active, 1, 3, 1)
        boundary = candidate_boundary_face_rows_soa_3d(be, conflict;
                                                       index_type = Int32)
        @test boundary isa CandidateBoundaryFaceRows3D
        @test boundary.source_count == 1
        @test boundary.tetra_count == 3
        @test boundary.max_candidates_per_source == 1
        @test Array(boundary.boundary) == Int32[0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0]

        compact = pack_boundary_faces_soa_3d(be, boundary;
                                             max_faces_per_candidate = 8,
                                             index_type = Int32)
        @test compact isa CompactBoundaryFaces3D
        @test compact.source_count == 1
        @test compact.max_candidates_per_source == 1
        @test compact.max_faces_per_candidate == 8
        @test Array(compact.counts) == Int32[6]
        @test Array(compact.source)[1:6] == Int32[1, 1, 1, 1, 1, 1]
        @test Array(compact.slot)[1:6] == Int32[1, 1, 1, 1, 1, 1]
        @test Array(compact.tetra)[1:6] == Int32[1, 1, 1, 2, 2, 2]
        @test Array(compact.local_face)[1:6] == Int32[2, 3, 4, 2, 3, 4]
        @test Array(compact.face_v1)[1:6] == Int32[1, 2, 1, 1, 2, 1]
        @test Array(compact.face_v2)[1:6] == Int32[2, 3, 3, 2, 3, 3]
        @test Array(compact.face_v3)[1:6] == Int32[4, 4, 4, 5, 5, 5]
        @test all(Array(compact.source)[7:8] .== 0)

        stencil = CandidateStencil3D(Int32[1], Int32[9], Int32[-1],
                                     Int32[0], Int32[1], Float64[0.25], 1)
        face_candidates = compact_face_candidates_soa_3d(
            be, compact, stencil; max_faces_per_source = 8,
            index_type = Int32)
        @test face_candidates isa CompactFaceCandidates3D
        @test face_candidates.source_count == 1
        @test face_candidates.max_faces_per_source == 8
        @test Array(face_candidates.counts) == Int32[6]
        @test Array(face_candidates.c1)[1:6] == Int32[1, 1, 1, 1, 1, 1]
        @test Array(face_candidates.c2)[1:6] == Int32[9, 9, 9, 9, 9, 9]
        @test Array(face_candidates.image_sx)[1:6] == Int32[-1, -1, -1, -1, -1, -1]
        @test Array(face_candidates.image_sy)[1:6] == Int32[0, 0, 0, 0, 0, 0]
        @test Array(face_candidates.image_sz)[1:6] == Int32[1, 1, 1, 1, 1, 1]
        @test Array(face_candidates.face_v1)[1:6] == Array(compact.face_v1)[1:6]
        @test all(Array(face_candidates.c1)[7:8] .== 0)

        csr = compact_face_candidate_csr_soa_3d(be, face_candidates;
                                                index_type = Int32)
        @test csr isa CompactFaceCandidateCSR3D
        @test csr.source_count == 1
        @test csr.max_faces_per_source == 8
        @test Array(csr.counts) == Int32[6]
        @test Array(csr.offsets) == Int32[1, 9]

        source_csr = source_owned_face_csr_soa_3d(be, face_candidates;
                                                  index_type = Int32)
        @test source_csr isa SourceOwnedFaceCSR3D
        @test source_csr.source_count == 1
        @test source_csr.max_faces_per_source == 8
        @test Array(source_csr.counts) == Int32[6]
        @test Array(source_csr.offsets) == Int32[1, 9]
        @test Array(source_csr.faces) == Int32[1, 2, 3, 4, 5, 6, 7, 8]
        @test Array(source_csr.signs) == Int32[-1, -1, -1, -1, -1, -1, 0, 0]

        one_sided_pairs = reciprocal_face_candidate_pairs_soa_3d(
            be, face_candidates; index_type = Int32)
        @test one_sided_pairs isa ReciprocalFaceCandidatePairs3D
        @test Array(one_sided_pairs.active) == Int32[1, 1, 1, 1, 1, 1, 0, 0]
        @test Array(one_sided_pairs.pair_row) == zeros(Int32, 8)
        @test Array(one_sided_pairs.canonical_row) == Int32[1, 2, 3, 4, 5, 6, 0, 0]
        @test Array(one_sided_pairs.owner) == Int32[1, 1, 1, 1, 1, 1, 0, 0]

        paired_candidates = CompactFaceCandidates3D(
            Int32[1, 1],
            Int32[1, 0, 2, 0],
            Int32[2, 0, 1, 0],
            Int32[0, 0, 0, 0],
            Int32[0, 0, 0, 0],
            Int32[0, 0, 0, 0],
            Int32[1, 0, 1, 0],
            Int32[1, 0, 1, 0],
            Int32[1, 0, 1, 0],
            Int32[4, 0, 4, 0],
            Int32[5, 0, 5, 0],
            Int32[6, 0, 6, 0],
            2, 2)
        reciprocal = reciprocal_face_candidate_pairs_soa_3d(
            be, paired_candidates; index_type = Int32)
        @test reciprocal isa ReciprocalFaceCandidatePairs3D
        @test Array(reciprocal.active) == Int32[1, 0, 1, 0]
        @test Array(reciprocal.pair_row) == Int32[3, 0, 1, 0]
        @test Array(reciprocal.canonical_row) == Int32[1, 0, 1, 0]
        @test Array(reciprocal.owner) == Int32[1, 0, 0, 0]

        canonical_csr = canonical_face_candidate_csr_soa_3d(
            be, paired_candidates, reciprocal; index_type = Int32)
        @test canonical_csr isa SourceOwnedFaceCSR3D
        @test Array(canonical_csr.offsets) == Int32[1, 3, 5]
        @test Array(canonical_csr.faces) == Int32[1, 2, 1, 4]
        @test Array(canonical_csr.signs) == Int32[-1, 0, 1, 0]

        canonical_mesh = canonical_face_candidate_mesh_arrays_3d(
            be, paired_candidates;
            pairs = reciprocal,
            volume = [0.5, 0.5],
            default_face_area = 0.0,
            default_normal = (1.0, 0.0, 0.0),
            T = Float64,
            index_type = Int32)
        @test canonical_mesh isa ArepoMeshArrays3D
        @test Array(canonical_mesh.c1) == Int32[1, 1, 2, 2]
        @test Array(canonical_mesh.c2) == Int32[2, 0, 0, 0]
        @test Array(canonical_mesh.face_area) == [0.0, 0.0, 0.0, 0.0]
        @test Array(canonical_mesh.cell_face_offsets) == Int32[1, 3, 5]
        @test Array(canonical_mesh.cell_faces) == Int32[1, 2, 1, 4]
        @test Array(canonical_mesh.cell_face_signs) == Int32[-1, 0, 1, 0]
        canonical_state = euler_state_3d(canonical_mesh; rho = 1.0,
                                         pressure = 1.0, gamma = 1.4)
        finite_volume_step_3d!(canonical_state, canonical_mesh;
                               dt = 1e-3, gamma = 1.4, riemann = :hll)
        canonical_prim = conserved_to_primitive_3d(canonical_state;
                                                   gamma = 1.4)
        @test Array(canonical_prim.rho) ≈ ones(2)
        @test Array(canonical_prim.pressure) ≈ ones(2)

        compact_canonical = compact_canonical_faces_soa_3d(
            be, paired_candidates, reciprocal; index_type = Int32)
        @test compact_canonical isa CompactCanonicalFaces3D
        @test Array(compact_canonical.source_row) == Int32[1]
        @test Array(compact_canonical.c1) == Int32[1]
        @test Array(compact_canonical.c2) == Int32[2]
        @test Array(compact_canonical.face_v1) == Int32[4]
        @test Array(compact_canonical.face_v2) == Int32[5]
        @test Array(compact_canonical.face_v3) == Int32[6]

        compact_mesh = compact_canonical_mesh_arrays_3d(
            be, compact_canonical;
            volume = [0.5, 0.5],
            face_area = [0.0],
            normal_x = [1.0],
            normal_y = [0.0],
            normal_z = [0.0],
            face_vx = [0.0],
            face_vy = [0.0],
            face_vz = [0.0],
            T = Float64,
            index_type = Int32)
        @test compact_mesh isa ArepoMeshArrays3D
        @test Array(compact_mesh.c1) == Int32[1]
        @test Array(compact_mesh.c2) == Int32[2]
        @test Array(compact_mesh.face_area) == [0.0]
        @test Array(compact_mesh.normal_x) == [1.0]
        @test Array(compact_mesh.normal_y) == [0.0]
        @test Array(compact_mesh.normal_z) == [0.0]
        @test Array(compact_mesh.face_vx) == [0.0]
        @test Array(compact_mesh.cell_face_offsets) == Int32[1, 2, 3]
        @test Array(compact_mesh.cell_faces) == Int32[1, 1]
        @test Array(compact_mesh.cell_face_signs) == Int32[-1, 1]
        @test_throws ErrorException compact_canonical_mesh_arrays_3d(
            be, compact_canonical; volume = [0.5, 0.5],
            face_area = [1.0, 2.0], T = Float64, index_type = Int32)
        compact_state = euler_state_3d(compact_mesh; rho = 1.0,
                                       pressure = 1.0, gamma = 1.4)
        finite_volume_step_3d!(compact_state, compact_mesh;
                               dt = 1e-3, gamma = 1.4, riemann = :hll)
        compact_prim = conserved_to_primitive_3d(compact_state; gamma = 1.4)
        @test Array(compact_prim.rho) ≈ ones(2)
        @test Array(compact_prim.pressure) ≈ ones(2)

        scan_candidates = CompactFaceCandidates3D(
            Int32[2, 2, 2],
            Int32[11, 12, 21, 22, 31, 32],
            Int32[12, 13, 22, 23, 32, 33],
            Int32[0, 0, 0, 0, 0, 0],
            Int32[0, 0, 0, 0, 0, 0],
            Int32[0, 0, 0, 0, 0, 0],
            Int32[1, 2, 1, 2, 1, 2],
            Int32[1, 1, 2, 2, 3, 3],
            Int32[1, 2, 3, 4, 1, 2],
            Int32[4, 5, 6, 7, 8, 9],
            Int32[5, 6, 7, 8, 9, 10],
            Int32[6, 7, 8, 9, 10, 11],
            2, 3)
        scan_pairs = ReciprocalFaceCandidatePairs3D(
            Int32[1, 0, 0, 0, 1, 0],
            Int32[1, 0, 0, 0, 5, 0],
            Int32[1, 0, 0, 0, 5, 0],
            Int32[1, 0, 0, 0, 1, 0],
            2, 3)
        scan_reference = compact_canonical_faces_soa_3d(
            be, scan_candidates, scan_pairs; index_type = Int32)
        scan_csr = compact_canonical_face_csr_soa_3d(
            be, scan_candidates, scan_pairs; index_type = Int32)
        @test scan_csr isa CompactCanonicalFaceCSR3D
        @test Array(scan_csr.compact.source_row) == Array(scan_reference.source_row)
        @test Array(scan_csr.compact.c1) == Array(scan_reference.c1)
        @test Array(scan_csr.compact.c2) == Array(scan_reference.c2)
        @test Array(scan_csr.counts) == Int32[1, 0, 1]
        @test Array(scan_csr.offsets) == Int32[1, 2, 2, 3]

        n8_candidates = CompactFaceCandidates3D(
            Int32[1, 1],
            Int32[1, 0, 0, 0, 2, 0, 0, 0],
            Int32[2, 0, 0, 0, 1, 0, 0, 0],
            Int32[0, 0, 0, 0, 0, 0, 0, 0],
            Int32[0, 0, 0, 0, 0, 0, 0, 0],
            Int32[0, 0, 0, 0, 0, 0, 0, 0],
            Int32[1, 0, 0, 0, 1, 0, 0, 0],
            Int32[1, 0, 0, 0, 1, 0, 0, 0],
            Int32[1, 0, 0, 0, 1, 0, 0, 0],
            Int32[4, 0, 0, 0, 4, 0, 0, 0],
            Int32[5, 0, 0, 0, 5, 0, 0, 0],
            Int32[6, 0, 0, 0, 6, 0, 0, 0],
            4, 2)

        n8_pairs = reciprocal_face_candidate_pairs_soa_3d(
            be, n8_candidates; index_type = Int32)
        @test n8_pairs isa ReciprocalFaceCandidatePairs3D
        @test Array(n8_pairs.active) == Int32[1, 0, 0, 0, 1, 0, 0, 0]
        @test Array(n8_pairs.pair_row) == Int32[5, 0, 0, 0, 1, 0, 0, 0]
        @test Array(n8_pairs.canonical_row) == Int32[1, 0, 0, 0, 1, 0, 0, 0]
        @test Array(n8_pairs.owner) == Int32[1, 0, 0, 0, 0, 0, 0, 0]

        n8_compact = compact_canonical_faces_soa_3d(
            be, n8_candidates, n8_pairs; index_type = Int32)
        @test n8_compact isa CompactCanonicalFaces3D
        @test Array(n8_compact.source_row) == Int32[1]
        @test Array(n8_compact.c1) == Int32[1]
        @test Array(n8_compact.c2) == Int32[2]

        n4_mesh = compact_canonical_mesh_arrays_3d(
            be, paired_candidates;
            volume = [0.5, 0.5],
            default_face_area = 1.0,
            default_normal = (1.0, 0.0, 0.0),
            default_face_velocity = (0.0, 0.0, 0.0),
            T = Float64,
            index_type = Int32)
        @test Array(n4_mesh.face_area) == [1.0]
        @test Array(n4_mesh.normal_x) == [1.0]
        @test Array(n4_mesh.normal_y) == [0.0]
        @test Array(n4_mesh.normal_z) == [0.0]
        @test Array(n4_mesh.cell_face_offsets) == Int32[1, 2, 3]
        @test Array(n4_mesh.cell_faces) == Int32[1, 1]
        @test Array(n4_mesh.cell_face_signs) == Int32[-1, 1]

        n8_mesh = compact_canonical_mesh_arrays_3d(
            be, n8_candidates;
            pairs = n8_pairs,
            volume = [0.25, 0.75],
            face_area = [2.5],
            normal_x = [0.0],
            normal_y = [1.0],
            normal_z = [0.0],
            face_vx = [0.1],
            face_vy = [0.2],
            face_vz = [0.3],
            T = Float64,
            index_type = Int32)
        @test Array(n8_mesh.face_area) == [2.5]
        @test Array(n8_mesh.normal_x) == [0.0]
        @test Array(n8_mesh.normal_y) == [1.0]
        @test Array(n8_mesh.normal_z) == [0.0]
        @test Array(n8_mesh.face_vx) == [0.1]
        @test Array(n8_mesh.face_vy) == [0.2]
        @test Array(n8_mesh.face_vz) == [0.3]
        @test all(Array(n8_mesh.face_area) .> 0)
        @test Array(n8_mesh.cell_face_offsets) == Int32[1, 2, 3]
        @test Array(n8_mesh.cell_faces) == Int32[1, 1]
        @test Array(n8_mesh.cell_face_signs) == Int32[-1, 1]

        mesh = compact_face_candidate_mesh_arrays_3d(
            be, face_candidates;
            volume = [1.0],
            default_face_area = 0.0,
            default_normal = (1.0, 0.0, 0.0),
            T = Float64,
            index_type = Int32)
        @test mesh isa ArepoMeshArrays3D
        @test Array(mesh.c1)[1:6] == Int32[1, 1, 1, 1, 1, 1]
        @test Array(mesh.c2)[1:6] == Int32[9, 9, 9, 9, 9, 9]
        @test all(Array(mesh.face_area)[1:6] .== 0.0)
        @test all(Array(mesh.face_area)[7:8] .== 0.0)
        @test Array(mesh.cell_face_offsets) == Int32[1, 9]
        @test Array(mesh.cell_faces) == Int32[1, 2, 3, 4, 5, 6, 7, 8]
        @test Array(mesh.cell_face_signs) == Int32[-1, -1, -1, -1, -1, -1, 0, 0]
        @test Array(mesh.volume) == [1.0]
    end

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

    @testset "2D periodic Voronoi arrays preserve uniform flow on rectangular box" begin
        nx = 4
        ny = 8
        pts = Matrix{Float64}(undef, nx * ny, 2)
        q = 1
        for j in 1:ny, i in 1:nx
            pts[q, 1] = (i - 0.5) / nx
            pts[q, 2] = (j - 0.5) * 2 / ny
            q += 1
        end
        built = periodic_power_mesh_arrays_2d(pts; domain = ((0.0, 1.0), (0.0, 2.0)),
                                             T = Float64)
        local_built = periodic_power_mesh_arrays_2d(pts;
                                                    domain = ((0.0, 1.0), (0.0, 2.0)),
                                                    T = Float64,
                                                    bins_per_axis = (nx, ny),
                                                    search_radius = 1)
        geom = built.geom
        @test length(geom.volume) == nx * ny
        @test length(geom.c1) == 2 * nx * ny
        @test count(==(0), geom.c2) == 0
        @test sum(geom.volume) ≈ 2.0
        vmin, vmax = extrema(geom.volume)
        @test vmin ≈ 2 / (nx * ny)
        @test vmax ≈ 2 / (nx * ny)
        @test all(diff(Int.(geom.cell_face_offsets)) .== 4)
        @test local_built.volume ≈ built.volume
        @test sort(local_built.geom.face_area) ≈ sort(built.geom.face_area)
        @test length(local_built.geom.c1) == length(built.geom.c1)

        rho = fill(1.0, nx * ny)
        vx = fill(0.1, nx * ny)
        vy = fill(-0.05, nx * ny)
        pressure = fill(1.0, nx * ny)
        state = EulerState2D(copy(rho), rho .* vx, rho .* vy,
                             pressure ./ (1.4 - 1) .+
                             0.5 .* rho .* (vx .* vx .+ vy .* vy))
        total0 = total_conserved_2d(state, geom)
        finite_volume_step_2d!(state, geom; dt = 0.001, gamma = 1.4,
                               riemann = :hll)
        total1 = total_conserved_2d(state, geom)
        @test total1.mass ≈ total0.mass
        @test total1.mx ≈ total0.mx
        @test total1.my ≈ total0.my
        @test total1.energy ≈ total0.energy
        @test conserved_to_primitive_2d(state; gamma = 1.4).rho ≈ rho

        recon = EulerState2D(copy(rho), rho .* vx, rho .* vy,
                             pressure ./ (1.4 - 1) .+
                             0.5 .* rho .* (vx .* vx .+ vy .* vy))
        prim = primitive_work_2d(recon)
        conserved_to_primitive_2d!(prim, recon; gamma = 1.4)
        gradients = hydro_gradient_work_2d(prim.rho)
        calculate_gradients_from_mesh_2d!(gradients, geom, prim,
                                          built.center, built.face_center;
                                          box_size = 1.0, box_size_y = 2.0,
                                          gamma = 1.4)
        total0 = total_conserved_2d(recon, geom)
        finite_volume_reconstructed_step_2d!(recon, geom, gradients, prim,
                                             built.center, built.face_center;
                                             dt = 0.001, gamma = 1.4,
                                             riemann = :hll,
                                             box_size = 1.0,
                                             box_size_y = 2.0)
        total1 = total_conserved_2d(recon, geom)
        @test total1.mass ≈ total0.mass
        @test total1.mx ≈ total0.mx
        @test total1.my ≈ total0.my
        @test total1.energy ≈ total0.energy
        @test conserved_to_primitive_2d(recon; gamma = 1.4).rho ≈ rho
    end

    @testset "2D reconstructed hydro backend work preserves linear data and uniform flow" begin
        pts = Matrix{Float64}(undef, 9, 2)
        q = 1
        for j in 1:3, i in 1:3
            pts[q, 1] = (i - 0.5) / 3
            pts[q, 2] = (j - 0.5) / 3
            q += 1
        end
        mesh = power_diagram(PowerSites2D(pts))
        center = cell_centroids(mesh)
        face_center = mesh.faces.center
        geom = arepo_mesh_arrays(mesh; T = Float64)

        rho = @. 1.0 + 0.2 * center[:, 1] - 0.1 * center[:, 2]
        vx = @. 0.05 + 0.03 * center[:, 1] + 0.02 * center[:, 2]
        vy = @. -0.04 + 0.01 * center[:, 1] - 0.025 * center[:, 2]
        pressure = @. 1.0 + 0.15 * center[:, 1] + 0.05 * center[:, 2]
        state = euler_state_2d(mesh; rho, vx, vy, pressure, gamma = 1.4)
        be = KernelAbstractions.CPU()
        bgeom = to_backend(be, geom; T = Float32)
        bstate = to_backend(be, state; T = Float32)
        prim = primitive_work_2d(bstate)
        conserved_to_primitive_2d!(prim, bstate; gamma = 1.4)
        parr = primitive_to_arrays_2d(prim)
        @test parr.rho ≈ Float32.(rho)
        @test parr.vx ≈ Float32.(vx)
        @test parr.vy ≈ Float32.(vy)
        @test parr.pressure ≈ Float32.(pressure)

        bcx = PowerFoam._backend_copy(be, collect(center[:, 1]), Float32)
        bcy = PowerFoam._backend_copy(be, collect(center[:, 2]), Float32)
        bfcx = PowerFoam._backend_copy(be, collect(face_center[:, 1]), Float32)
        bfcy = PowerFoam._backend_copy(be, collect(face_center[:, 2]), Float32)
        gradients = hydro_gradient_work_2d(prim.rho)
        calculate_gradients_from_mesh_2d!(gradients, bgeom, prim, bcx, bcy, bfcx, bfcy;
                                          box_size = 0.0, gamma = 1.4)
        g = hydro_gradients_to_arrays(gradients)
        mid = 5
        @test g.drho[mid, :] ≈ Float32[0.2, -0.1] atol=1f-5
        @test g.dvel[mid, 1, :] ≈ Float32[0.03, 0.02] atol=1f-5
        @test g.dvel[mid, 2, :] ≈ Float32[0.01, -0.025] atol=1f-5
        @test g.dpress[mid, :] ≈ Float32[0.15, 0.05] atol=1f-5

        face_states = face_prediction_work_2d(bgeom)
        zdt = PowerFoam._backend_copy(be, zeros(Float32, length(bstate.D)), Float32)
        predict_face_states_2d!(face_states, bgeom, gradients, prim, bcx, bcy, bfcx, bfcy;
                                dt_extrapolation = zdt, box_size = 0.0, gamma = 1.4)
        fs = face_states_to_arrays(face_states)
        f = first(findall((mesh.faces.c1 .== mid) .& (mesh.faces.c2 .> 0)))
        expected_rho = rho[mid] + 0.2 * (face_center[f, 1] - center[mid, 1]) -
                       0.1 * (face_center[f, 2] - center[mid, 2])
        expected_vx = vx[mid] + 0.03 * (face_center[f, 1] - center[mid, 1]) +
                      0.02 * (face_center[f, 2] - center[mid, 2])
        expected_vy = vy[mid] + 0.01 * (face_center[f, 1] - center[mid, 1]) -
                      0.025 * (face_center[f, 2] - center[mid, 2])
        expected_p = pressure[mid] + 0.15 * (face_center[f, 1] - center[mid, 1]) +
                     0.05 * (face_center[f, 2] - center[mid, 2])
        @test fs.left.rho[f] ≈ Float32(expected_rho) atol=1f-5
        @test fs.left.vx[f] ≈ Float32(expected_vx) atol=1f-5
        @test fs.left.vy[f] ≈ Float32(expected_vy) atol=1f-5
        @test fs.left.pressure[f] ≈ Float32(expected_p) atol=1f-5

        uniform = euler_state_2d(mesh; rho = 1.0, vx = 0.1, vy = -0.05,
                                 pressure = 1.0, gamma = 1.4)
        buniform = to_backend(be, uniform; T = Float32)
        uprim = primitive_work_2d(buniform)
        conserved_to_primitive_2d!(uprim, buniform; gamma = 1.4)
        ugrad = hydro_gradient_work_2d(uprim.rho)
        calculate_gradients_from_mesh_2d!(ugrad, bgeom, uprim, bcx, bcy, bfcx, bfcy;
                                          box_size = 0.0, gamma = 1.4)
        total0 = total_conserved_2d(buniform, bgeom)
        finite_volume_reconstructed_step_2d!(buniform, bgeom, ugrad, uprim,
                                             bcx, bcy, bfcx, bfcy;
                                             dt = 0.001f0, gamma = 1.4,
                                             riemann = :hll, box_size = 0.0)
        total1 = total_conserved_2d(buniform, bgeom)
        @test total1.mass ≈ total0.mass
        @test total1.mx ≈ total0.mx
        @test total1.my ≈ total0.my
        @test total1.energy ≈ total0.energy
        @test all(conserved_to_primitive_2d(buniform; gamma = 1.4).pressure .> 0)
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

        recon_static = euler_state_2d(mesh; rho = [1.0, 2.0], vx = 0.0, vy = 0.0,
                                      pressure = 1.0, gamma = 1.4)
        recon_moving = euler_state_2d(mesh; rho = [1.0, 2.0], vx = 0.0, vy = 0.0,
                                      pressure = 1.0, gamma = 1.4)
        prim = primitive_work_2d(recon_static)
        conserved_to_primitive_2d!(prim, recon_static; gamma = 1.4)
        gradients = hydro_gradient_work_2d(prim.rho)
        calculate_gradients_from_mesh_2d!(gradients, geom, prim,
                                          cell_centroids(mesh), mesh.faces.center;
                                          box_size = 0.0, gamma = 1.4)
        finite_volume_reconstructed_step_2d!(recon_static, geom, gradients, prim,
                                             cell_centroids(mesh), mesh.faces.center;
                                             dt = 0.01, gamma = 1.4,
                                             riemann = :hll, box_size = 0.0)
        moved_recon0 = moving_mesh_reconstructed_step_2d!(recon_moving, mesh;
                                                          dt = 0.01, gamma = 1.4,
                                                          riemann = :hll,
                                                          mesh_velocity = zero_v)
        @test moved_recon0.mesh.generators ≈ mesh.generators
        @test recon_moving.D ≈ recon_static.D
        @test recon_moving.Mx ≈ recon_static.Mx
        @test recon_moving.My ≈ recon_static.My
        @test recon_moving.E ≈ recon_static.E

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
        skew_face = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                     (2.0, 2.0, 0.0), (0.0, 1.0, 0.0)]
        @test collect(PowerFoam._polygon_area_centroid3(skew_face)) ≈
              [10 / 9, 7 / 9, 0.0]

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
        @test size(mesh.face_image_shift) == (length(mesh.geom.c1), 3)
        @test count(!=(0), vec(mesh.face_image_shift)) > 0

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
        @test size(local_mesh.face_image_shift) == (length(local_mesh.geom.c1), 3)
        @test count(!=(0), vec(local_mesh.face_image_shift)) > 0
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

        inactive_geom = ArepoMeshArrays3D(geom.c1, geom.c2, geom.cell_face_offsets,
                                          geom.cell_faces, geom.cell_face_signs,
                                          geom.volume,
                                          [i == 1 ? 0.0 : geom.face_area[i] for i in 1:nf],
                                          geom.normal_x, geom.normal_y, geom.normal_z,
                                          geom.face_vx, geom.face_vy, geom.face_vz)
        inactive_states = face_prediction_work_3d(inactive_geom)
        predict_face_states_3d!(inactive_states, inactive_geom, gradients, rho,
                                zero_cell, zero_cell, zero_cell, one_cell,
                                center, face_center; box_size = 1.0, gamma = 1.4)
        inactive = face_states_to_arrays(inactive_states)
        @test inactive.left.rho[1] == 0.0
        @test inactive.right.pressure[1] == 0.0
        @test inactive.left.rho[2:end] ≈ expected_left[2:end]
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
        active_table = active_face_table_3d(geom, trues(nc))
        active_stride = active_table.active_stride
        active_counts = active_table.active_counts
        active_faces = active_table.active_faces
        active_signs = active_table.active_signs
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
        active_table = active_face_table_3d(geom, trues(nc))
        active_stride = active_table.active_stride
        active_counts = active_table.active_counts
        active_faces = active_table.active_faces
        active_signs = active_table.active_signs
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

    @testset "3D production tessellator API exposes stable reference schema" begin
        pts = Matrix{Float64}(undef, 8, 3)
        q = 1
        for k in 1:2, j in 1:2, i in 1:2
            pts[q, 1] = (i - 0.5) / 2
            pts[q, 2] = (j - 0.5) / 2
            pts[q, 3] = (k - 0.5) / 2
            q += 1
        end
        ref = build_arepo_tessellation_3d(pts; bins_per_axis = 2,
                                          threaded = false)
        @test ref isa TessellationReference3D
        @test ref.algorithm == :local_periodic_halfspace
        @test ref.backend_residency == :host_reference
        @test ref.metadata.cells == 8
        @test ref.metadata.faces == length(ref.geom.c1)
        @test size(ref.face_image_shift) == (length(ref.geom.c1), 3)
        @test length(ref.canonical_face_keys) == length(ref.geom.c1)
        @test sort(ref.canonical_face_order) == collect(eachindex(ref.geom.c1))
        @test sum(ref.geom.volume) ≈ 1.0

        keys = canonical_face_keys_3d(ref.geom;
                                      face_image_shift = ref.face_image_shift)
        @test keys == ref.canonical_face_keys
        order = canonical_face_order_3d(ref.geom;
                                        face_image_shift = ref.face_image_shift)
        @test order == ref.canonical_face_order
    end

    @testset "3D production tessellator semantic scaffold" begin
        @test TessellationPredicateAdaptive3D != TessellationPredicateExactCPU3D
        @test TessellationPredicateFloat64Only3D != TessellationPredicateCPUFallback3D

        point = TessellationPointIdentity3D(12, 3; owner_task = 7,
                                            owner_index = 9, timebin = 4,
                                            image_flags = 0x11,
                                            image_shift = (1, -1, 0))
        @test point.original_index == 12
        @test point.active_index == 3
        @test point.owner_task == 7
        @test point.owner_index == 9
        @test point.timebin == 4
        @test point.image_flags == 0x11
        @test point.image_shift == (1, -1, 0)

        face = TessellationFaceProvenance3D(5, 2, 7; owner_task = 8,
                                            owner_index = 4,
                                            image_shift = (0, 0, -1),
                                            orientation = -1,
                                            duplicate = true)
        @test face.face_index == 5
        @test face.c1 == 2
        @test face.c2 == 7
        @test face.owner_task == 8
        @test face.owner_index == 4
        @test face.image_shift == (0, 0, -1)
        @test face.orientation == -1
        @test face.duplicate

        counters = TessellationFallbackCounters3D()
        record_in_sphere_test!(counters)
        record_in_sphere_test!(counters; exact = true)
        record_convex_edge_test!(counters; exact = true)
        record_in_tetra_test!(counters)
        record_orient3d_test!(counters; exact = true)
        record_exact_cpu_fallback!(counters)
        record_gpu_fallback!(counters)
        record_topology_retry!(counters)
        record_degenerate_face!(counters)
        record_skipped_infinite_tetra!(counters)

        @test counters.count_in_sphere_tests == 2
        @test counters.count_in_sphere_tests_exact == 1
        @test counters.count_convex_edge_test == 1
        @test counters.count_convex_edge_test_exact == 1
        @test counters.count_in_tetra == 1
        @test counters.count_in_tetra_exact == 0
        @test counters.orient3d_tests == 1
        @test counters.orient3d_tests_exact == 1
        @test counters.exact_cpu_fallbacks == 1
        @test counters.gpu_fallbacks == 1
        @test counters.topology_retries == 1
        @test counters.degenerate_faces == 1
        @test counters.skipped_infinite_tetra == 1

        PowerFoam.reset!(counters)
        @test all(getfield(counters, name) == 0
                  for name in fieldnames(typeof(counters)))
    end

    @testset "3D AREPO Delaunay reference tessellator builds hydro mesh" begin
        pts = [0.20 0.20 0.20;
               0.70 0.20 0.25;
               0.25 0.75 0.20;
               0.25 0.25 0.75;
               0.72 0.72 0.72;
               0.52 0.58 0.32;
               0.33 0.54 0.66;
               0.82 0.36 0.61]
        ref = build_arepo_tessellation_3d(pts;
                                          algorithm = :arepo_delaunay_reference,
                                          return_delaunay = true,
                                          min_face_surface_fraction = 1e-10)
        @test ref.algorithm == :arepo_delaunay_reference
        @test ref.delaunay isa DelaunayTetrahedra3D
        @test ref.metadata.has_delaunay
        @test length(ref.geom.c1) > 0
        @test sum(ref.geom.volume) ≈ 1.0 atol = 1e-10
        @test all(>(0), ref.geom.volume)
        @test size(ref.face_center) == (length(ref.geom.c1), 3)
        @test size(ref.face_image_shift) == (length(ref.geom.c1), 3)
        @test sort(ref.canonical_face_order) == collect(eachindex(ref.geom.c1))

        soa = tessellation_soa_3d(ref; T = Float64, index_type = Int32)
        @test soa isa TessellationSoA3D
        @test soa.delaunay isa DelaunaySoA3D
        @test soa.delaunay.point_x ≈ ref.delaunay.points[:, 1]
        @test soa.delaunay.point_y ≈ ref.delaunay.points[:, 2]
        @test soa.delaunay.point_z ≈ ref.delaunay.points[:, 3]
        @test Int.(soa.delaunay.original_index) == ref.delaunay.original_index
        @test hcat(Int.(soa.delaunay.image_sx),
                   Int.(soa.delaunay.image_sy),
                   Int.(soa.delaunay.image_sz)) == ref.delaunay.image_shift
        @test length(soa.delaunay.tet_p1) == length(ref.delaunay.tetras)
        @test soa.delaunay.circum_x ≈ ref.delaunay.circumcenters[:, 1]
        @test all(==(1), soa.delaunay.circum_valid)
        @test soa.center_x ≈ ref.center[:, 1]
        @test soa.face_center_x ≈ ref.face_center[:, 1]
        @test Int.(soa.face_image_sx) == ref.face_image_shift[:, 1]

        images = periodic_point_images_soa_3d(KernelAbstractions.CPU(), pts;
                                              T = Float64, index_type = Int32)
        @test images isa PeriodicPointImages3D
        @test Array(images.point_x) ≈ ref.delaunay.points[:, 1]
        @test Array(images.point_y) ≈ ref.delaunay.points[:, 2]
        @test Array(images.point_z) ≈ ref.delaunay.points[:, 3]
        @test Int.(Array(images.original_index)) == ref.delaunay.original_index
        @test hcat(Int.(Array(images.image_sx)),
                   Int.(Array(images.image_sy)),
                   Int.(Array(images.image_sz))) == ref.delaunay.image_shift

        candidates = dense_candidate_pairs_soa_3d(KernelAbstractions.CPU(), pts;
                                                  T = Float64,
                                                  index_type = Int32,
                                                  bins_per_axis = 2,
                                                  search_radius = 1)
        @test candidates isa DenseCandidatePairs3D
        nsrc = size(pts, 1)
        nimg = length(images.point_x)
        @test length(candidates.source) == nsrc * nimg
        expected_source = Vector{Int32}(undef, nsrc * nimg)
        expected_candidate = Vector{Int32}(undef, nsrc * nimg)
        expected_active = Vector{Int32}(undef, nsrc * nimg)
        expected_dist2 = Vector{Float64}(undef, nsrc * nimg)
        imgx = Array(images.point_x)
        imgy = Array(images.point_y)
        imgz = Array(images.point_z)
        imgorig = Int.(Array(images.original_index))
        imgsx = Int.(Array(images.image_sx))
        imgsy = Int.(Array(images.image_sy))
        imgsz = Int.(Array(images.image_sz))
        for img in 1:nimg, i in 1:nsrc
            row = (img - 1) * nsrc + i
            bix = floor(Int, pts[i, 1] * 2) + 1
            biy = floor(Int, pts[i, 2] * 2) + 1
            biz = floor(Int, pts[i, 3] * 2) + 1
            bjx = floor(Int, imgx[img] * 2) + 1
            bjy = floor(Int, imgy[img] * 2) + 1
            bjz = floor(Int, imgz[img] * 2) + 1
            self_image = i == imgorig[img] && imgsx[img] == 0 &&
                         imgsy[img] == 0 && imgsz[img] == 0
            expected_source[row] = i
            expected_candidate[row] = imgorig[img]
            expected_active[row] = !self_image &&
                                   abs(bjx - bix) <= 1 &&
                                   abs(bjy - biy) <= 1 &&
                                   abs(bjz - biz) <= 1 ? 1 : 0
            dx = imgx[img] - pts[i, 1]
            dy = imgy[img] - pts[i, 2]
            dz = imgz[img] - pts[i, 3]
            expected_dist2[row] = dx * dx + dy * dy + dz * dz
        end
        @test Array(candidates.source) == expected_source
        @test Array(candidates.candidate) == expected_candidate
        @test Array(candidates.active) == expected_active
        @test Array(candidates.distance2) ≈ expected_dist2
        @test sum(Int.(Array(candidates.active))) > 0
        expected_counts = zeros(Int32, nsrc)
        for row in eachindex(expected_active)
            expected_active[row] == 0 && continue
            i = Int(expected_source[row])
            expected_counts[i] += 1
        end
        max_candidates = maximum(Int.(expected_counts))
        stencil = pack_candidate_stencil_soa_3d(KernelAbstractions.CPU(),
                                                candidates, nsrc;
                                                max_candidates_per_source = max_candidates)
        @test stencil isa CandidateStencil3D
        @test Array(stencil.counts) == expected_counts
        expected_stencil_candidate = zeros(Int32, nsrc * max_candidates)
        expected_stencil_sx = zeros(Int32, nsrc * max_candidates)
        expected_stencil_sy = zeros(Int32, nsrc * max_candidates)
        expected_stencil_sz = zeros(Int32, nsrc * max_candidates)
        expected_stencil_distance2 = zeros(Float64, nsrc * max_candidates)
        cursor = zeros(Int, nsrc)
        for row in eachindex(expected_active)
            expected_active[row] == 0 && continue
            i = Int(expected_source[row])
            cursor[i] += 1
            out = (i - 1) * max_candidates + cursor[i]
            expected_stencil_candidate[out] = expected_candidate[row]
            expected_stencil_sx[out] = Int32(imgsx[div(row - 1, nsrc) + 1])
            expected_stencil_sy[out] = Int32(imgsy[div(row - 1, nsrc) + 1])
            expected_stencil_sz[out] = Int32(imgsz[div(row - 1, nsrc) + 1])
            expected_stencil_distance2[out] = expected_dist2[row]
        end
        @test Array(stencil.candidate) == expected_stencil_candidate
        @test Array(stencil.image_sx) == expected_stencil_sx
        @test Array(stencil.image_sy) == expected_stencil_sy
        @test Array(stencil.image_sz) == expected_stencil_sz
        @test Array(stencil.distance2) ≈ expected_stencil_distance2

        be_soa = to_backend(KernelAbstractions.CPU(), soa; T = Float64,
                            index_type = Int32)
        @test Array(be_soa.delaunay.tet_p1) == soa.delaunay.tet_p1
        @test Array(be_soa.delaunay.circum_z) ≈ soa.delaunay.circum_z
        @test Array(be_soa.geom.c1) == Array(ref.geom.c1)
        @test Array(be_soa.center_y) ≈ ref.center[:, 2]

        predicates = candidate_tetra_predicates_soa_3d(
            KernelAbstractions.CPU(), pts, stencil, be_soa.delaunay;
            T = Float64, index_type = Int32, tol = 1e-10)
        @test predicates isa CandidateTetraPredicates3D
        ntetra = length(ref.delaunay.tetras)
        @test predicates.source_count == nsrc
        @test predicates.tetra_count == ntetra
        @test predicates.max_candidates_per_source == max_candidates
        expected_valid = zeros(Int32, nsrc * max_candidates * ntetra)
        expected_inside = zeros(Int32, nsrc * max_candidates * ntetra)
        expected_margin = zeros(Float64, nsrc * max_candidates * ntetra)
        for t in 1:ntetra
            center = ref.delaunay.circumcenters[t, :]
            p1 = ref.delaunay.points[ref.delaunay.tetras[t][1], :]
            radius2 = sum(abs2, p1 .- center)
            for slot in 1:max_candidates, source in 1:nsrc
                row = (t - 1) * nsrc * max_candidates +
                      (slot - 1) * nsrc + source
                slot <= expected_counts[source] || continue
                idx = (source - 1) * max_candidates + slot
                cand = Int(expected_stencil_candidate[idx])
                shift = (Int(expected_stencil_sx[idx]),
                         Int(expected_stencil_sy[idx]),
                         Int(expected_stencil_sz[idx]))
                pos = [pts[cand, 1] + shift[1],
                       pts[cand, 2] + shift[2],
                       pts[cand, 3] + shift[3]]
                margin = radius2 - sum(abs2, pos .- center)
                expected_valid[row] = 1
                expected_margin[row] = margin
                expected_inside[row] = margin >= -1e-10 ? 1 : 0
            end
        end
        @test Array(predicates.valid) == expected_valid
        @test Array(predicates.inside) == expected_inside
        @test Array(predicates.margin) ≈ expected_margin
        @test sum(Int.(Array(predicates.inside))) > 0

        conflict_faces = candidate_conflict_face_rows_soa_3d(
            KernelAbstractions.CPU(), predicates, be_soa.delaunay;
            index_type = Int32)
        @test conflict_faces isa CandidateConflictFaceRows3D
        @test conflict_faces.source_count == nsrc
        @test conflict_faces.tetra_count == ntetra
        @test conflict_faces.max_candidates_per_source == max_candidates
        total_conflict_faces = nsrc * max_candidates * ntetra * 4
        @test length(conflict_faces.active) == total_conflict_faces
        expected_face_source = Vector{Int32}(undef, total_conflict_faces)
        expected_face_slot = Vector{Int32}(undef, total_conflict_faces)
        expected_face_tetra = Vector{Int32}(undef, total_conflict_faces)
        expected_face_local = Vector{Int32}(undef, total_conflict_faces)
        expected_face_v1 = Vector{Int32}(undef, total_conflict_faces)
        expected_face_v2 = Vector{Int32}(undef, total_conflict_faces)
        expected_face_v3 = Vector{Int32}(undef, total_conflict_faces)
        expected_face_active = zeros(Int32, total_conflict_faces)
        face_lut = ((1, 2, 3), (1, 4, 2), (2, 4, 3), (3, 4, 1))
        for t in 1:ntetra
            tet = ref.delaunay.tetras[t]
            for lf in 1:4, slot in 1:max_candidates, source in 1:nsrc
                row = (t - 1) * nsrc * max_candidates * 4 +
                      (lf - 1) * nsrc * max_candidates +
                      (slot - 1) * nsrc + source
                pred_row = (t - 1) * nsrc * max_candidates +
                           (slot - 1) * nsrc + source
                face = Tuple(sort(collect((tet[face_lut[lf][1]],
                                           tet[face_lut[lf][2]],
                                           tet[face_lut[lf][3]]))))
                expected_face_source[row] = source
                expected_face_slot[row] = slot
                expected_face_tetra[row] = t
                expected_face_local[row] = lf
                expected_face_v1[row] = face[1]
                expected_face_v2[row] = face[2]
                expected_face_v3[row] = face[3]
                expected_face_active[row] = expected_valid[pred_row] == 1 &&
                                            expected_inside[pred_row] == 1 ? 1 : 0
            end
        end
        @test Array(conflict_faces.source) == expected_face_source
        @test Array(conflict_faces.slot) == expected_face_slot
        @test Array(conflict_faces.tetra) == expected_face_tetra
        @test Array(conflict_faces.local_face) == expected_face_local
        @test Array(conflict_faces.face_v1) == expected_face_v1
        @test Array(conflict_faces.face_v2) == expected_face_v2
        @test Array(conflict_faces.face_v3) == expected_face_v3
        @test Array(conflict_faces.active) == expected_face_active

        recomputed = recompute_circumcenters_soa_3d(KernelAbstractions.CPU(),
                                                    be_soa.delaunay;
                                                    tol = 1e-10)
        @test all(==(1), Array(recomputed.valid))
        @test Array(recomputed.x) ≈ ref.delaunay.circumcenters[:, 1]
        @test Array(recomputed.y) ≈ ref.delaunay.circumcenters[:, 2]
        @test Array(recomputed.z) ≈ ref.delaunay.circumcenters[:, 3]

        state = euler_state_3d(ref.geom; rho = 1.0, pressure = 1.0,
                               gamma = 1.4)
        total0 = total_conserved_3d(state, ref.geom)
        finite_volume_step_3d!(state, ref.geom; dt = 1e-3, gamma = 1.4,
                               riemann = :hll)
        total1 = total_conserved_3d(state, ref.geom)
        prim = conserved_to_primitive_3d(state; gamma = 1.4)
        @test total1.mass ≈ total0.mass
        @test total1.energy ≈ total0.energy
        @test prim.rho ≈ ones(length(ref.geom.volume))
        @test prim.pressure ≈ ones(length(ref.geom.volume))
    end

    @testset "AREPO hydro timestep helper follows Courant radius" begin
        volume = fill(1 / 8, 8)
        pressure = ones(8)
        rho = ones(8)
        dt = arepo_hydro_dt_3d(volume, pressure, rho; gamma = 1.4,
                               courant = 0.3, max_dt = 1.0, min_dt = 1e-6)
        radius = cbrt(3 * volume[1] / (4pi))
        @test dt ≈ fill(0.3 * radius / sqrt(1.4), 8)
        vel = hcat(fill(0.5, 8), zeros(8), zeros(8))
        mesh_vel = zeros(8, 3)
        dt_moving = arepo_hydro_dt_3d(volume, pressure, rho; gamma = 1.4,
                                      courant = 0.3, max_dt = 1.0,
                                      min_dt = 1e-6, velocity = vel,
                                      mesh_velocity = mesh_vel)
        @test dt_moving ≈ fill(0.3 * radius / (sqrt(1.4) + 0.5), 8)
        @test arepo_timebin_3d([0.125, 0.25, 0.5]; timebase_interval = 0.125) == [0, 1, 2]
        @test arepo_system_step_3d([0.046526287]; timebase_interval = 2.0^-28) ≈ [0.03125]
        @test arepo_system_step_3d([0.023263143]; timebase_interval = 2.0^-28) ≈ [0.015625]
        hierarchy = arepo_hydro_timebins_3d(volume, pressure, rho; gamma = 1.4,
                                            courant = 0.3, max_dt = 1.0,
                                            min_dt = 1e-6,
                                            timebase_interval = 0.125)
        @test hierarchy.integer_steps == (1 .<< hierarchy.bins)
        @test arepo_active_cells_3d([0, 1, 2], 0) == [true, true, true]
        @test arepo_active_cells_3d([0, 1, 2], 1) == [true, false, false]
        @test arepo_active_cells_3d([0, 1, 2], 2) == [true, true, false]
        @test arepo_active_cells_3d([0, 1, 2], 4) == [true, true, true]
        @test arepo_next_sync_step_3d([0, 1, 2], 0) == 1
        @test arepo_next_sync_step_3d([0, 1, 2], 1) == 1
        @test arepo_next_sync_step_3d([1, 2], 2) == 2
    end

    @testset "3D active face table packs only active cells" begin
        geom = cartesian_periodic_mesh_arrays_3d(2; T = Float64)
        nc = length(geom.volume)
        active = falses(nc)
        active[[1, 3, 8]] .= true
        tbl = active_face_table_3d(geom, active)
        counts = Array(tbl.active_counts)
        @test counts[active] == Int32.(diff(geom.cell_face_offsets))[active]
        @test all(counts[.!active] .== 0)
        @test tbl.active_stride == maximum(Int.(diff(geom.cell_face_offsets)))
        for i in findall(active)
            p0 = Int(geom.cell_face_offsets[i])
            p1 = Int(geom.cell_face_offsets[i + 1]) - 1
            for (q, p) in enumerate(p0:p1)
                idx = (i - 1) * tbl.active_stride + q
                @test Array(tbl.active_faces)[idx] == geom.cell_faces[p]
                @test Array(tbl.active_signs)[idx] == geom.cell_face_signs[p]
            end
        end
    end

    @testset "3D update-target CSR handles one-sided periodic rows" begin
        base_offsets, base_faces, base_signs = PowerFoam._cell_face_csr(2, Int32[1], Int32[0], Int32)
        geom_base = ArepoMeshArrays3D(Int32[1], Int32[0], base_offsets,
                                      base_faces, base_signs,
                                      ones(2), [1.0], [1.0], [0.0], [0.0],
                                      [0.0], [0.0], [0.0])
        geom = with_update_targets_3d(geom_base, Int32[1], Int32[2])
        @test Int.(geom.c2) == [0]
        @test Int.(diff(geom.cell_face_offsets)) == [1, 1]
        @test Int.(geom.cell_face_signs) == [-1, 1]
        @test face_update_activity_3d(Int32[1], Int32[2]) == Int32[1]
        @test face_update_activity_3d(Int32[1], Int32[0]) == Int32[1]
        @test face_update_activity_3d(Int32[0], Int32[0]) == Int32[0]
        side2_offsets, side2_faces, side2_signs =
            PowerFoam._cell_face_csr(2, Int32[0], Int32[2], Int32)
        @test Int.(diff(side2_offsets)) == [0, 1]
        @test Int.(side2_faces) == [1]
        @test Int.(side2_signs) == [1]
        state = EulerState3D([1.0, 1.0], [0.0, 0.0], [0.0, 0.0],
                             [0.0, 0.0], [2.5, 2.5])
        flux = FaceFluxWork3D([0.2], [0.3], [0.4], [0.5], [0.6])
        be = KernelAbstractions.CPU()
        PowerFoam._cell_update_3d_k!(be)(
            state.D, state.Mx, state.My, state.Mz, state.E,
            flux.FD, flux.FMx, flux.FMy, flux.FMz, flux.FE,
            geom.volume, geom.volume, geom.cell_face_offsets,
            geom.cell_faces, geom.cell_face_signs, 0.5;
            ndrange = 2)
        KernelAbstractions.synchronize(be)
        @test state.D ≈ [0.9, 1.1]
        @test state.Mx ≈ [-0.15, 0.15]
        @test state.My ≈ [-0.2, 0.2]
        @test state.Mz ≈ [-0.25, 0.25]
        @test state.E ≈ [2.2, 2.8]
    end

    @testset "AREPO mesh velocity reconstruction applies pressure half-step" begin
        pos = [0.25 0.25 0.25;
               0.75 0.25 0.25]
        center = copy(pos)
        rho = ones(2)
        pressure = ones(2)
        vel = [0.1 0.2 0.3;
               -0.1 0.0 0.05]
        gradients = (; dpress = [2.0 0.0 -1.0;
                                 0.0 -4.0 0.0])
        geometry = (; c1 = Int[], c2 = Int[], nv = Int[], normals = zeros(0, 3),
                    verts = zeros(0, 3))
        vmesh = arepo_mesh_velocity_3d(pos, center, rho, pressure, vel,
                                       gradients, ones(2), geometry;
                                       dt = 0.25, gamma = 1.4,
                                       use_face_angle = false)
        @test vmesh[1, :] ≈ [0.1 - 0.25, 0.2, 0.3 + 0.125]
        @test vmesh[2, :] ≈ [-0.1, 0.0 + 0.5, 0.05]
    end

    @testset "3D reconstructed hierarchy step preserves uniform active cells" begin
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
        state = euler_state_3d(geom; rho = 1.0, vx = 0.0, vy = 0.0,
                               vz = 0.0, pressure = 1.0, gamma = 1.4)
        prim = primitive_work_3d(state)
        conserved_to_primitive_3d!(prim, state; gamma = 1.4)
        z = zeros(nc)
        gradients = HydroGradients3D((copy(z) for _ in 1:15)...)
        bins = fill(1, nc)
        result = finite_volume_reconstructed_hierarchy_step_3d!(
            state, geom, gradients, prim,
            collect(view(center, :, 1)), collect(view(center, :, 2)),
            collect(view(center, :, 3)),
            collect(view(face_center, :, 1)), collect(view(face_center, :, 2)),
            collect(view(face_center, :, 3)), bins;
            ti_current = 0, timebase_interval = 0.125,
            gamma = 1.4, riemann = :hll)
        p = conserved_to_primitive_3d(state; gamma = 1.4)
        @test result.ti_step == 2
        @test result.ti_next == 2
        @test result.dt ≈ 0.25
        @test all(result.active)
        @test p.rho ≈ ones(nc)
        @test p.pressure ≈ ones(nc)
    end

    @testset "AREPO PM gravity preflight produces numeric diagnostics" begin
        fixture = arepo_pm_gravity_fixture()
        direct0 = periodic_image_sum_accel(fixture.x, fixture.y, fixture.z, fixture.m;
                                           boxsize = fixture.boxsize, nimg = 0)
        open_box = arepo_direct_gravity_accel(fixture.x, fixture.y, fixture.z, fixture.m)
        @test direct0[1] ≈ open_box[1]
        @test direct0[2] ≈ open_box[2]
        @test direct0[3] ≈ open_box[3]
        oracle = periodic_background_subtracted_image_oracle(
            fixture.x, fixture.y, fixture.z, fixture.m;
            boxsize = fixture.boxsize, nimg = 2, previous_nimg = 1)
        @test oracle.nimg == 2
        @test oracle.previous_nimg == 1
        @test oracle.shell_max_component_change !== nothing
        @test abs(oracle.neutralized_net_force.x) < 1e-12
        @test abs(oracle.neutralized_net_force.y) < 1e-12
        @test abs(oracle.neutralized_net_force.z) < 1e-12

        fallback = run_arepo_pm_gravity_preflight(nothing; fixture = fixture)
        @test any(row -> row.category == "direct_diag" && row.value !== nothing, fallback.rows)
        @test any(row -> row.category == "direct_oracle" &&
                         row.label == "periodic_background_subtracted_image_sum" &&
                         row.status == "ok", fallback.rows)
        @test any(row -> row.category == "blocker" &&
                         row.label == "pm_fft_chain" &&
                         row.status == "blocker", fallback.rows)
        pk_probe = PowerFoam.probe_poissonkernels_monorepo()
        scoped = run_arepo_pm_gravity_preflight(pk_probe.pm_module; fixture = fixture,
                                                pk_probe = pk_probe)
        if pk_probe.pm_module === nothing
            @test any(row -> row.label == "poissonkernels_load_error", scoped.rows)
        else
            @test scoped.pm !== nothing
            @test any(row -> row.category == "pm" && row.label == "mass_sum", scoped.rows)
        end

        pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..", "PoissonKernels")))
        have_pk = Base.find_package("PoissonKernels") !== nothing &&
                  Base.find_package("FFTW") !== nothing &&
                  Base.find_package("KernelAbstractions") !== nothing
        if have_pk
            try
                @eval using PoissonKernels
            catch
                have_pk = false
            end
        end

        if have_pk
            result = run_arepo_pm_gravity_preflight(PoissonKernels; fixture = fixture)
            mass_row = only(filter(row -> row.category == "pm" && row.label == "mass_sum",
                                   result.rows))
            rhs_row = only(filter(row -> row.category == "pm" && row.label == "rhs_sum",
                                  result.rows))
            diff_row = only(filter(row -> row.category == "pm_vs_direct_oracle" &&
                                          occursin("max_component_diff", row.label),
                                   result.rows))
            self_row = only(filter(row -> row.category == "pm_self_control" &&
                                          row.label == "one_particle_max_abs_accel",
                                   result.rows))
            @test mass_row.delta !== nothing
            @test abs(mass_row.delta) < 1e-12
            @test rhs_row.value !== nothing
            @test abs(rhs_row.value) < 1e-12
            @test diff_row.value !== nothing
            @test isfinite(diff_row.value)
            @test self_row.value !== nothing
            @test self_row.value < 1e-12
            @test result.pm !== nothing
            @test length(result.pm.ax) == length(fixture.x)
        else
            @test_skip "PoissonKernels not available on LOAD_PATH in PowerFoam test env"
        end
    end

    @testset "AREPO standard problem Noh2D helper runs a short executable rung" begin
        built = pf_noh2d_mesh(6; domain_radius = 3.0)
        prim = pf_noh2d_initial_primitives(built.mesh; rho0 = 1.0, p0 = 1e-4, vrad = 1.0)
        @test length(prim.rho) == 36
        @test all(==(1.0), prim.rho)
        @test minimum(prim.pressure) == 1e-4
        @test maximum(hypot.(prim.vx, prim.vy)) ≤ 1.0 + 1e-12

        run = pf_noh2d_run(; n_side = 6, t_final = 0.03, nbins = 6, cfl = 0.12,
                           domain_radius = 3.0, riemann = :hll, max_steps = 256)
        @test !isempty(run.history)
        @test length(run.radial_bins) == 6
        @test run.history[end].t ≈ 0.03 atol = 1e-12
        @test run.history[end].rho_min > 0
        @test run.history[end].p_min > 0
        @test isfinite(run.history[end].mass_rel_drift)
        @test isfinite(run.history[end].energy_rel_drift)
        @test run.status in ("calibration-PENDING", "run-FAIL")
    end
end

@testset "AREPO snapshot IO runtime preflight" begin
    payload = (
        header = (time = 0.0, box_size = 1.0, num_files = 1),
        gas = (
            density = [1.0, 0.5],
            masses = [0.25, 0.25],
            internal_energy = [2.0, 2.5],
            velocities = [0.0 0.1 0.0;
                          0.0 0.0 0.2],
            Coordinates = [0.25 0.25 0.25;
                           0.75 0.25 0.25],
            particle_ids = [1, 2],
        ),
    )

    snapshot = read_arepo_snapshot(payload; root = "memory", snapshot_index = 1)
    validation = validate_arepo_snapshot(snapshot)
    @test validation.valid
    @test snapshot.gas.density == [1.0, 0.5]
    @test snapshot.gas.masses == [0.25, 0.25]
    @test snapshot.gas.internal_energy == [2.0, 2.5]
    @test snapshot.gas.velocities == [0.0 0.1 0.0;
                                      0.0 0.0 0.2]
    @test snapshot.derived.volume_derived
    @test snapshot.derived.pressure_derived
    @test snapshot.derived.center_derived
    @test snapshot.gas.volume ≈ [0.25, 0.5]
    @test snapshot.gas.pressure ≈ [4 / 3, 5 / 6]
    @test snapshot.gas.center ≈ [0.25 0.25 0.25;
                                 0.75 0.25 0.25]
    @test snapshot.gas.particle_ids == [1, 2]
    @test :volume in snapshot.header.fields_present
    @test :pressure in snapshot.header.fields_present
    @test :center in snapshot.header.fields_present
    @test :particle_ids in snapshot.header.fields_present

    locator = locate_arepo_snapshot(mktempdir(), 1)
    @test locator.layout == :planned
    @test endswith(locator.resolved_paths[1], "snap_001.hdf5")
    @test endswith(locator.resolved_paths[2], "snapdir_001/snap_001.0.hdf5")

    splitdir = mktempdir()
    splitpath = joinpath(splitdir, "snapdir_001")
    mkpath(splitpath)
    touch(joinpath(splitpath, "snap_001.0.hdf5"))
    split_locator = locate_arepo_snapshot(splitdir, 1)
    @test split_locator.layout == :split
    split_preflight = arepo_snapshot_read_preflight(splitdir, 1)
    @test split_preflight.ok
    @test split_preflight.status == :ready
    @test split_preflight.path == split_locator.resolved_paths[2]

    ambiguousdir = mktempdir()
    touch(joinpath(ambiguousdir, "snap_001.hdf5"))
    mkpath(joinpath(ambiguousdir, "snapdir_001"))
    touch(joinpath(ambiguousdir, "snapdir_001", "snap_001.0.hdf5"))
    ambiguous_locator = locate_arepo_snapshot(ambiguousdir, 1)
    @test ambiguous_locator.layout == :ambiguous
    ambiguous_preflight = arepo_snapshot_read_preflight(ambiguousdir, 1)
    @test !ambiguous_preflight.ok
    @test ambiguous_preflight.status == :ambiguous_layout

    tmpdir = mktempdir()
    result = write_arepo_snapshot(joinpath(tmpdir, "snap_001.hdf5"), snapshot)
    @test result.ok
    @test result.status in (:preflight_only, :wrote_hdf5)
    if result.status == :wrote_hdf5
        reread = read_arepo_snapshot(tmpdir, 1)
        @test reread.header.time == snapshot.header.time
        @test reread.gas.density == snapshot.gas.density
    else
        @test result.backend == :preflight
        @test any(contains("preflight"), result.messages)
    end
end
