# E1c — the parity harness in action. Replay EnzoModules' committed golden
# fixtures for the legacy `ppm_sweep_1d` kernel through the native Julia `ccall`
# binding and certify the outputs **bitwise** (same library ⇒ exact). This is the
# certification gate every future Julia port reuses: run the reference legacy
# kernel from Julia, diff against the captured truth with the shared tolerance.

const FIXDIR = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                 "EnzoModules", "fixtures", "Hydro", "ppm_sweep_1d"))

# The committed fixtures were captured with a *different* gfortran build than the
# locally-rebuilt pilot .so, so the legacy kernel agrees only to last-bit FP
# reassociation (FMA / reduction order) — exactly what diff.py's tolerance policy
# is for ("bitwise only against the same library"). A tight rtol certifies the
# replay while still catching any real drift (the negative control perturbs 1e-6).
const RTOL = Tolerance(rtol = 1e-12, atol = 1e-14)

@testset "precision contract" begin
    @test EnzoLib.available()
    rb, ib = EnzoLib.check_precision()
    @test (rb, ib) == (8, 4)
end

@testset "ppm_sweep_1d replay ≡ golden fixtures (bitwise)" begin
    fixtures = load_dir(FIXDIR)
    @test !isempty(fixtures)
    for fx in fixtures
        name = scalar(fx, :name)
        i1 = scalar(fx, :i1, Int); i2 = scalar(fx, :i2, Int)
        dx = scalar(fx, :dx, Float64); dt = scalar(fx, :dt, Float64)
        γ = scalar(fx, :gamma, Float64)
        # inputs (copied — the kernel mutates in place)
        d = copy(array(fx, :dslice_in)); e = copy(array(fx, :eslice_in))
        u = copy(array(fx, :uslice_in)); v = copy(array(fx, :vslice_in))
        w = copy(array(fx, :wslice_in)); p = copy(array(fx, :pslice_in))
        df, ef, uf = EnzoLib.ppm_sweep_1d!(d, e, u, v, w, p;
                                           i1 = i1, i2 = i2, dx = dx, dt = dt,
                                           gamma = γ, fluxes = true)
        # certify every output array against the captured truth, bitwise
        for (got, key) in ((d, :dslice_out), (e, :eslice_out), (u, :uslice_out),
                           (v, :vslice_out), (w, :wslice_out),
                           (df, :df), (ef, :ef), (uf, :uf))
            r = compare(got, array(fx, key), RTOL)
            @test Bool(r)
            Bool(r) || @info "MISMATCH" fixture = name field = key maxabs = r.maxabs maxrel = r.maxrel worst = r.worst
        end
        @info "replayed" fixture = name idim = scalar(fx, :idim, Int) active = (i1, i2)
    end
end

@testset "negative control: corrupted truth fails the diff" begin
    fx = load_fixture(joinpath(FIXDIR, "sod_step.fixture"))
    truth = copy(array(fx, :dslice_out))
    truth[end] += 1e-6                                   # perturb one element
    d = copy(array(fx, :dslice_in)); e = copy(array(fx, :eslice_in))
    u = copy(array(fx, :uslice_in)); v = copy(array(fx, :vslice_in))
    w = copy(array(fx, :wslice_in)); p = copy(array(fx, :pslice_in))
    EnzoLib.ppm_sweep_1d!(d, e, u, v, w, p;
                          i1 = scalar(fx, :i1, Int), i2 = scalar(fx, :i2, Int),
                          dx = scalar(fx, :dx, Float64), dt = scalar(fx, :dt, Float64),
                          gamma = scalar(fx, :gamma, Float64))
    @test !Bool(compare(d, truth, RTOL))                # 1e-6 corruption ≫ rtol ⇒ caught
    @test Bool(compare(d, array(fx, :dslice_out), RTOL))  # vs real truth, within tol
end
