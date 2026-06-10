# ── the gravity guest slot: KA Poisson kernels inside RAMSES (ADR-0006 D4) ────
#
# The first slice of the mini-ramses-kernel KA re-expression (the ADR "Next"
# list): RAMSES owns the mesh, the density deposit (`rho_fine`), and the force
# differencing (`force_fine`); the GUEST solves its Poisson equation
#
#   ∇²φ = 4π·(ρ − ρ̄)            (fourpi → 1.5·Ωm·aexp on a cosmo build)
#
# with PoissonKernels' FFT solve using the DISCRETE 7-point Green's function —
# the exact solution of the same linear system RAMSES's own multigrid/CG
# iterates on, which is what makes the certification tight: tighten RAMSES's
# `epsilon` and the two potentials must converge on each other.

"""
    ramses_grid_field(h, which, lev; lib=:cpu) -> (ck, A)

Read mesh field `which` (:rho/:phi/:fx/…) on a UNIFORM level as an `n1d³`
`Array{Float64,3}` (the same exact oct→Cartesian addressing as the hydro
raster — pure addressing, no physics).
"""
function ramses_grid_field(h::RamsesLib.Handle, which::Symbol, lev::Integer; lib::Symbol = :cpu)
    ck, val = RamsesLib.get_field(h, which, lev; lib = lib)
    noct = size(ck, 1)
    n1d = 2^lev
    8 * noct == n1d^3 || error("ramses_grid_field: level $lev is not the full uniform grid")
    A = Array{Float64,3}(undef, n1d, n1d, n1d)
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * Int(ck[o, 1]) + ((c - 1) & 1) + 1
        j = 2 * Int(ck[o, 2]) + ((c - 1) >> 1 & 1) + 1
        k = 2 * Int(ck[o, 3]) + ((c - 1) >> 2 & 1) + 1
        A[i, j, k] = val[o, c]
    end
    return ck, A
end

"Write an `n1d³` array back into mesh field `which` on a uniform level (inverse addressing)."
function ramses_set_grid_field!(h::RamsesLib.Handle, which::Symbol, lev::Integer,
                                ck::Matrix{Int32}, A::Array{Float64,3}; lib::Symbol = :cpu)
    noct = size(ck, 1)
    val = Matrix{Float64}(undef, noct, 8)
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * Int(ck[o, 1]) + ((c - 1) & 1) + 1
        j = 2 * Int(ck[o, 2]) + ((c - 1) >> 1 & 1) + 1
        k = 2 * Int(ck[o, 3]) + ((c - 1) >> 2 & 1) + 1
        val[o, c] = A[i, j, k]
    end
    RamsesLib.set_field!(h, which, lev, ck, val; lib = lib)
    return nothing
end

"""
    ramses_ka_poisson!(h; lev, boxlen=1.0, fourpi=4π, greens=:discrete7, lib=:cpu)
        -> (; phi, rho)

THE gravity slot step: replace RAMSES's `multigrid`/`phi_fine_cg` on a uniform
level with PoissonKernels' periodic FFT solve.  Reads the host's deposited
`rho` (whatever `rho_fine` produced), solves `∇²φ = fourpi·(ρ − mean ρ)` in the
zero-mean gauge, and writes `phi` back into the live mesh — `force_fine` and
the kick see exactly the layout the host's own solver would have produced.
"""
function ramses_ka_poisson!(h::RamsesLib.Handle; lev::Integer, boxlen::Real = 1.0,
                            fourpi::Real = 4π, greens::Symbol = :discrete7,
                            lib::Symbol = :cpu)
    ck, rho = ramses_grid_field(h, :rho, lev; lib = lib)
    rhs = rho .- (sum(rho) / length(rho))         # the periodic null space (unweighted mean)
    phi = similar(rhs)
    PoissonKernels.fft_poisson_root!(phi, rhs; G = fourpi, a = 1.0,
                                     boxsize = Float64(boxlen), greens = greens)
    ramses_set_grid_field!(h, :phi, lev, ck, phi; lib = lib)
    return (phi = phi, rho = rho)
end

# ── the certification driver: oracle vs guest on the identical deposit ────────

function _ramses_gravity_namelist(; level::Integer)
    return """
    Uniform box for the gravity-slot certification (MultiCode.jl)

    &RUN_PARAMS
    hydro=.true.
    poisson=.true.
    ncontrol=1
    nrestart=0
    nremap=0
    nsubcycle=10*1
    nstepmax=100000
    nsuperoct=2
    verbose=.false.
    /

    &AMR_PARAMS
    levelmin=$(level)
    levelmax=$(level)
    ngridtot=3000000
    ncachemax=30000
    nexpand=1
    boxlen=1.0
    /

    &INIT_PARAMS
    nregion=1
    region_type(1)='square'
    x_center=0.5
    y_center=0.5
    z_center=0.5
    length_x=10.0
    length_y=10.0
    length_z=10.0
    exp_region=10.0
    d_region=1.0
    u_region=0.0
    v_region=0.0
    w_region=0.0
    p_region=1.0
    /

    &OUTPUT_PARAMS
    foutput=0
    tout=100.0
    /

    &HYDRO_PARAMS
    gamma=1.4
    courant_factor=0.8
    slope_type=1
    scheme='muscl'
    riemann='hllc'
    /

    &POISSON_PARAMS
    epsilon=1.0d-12
    epsilon_base=1.0d-12
    /

    &REFINE_PARAMS
    interpol_type=0
    interpol_var=0
    /
    """
end

"""
    run_ramses_gravity_compare(; level=5, amp=0.05, eps=1e-12) -> (; ...)

One density field, two Poisson solvers, one discrete system: perturb the gas
density on a uniform level (a zero-mean product-of-sines mode), let the HOST
deposit it (`rho_fine` — the guest consumes whatever the host deposited, not
the injected analytic field), then solve with (a) RAMSES's own multigrid at
tolerance `eps` — the ORACLE — and (b) the KA FFT solve with the discrete
7-point Green's function, and difference the potentials (mean-removed — the
periodic gauge) and the `force_fine` accelerations from each.
"""
function run_ramses_gravity_compare(; level::Integer = 5, amp::Real = 0.05,
                                    eps::Real = 1e-12, lib::Symbol = :cpu)
    RamsesLib.available() || error("RAMSES library not found (set RAMSES_LIB to the bin64h build)")
    n = 2^level
    dir = mktempdir()
    write(joinpath(dir, "gravity.nml"), _ramses_gravity_namelist(level = level))
    return cd(dir) do
        h = RamsesLib.init("gravity.nml"; lib = lib)
        lev = RamsesLib.info(h; lib = lib).levelmin
        RamsesLib.set_epsilon!(h, eps; lib = lib)
        # ── a smooth zero-mean density mode, injected into the GAS state ──────
        ck, U = RamsesLib.get_hydro_all(h, :uold, lev; lib = lib)
        noct = size(ck, 1)
        rho_in = Matrix{Float64}(undef, noct, 8)
        @inbounds for o in 1:noct, c in 1:8
            x = (2 * ck[o, 1] + ((c - 1) & 1) + 0.5) / n
            y = (2 * ck[o, 2] + ((c - 1) >> 1 & 1) + 0.5) / n
            z = (2 * ck[o, 3] + ((c - 1) >> 2 & 1) + 0.5) / n
            rho_in[o, c] = 1.0 + amp * sin(2π * x) * sin(4π * y) * cos(2π * z)
        end
        RamsesLib.set_hydro!(h, :uold, 1, lev, ck, rho_in; lib = lib)
        # the HOST deposits: rho := what its own solver would see
        RamsesLib.rho_fine!(h, lev, 0; lib = lib)
        _, rho = ramses_grid_field(h, :rho, lev; lib = lib)
        # ── ORACLE: RAMSES's own multigrid + force ────────────────────────────
        RamsesLib.multigrid!(h, lev, 1; lib = lib)
        ckf, phi_mg = ramses_grid_field(h, :phi, lev; lib = lib)
        RamsesLib.force_fine!(h, lev, 1; lib = lib)
        _, fx_mg = ramses_grid_field(h, :fx, lev; lib = lib)
        _, fy_mg = ramses_grid_field(h, :fy, lev; lib = lib)
        _, fz_mg = ramses_grid_field(h, :fz, lev; lib = lib)
        # ── GUEST: the KA discrete-Green's FFT solve + the SAME force_fine ────
        ka = ramses_ka_poisson!(h; lev = lev, boxlen = 1.0, fourpi = 4π,
                                greens = :discrete7, lib = lib)
        RamsesLib.force_fine!(h, lev, 1; lib = lib)
        _, fx_ka = ramses_grid_field(h, :fx, lev; lib = lib)
        _, fy_ka = ramses_grid_field(h, :fy, lev; lib = lib)
        _, fz_ka = ramses_grid_field(h, :fz, lev; lib = lib)
        # ── difference in the zero-mean gauge ─────────────────────────────────
        c_mg = phi_mg .- (sum(phi_mg) / length(phi_mg))
        c_ka = ka.phi .- (sum(ka.phi) / length(ka.phi))
        scale = maximum(abs, c_mg)
        dphi = maximum(abs, c_mg .- c_ka) / scale
        fscale = max(maximum(abs, fx_mg), maximum(abs, fy_mg), maximum(abs, fz_mg))
        df = max(maximum(abs, fx_mg .- fx_ka), maximum(abs, fy_mg .- fy_ka),
                 maximum(abs, fz_mg .- fz_ka)) / fscale
        # how non-trivial the deposit was (guards against a vacuous gate)
        rho_dev = maximum(abs, rho .- (sum(rho) / length(rho)))
        return (dphi = dphi, dforce = df, phi_scale = scale, force_scale = fscale,
                rho_dev = rho_dev, n = n, handle = h,
                free = () -> RamsesLib.finalize(h; lib = lib))
    end
end
