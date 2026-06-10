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

# ── the refined-level Dirichlet solve (Next-4 slice 2) ────────────────────────
#
# RAMSES's fine-level Poisson system (cmp_residual_cg, phi_fine_cg.f90):
#
#   Σ_nbr φ − 6φ = 4π·dx²·(ρ − ρ̄)
#
# over the level's cells, where a neighbor in a MISSING oct takes the value
# `interpol_phi` produces from the 27 coarse parent cells around that oct
# (CIC weights + tfrac time extrapolation; tfrac = 0 when not subcycling).
# For a CUBOID refined region this is exactly a Dirichlet box problem: ghost
# ring = the interpolated virtual-oct values, interior = unknowns — which is
# `vcycle_solve!(dirichlet = true)` on a (n+2)-padded array, with Enzo's MG
# rhs scaling  rhs = (d₁−1)(d₂−1)(d₃−1)·4π·dx²·(ρ − ρ̄).

"Raster one mesh field of a (possibly partial) level into its bbox; assert it is a full cuboid."
function _ramses_field_bbox(h::RamsesLib.Handle, which::Symbol, lev::Integer; lib::Symbol = :cpu)
    ck, val = RamsesLib.get_field(h, which, lev; lib = lib)
    noct = size(ck, 1)
    noct > 0 || error("_ramses_field_bbox: level $lev is empty")
    lo = (typemax(Int), typemax(Int), typemax(Int))
    hi = (typemin(Int), typemin(Int), typemin(Int))
    @inbounds for o in 1:noct
        b = (2 * Int(ck[o, 1]), 2 * Int(ck[o, 2]), 2 * Int(ck[o, 3]))
        lo = min.(lo, b); hi = max.(hi, b .+ 1)
    end
    nloc = hi .- lo .+ 1
    8 * noct == prod(nloc) ||
        error("_ramses_field_bbox: level $lev is not a full cuboid " *
              "($(8noct) cells in a $(nloc) bbox) — the Dirichlet slot needs one")
    A = Array{Float64,3}(undef, nloc...)
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * Int(ck[o, 1]) + ((c - 1) & 1) - lo[1] + 1
        j = 2 * Int(ck[o, 2]) + ((c - 1) >> 1 & 1) - lo[2] + 1
        k = 2 * Int(ck[o, 3]) + ((c - 1) >> 2 & 1) - lo[3] + 1
        A[i, j, k] = val[o, c]
    end
    return (ck = ck, A = A, off = lo, nloc = nloc, n1d = 2^lev)
end

"Write a bbox array back into a partial level's mesh field."
function _ramses_field_bbox_set!(h::RamsesLib.Handle, which::Symbol, lev::Integer,
                                 fb, A::Array{Float64,3}; lib::Symbol = :cpu)
    ck = fb.ck; noct = size(ck, 1)
    val = Matrix{Float64}(undef, noct, 8)
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * Int(ck[o, 1]) + ((c - 1) & 1) - fb.off[1] + 1
        j = 2 * Int(ck[o, 2]) + ((c - 1) >> 1 & 1) - fb.off[2] + 1
        k = 2 * Int(ck[o, 3]) + ((c - 1) >> 2 & 1) - fb.off[3] + 1
        val[o, c] = A[i, j, k]
    end
    RamsesLib.set_field!(h, which, lev, ck, val; lib = lib)
    return nothing
end

"""
Reconstruct the Dirichlet ghost ring exactly as RAMSES's fine solver sees it:
for each ghost cell (one ring around the fine cuboid) the value is
`interpol_phi` of the 27 coarse cells around the cell's (missing) oct's parent
cell, evaluated at the cell's child position.  `phic` is the FULL coarse-level
phi grid (periodic).  Fills the face ring of `sol` (size `fb.nloc .+ 2`).
"""
function _fine_dirichlet_ghosts!(sol::Array{Float64,3}, fb, phic::Array{Float64,3};
                                 tfrac::Real = 0.0, lib::Symbol = :cpu)
    n1dc = size(phic, 1)
    n1df = fb.n1d
    nx, ny, nz = size(sol)
    phi27 = Vector{Float64}(undef, 27)
    cache = Dict{NTuple{3,Int},Vector{Float64}}()   # per-virtual-oct interpolation
    for k in 1:nz, j in 1:ny, i in 1:nx
        onface = (i == 1 || i == nx || j == 1 || j == ny || k == 1 || k == nz)
        onface || continue
        g = (mod(fb.off[1] + (i - 2), n1df), mod(fb.off[2] + (j - 2), n1df),
             mod(fb.off[3] + (k - 2), n1df))
        oct = g .>> 1                                   # virtual fine oct == coarse cell
        out = get!(cache, oct) do
            for dk in 0:2, dj in 0:2, di in 0:2
                phi27[1 + di + 3dj + 9dk] =
                    phic[mod(oct[1] - 1 + di, n1dc) + 1,
                         mod(oct[2] - 1 + dj, n1dc) + 1,
                         mod(oct[3] - 1 + dk, n1dc) + 1]
            end
            RamsesLib.interpol_phi(phi27, phi27, tfrac; lib = lib)
        end
        c = 1 + (g[1] & 1) + 2 * (g[2] & 1) + 4 * (g[3] & 1)
        sol[i, j, k] = out[c]
    end
    return sol
end

# ── the IRREGULAR refined region (Next-6): the masked fine-level solve ────────
#
# Beyond cuboids: an arbitrary (blob) refined region is an irregular-domain
# Dirichlet problem — same 7-point system, same interpol_phi ghosts at every
# region-adjacent cell of a missing oct, but the unknowns live on a MASK, not
# a box.  Solved matrix-free with conjugate gradients (the operator is SPD on
# the masked cells; ghost contributions move to the RHS).  This is the
# oct-irregular capability; a KA-kernelized masked smoother is the
# performance follow-up.

"Masked bbox raster of one mesh field on a partial level (no cuboid assumption)."
function _ramses_field_bbox_masked(h::RamsesLib.Handle, which::Symbol, lev::Integer;
                                   lib::Symbol = :cpu)
    ck, val = RamsesLib.get_field(h, which, lev; lib = lib)
    noct = size(ck, 1)
    noct > 0 || error("_ramses_field_bbox_masked: level $lev is empty")
    lo = (typemax(Int), typemax(Int), typemax(Int))
    hi = (typemin(Int), typemin(Int), typemin(Int))
    @inbounds for o in 1:noct
        b = (2 * Int(ck[o, 1]), 2 * Int(ck[o, 2]), 2 * Int(ck[o, 3]))
        lo = min.(lo, b); hi = max.(hi, b .+ 1)
    end
    nloc = hi .- lo .+ 1
    A = zeros(Float64, (nloc .+ 2)...)               # 1-cell halo (ghost ring lives here)
    covered = falses((nloc .+ 2)...)
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * Int(ck[o, 1]) + ((c - 1) & 1) - lo[1] + 2
        j = 2 * Int(ck[o, 2]) + ((c - 1) >> 1 & 1) - lo[2] + 2
        k = 2 * Int(ck[o, 3]) + ((c - 1) >> 2 & 1) - lo[3] + 2
        A[i, j, k] = val[o, c]
        covered[i, j, k] = true
    end
    return (ck = ck, A = A, covered = covered, off = lo, nloc = nloc, n1d = 2^lev)
end

"Write the covered cells of a haloed masked array back into the level's field."
function _ramses_field_masked_set!(h::RamsesLib.Handle, which::Symbol, lev::Integer,
                                   fb, A::Array{Float64,3}; lib::Symbol = :cpu)
    ck = fb.ck; noct = size(ck, 1)
    val = Matrix{Float64}(undef, noct, 8)
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * Int(ck[o, 1]) + ((c - 1) & 1) - fb.off[1] + 2
        j = 2 * Int(ck[o, 2]) + ((c - 1) >> 1 & 1) - fb.off[2] + 2
        k = 2 * Int(ck[o, 3]) + ((c - 1) >> 2 & 1) - fb.off[3] + 2
        val[o, c] = A[i, j, k]
    end
    RamsesLib.set_field!(h, which, lev, ck, val; lib = lib)
    return nothing
end

"""
Fill every NON-covered cell that is face-adjacent to a covered cell with its
interpol_phi value (the virtual-oct ghost RAMSES's fine solver sees) — the
mask-driven generalization of the cuboid face ring.
"""
function _fine_dirichlet_ghosts_masked!(A::Array{Float64,3}, fb, phic::Array{Float64,3};
                                        tfrac::Real = 0.0, lib::Symbol = :cpu)
    n1dc = size(phic, 1); n1df = fb.n1d
    nx, ny, nz = size(A)
    phi27 = Vector{Float64}(undef, 27)
    cache = Dict{NTuple{3,Int},Vector{Float64}}()
    cov = fb.covered
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        cov[i, j, k] && continue
        adj = (i > 1 && cov[i-1, j, k]) || (i < nx && cov[i+1, j, k]) ||
              (j > 1 && cov[i, j-1, k]) || (j < ny && cov[i, j+1, k]) ||
              (k > 1 && cov[i, j, k-1]) || (k < nz && cov[i, j, k+1])
        adj || continue
        g = (mod(fb.off[1] + (i - 2), n1df), mod(fb.off[2] + (j - 2), n1df),
             mod(fb.off[3] + (k - 2), n1df))
        oct = g .>> 1
        out = get!(cache, oct) do
            for dk in 0:2, dj in 0:2, di in 0:2
                phi27[1 + di + 3dj + 9dk] =
                    phic[mod(oct[1] - 1 + di, n1dc) + 1,
                         mod(oct[2] - 1 + dj, n1dc) + 1,
                         mod(oct[3] - 1 + dk, n1dc) + 1]
            end
            RamsesLib.interpol_phi(phi27, phi27, tfrac; lib = lib)
        end
        A[i, j, k] = out[1 + (g[1] & 1) + 2 * (g[2] & 1) + 4 * (g[3] & 1)]
    end
    return A
end

# the masked 7-point operator  (A·x)[c] = 6x[c] − Σ_covered-nbr x[nbr]  (SPD)
function _masked_apply!(out::Array{Float64,3}, x::Array{Float64,3}, cov)
    nx, ny, nz = size(x)
    @inbounds for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
        cov[i, j, k] || continue
        s = 6.0 * x[i, j, k]
        cov[i-1, j, k] && (s -= x[i-1, j, k]); cov[i+1, j, k] && (s -= x[i+1, j, k])
        cov[i, j-1, k] && (s -= x[i, j-1, k]); cov[i, j+1, k] && (s -= x[i, j+1, k])
        cov[i, j, k-1] && (s -= x[i, j, k-1]); cov[i, j, k+1] && (s -= x[i, j, k+1])
        out[i, j, k] = s
    end
    return out
end

_masked_dot(a, b, cov) = (s = 0.0; @inbounds for q in eachindex(a)
                              cov[q] && (s += a[q] * b[q])
                          end; s)

"""
    ramses_ka_poisson_fine!(h; levc, levf, boxlen=1.0, fourpi=4π,
                            rtol=1e-12, maxiter=2000, lib=:cpu) -> (; ...)

The IRREGULAR-region fine-level Poisson solve: masked bbox raster, interpol_phi
ghosts on every region-adjacent missing-oct cell, matrix-free CG on
`6φ − Σφ_nbr = −4π·dx²·(ρ−ρ̄) + Σ ghosts`, write-back into the live `phi`.
Replaces `phi_fine_cg`/`multigrid` for ANY refined-region shape (the cuboid
fast path remains `vcycle_solve!(dirichlet=true)`).
"""
function ramses_ka_poisson_fine!(h::RamsesLib.Handle; levc::Integer, levf::Integer,
                                 boxlen::Real = 1.0, fourpi::Real = 4π,
                                 rtol::Real = 1e-12, maxiter::Integer = 2000,
                                 device::Symbol = :cpu, lib::Symbol = :cpu)
    _, phic = ramses_grid_field(h, :phi, levc; lib = lib)
    _, rhoc = ramses_grid_field(h, :rho, levc; lib = lib)
    rho_tot = sum(rhoc) / length(rhoc)
    rb = _ramses_field_bbox_masked(h, :rho, levf; lib = lib)
    cov = rb.covered
    dxf = Float64(boxlen) / rb.n1d
    # ghosts live in a zero x with ONLY boundary values set; their stencil
    # contribution moves to the RHS:  b = −fact·(ρ−ρ̄) + Σ_ghost-nbr g
    gh = zeros(Float64, size(rb.A))
    _fine_dirichlet_ghosts_masked!(gh, rb, phic; lib = lib)
    fact = Float64(fourpi) * dxf^2
    b = zeros(Float64, size(rb.A))
    nx, ny, nz = size(rb.A)
    @inbounds for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
        cov[i, j, k] || continue
        s = -fact * (rb.A[i, j, k] - rho_tot)
        cov[i-1, j, k] || (s += gh[i-1, j, k]); cov[i+1, j, k] || (s += gh[i+1, j, k])
        cov[i, j-1, k] || (s += gh[i, j-1, k]); cov[i, j+1, k] || (s += gh[i, j+1, k])
        cov[i, j, k-1] || (s += gh[i, j, k-1]); cov[i, j, k+1] || (s += gh[i, j, k+1])
        b[i, j, k] = s
    end
    # the KA masked CG (PoissonKernels.masked_cg!): one source, CPU f64 or
    # Metal f32 — the mask travels as a FIELD so the kernel is branch-free
    mfield = Float64.(cov)
    be = PoissonKernels.backend(device)
    T = device === :cpu ? Float64 : Float32
    xd = PoissonKernels.device_zeros(be, T, size(rb.A))
    bd = PoissonKernels.to_device(be, b, T)
    md = PoissonKernels.to_device(be, mfield, T)
    _, iters, relres = PoissonKernels.masked_cg!(xd, bd, md;
                                                 rtol = device === :cpu ? rtol : 1e-7,
                                                 maxiter = maxiter)
    x = Float64.(PoissonKernels.to_host(xd))
    _ramses_field_masked_set!(h, :phi, levf, rb, x; lib = lib)
    return (phi = x, ghosts = gh, covered = cov, rb = rb, b = b,
            iters = iters, relres = Float64(relres), rho_tot = rho_tot, fact = fact)
end

"""
    run_ramses_gravity_amr_compare(; levc=5, half=4, amp=0.05, eps=1e-12)

The refined-level certification: a CUBOID fine region (flag1 written through
the bridge, `refine_fine!` consumes it), the host deposits both levels and
solves the coarse level; then the fine level is solved twice on identical
inputs — RAMSES's own `phi_fine_cg` at tolerance `eps` (the ORACLE) and the
KA `vcycle_solve!(dirichlet = true)` with the interpol_phi-reconstructed ghost
ring — and differenced.  Also returns the residual of the ORACLE solution
under OUR assembled system (ghosts + rhs): if that is ~ε, the system
replication itself is certified independently of the KA solver.
"""
function run_ramses_gravity_amr_compare(; levc::Integer = 5, half::Integer = 4,
                                        amp::Real = 0.05, eps::Real = 1e-12,
                                        lib::Symbol = :cpu)
    RamsesLib.available() || error("RAMSES library not found (set RAMSES_LIB to the bin64h build)")
    nc = 2^levc
    levf = levc + 1
    dir = mktempdir()
    nml = _ramses_gravity_namelist(level = levc)
    nml = replace(nml, "levelmax=$(levc)" => "levelmax=$(levf)")
    write(joinpath(dir, "gravity_amr.nml"), nml)
    return cd(dir) do
        h = RamsesLib.init("gravity_amr.nml"; lib = lib)
        RamsesLib.set_epsilon!(h, eps; lib = lib)
        # smooth density mode on the coarse gas
        ck, _ = RamsesLib.get_hydro_all(h, :uold, levc; lib = lib)
        noct = size(ck, 1)
        rho_in = Matrix{Float64}(undef, noct, 8)
        flag = Matrix{Float64}(undef, noct, 8)
        c0 = nc ÷ 2
        @inbounds for o in 1:noct, c in 1:8
            ix = 2 * ck[o, 1] + ((c - 1) & 1)
            iy = 2 * ck[o, 2] + ((c - 1) >> 1 & 1)
            iz = 2 * ck[o, 3] + ((c - 1) >> 2 & 1)
            x = (ix + 0.5) / nc; y = (iy + 0.5) / nc; z = (iz + 0.5) / nc
            rho_in[o, c] = 1.0 + amp * sin(2π * x) * sin(4π * y) * cos(2π * z)
            flag[o, c] = (abs(ix - c0 + 0.5) < half && abs(iy - c0 + 0.5) < half &&
                          abs(iz - c0 + 0.5) < half) ? 1.0 : 0.0
        end
        RamsesLib.set_hydro!(h, :uold, 1, levc, ck, rho_in; lib = lib)
        # the cuboid hierarchy: explicit flag map → the host's own refine
        RamsesLib.set_field!(h, :flag1, levc, ck, flag; lib = lib)
        RamsesLib.refine_fine!(h, levc; lib = lib)
        RamsesLib.level_noct(h, levf; lib = lib) > 0 ||
            error("run_ramses_gravity_amr_compare: refinement produced no fine octs")
        # host deposits + coarse solve (shared by both fine paths)
        RamsesLib.rho_fine!(h, levc, 0; lib = lib)
        RamsesLib.rho_fine!(h, levf, 0; lib = lib)
        RamsesLib.multigrid!(h, levc, 1; lib = lib)
        _, phic = ramses_grid_field(h, :phi, levc; lib = lib)
        _, rhoc = ramses_grid_field(h, :rho, levc; lib = lib)
        rho_tot = sum(rhoc) / length(rhoc)
        # ── ORACLE: RAMSES's own fine-level CG ────────────────────────────────
        RamsesLib.phi_fine_cg!(h, levf, 1; lib = lib)
        fb = _ramses_field_bbox(h, :phi, levf; lib = lib)
        rb = _ramses_field_bbox(h, :rho, levf; lib = lib)
        dxf = 1.0 / 2^levf
        nf = fb.nloc
        # ── assemble OUR system: ghosts + rhs ─────────────────────────────────
        sol = zeros(Float64, (nf .+ 2)...)
        _fine_dirichlet_ghosts!(sol, fb, phic; lib = lib)
        ghosts = copy(sol)
        d = nf .+ 2
        hfac = Float64(d[1] - 1) * Float64(d[2] - 1) * Float64(d[3] - 1)
        rhs = zeros(Float64, d...)
        @inbounds for k in 1:nf[3], j in 1:nf[2], i in 1:nf[1]
            rhs[i+1, j+1, k+1] = hfac * 4π * dxf^2 * (rb.A[i, j, k] - rho_tot)
        end
        # replication check: the oracle solution must satisfy OUR system to ~ε
        sol .= ghosts
        sol[2:end-1, 2:end-1, 2:end-1] .= fb.A
        defect = similar(sol)
        resid_oracle = PoissonKernels.mg_calc_defect!(defect, sol, rhs)
        # ── GUEST: the KA Dirichlet V-cycle from scratch ──────────────────────
        sol .= ghosts                                 # faces = BC, interior = 0
        PoissonKernels.vcycle_solve!(sol, rhs; rtol = 1e-12, maxcycles = 200,
                                     cycle = :W, dirichlet = true)
        scale = maximum(abs, fb.A .- sum(fb.A) / length(fb.A))
        dphi = maximum(abs, sol[2:end-1, 2:end-1, 2:end-1] .- fb.A) / scale
        return (dphi = dphi, resid_oracle = resid_oracle, phi_scale = scale,
                nf = nf, n_fine_octs = RamsesLib.level_noct(h, levf; lib = lib),
                handle = h, fb = fb, sol = sol,
                free = () -> RamsesLib.finalize(h; lib = lib))
    end
end

"""
    run_ramses_gravity_blob_compare(; levc=5, radius=0.18, amp=0.05, eps=1e-12)

The IRREGULAR-region certification: a SPHERICAL blob of refined coarse cells
(genuinely non-cuboid — asserted), host deposits + coarse solve, then the fine
level solved by (a) RAMSES's `phi_fine_cg` at `eps` — the ORACLE — and (b) the
masked CG guest (`ramses_ka_poisson_fine!`).  Returns the oracle's residual
under OUR masked system (certifies the irregular-domain replication
solver-free) and the φ parity.
"""
function run_ramses_gravity_blob_compare(; levc::Integer = 5, radius::Real = 0.18,
                                         amp::Real = 0.05, eps::Real = 1e-12,
                                         device::Symbol = :cpu, lib::Symbol = :cpu)
    RamsesLib.available() || error("RAMSES library not found (set RAMSES_LIB to the bin64h build)")
    nc = 2^levc
    levf = levc + 1
    dir = mktempdir()
    nml = _ramses_gravity_namelist(level = levc)
    nml = replace(nml, "levelmax=$(levc)" => "levelmax=$(levf)")
    write(joinpath(dir, "gravity_blob.nml"), nml)
    return cd(dir) do
        h = RamsesLib.init("gravity_blob.nml"; lib = lib)
        RamsesLib.set_epsilon!(h, eps; lib = lib)
        ck, _ = RamsesLib.get_hydro_all(h, :uold, levc; lib = lib)
        noct = size(ck, 1)
        rho_in = Matrix{Float64}(undef, noct, 8)
        flag = Matrix{Float64}(undef, noct, 8)
        @inbounds for o in 1:noct, c in 1:8
            ix = 2 * ck[o, 1] + ((c - 1) & 1)
            iy = 2 * ck[o, 2] + ((c - 1) >> 1 & 1)
            iz = 2 * ck[o, 3] + ((c - 1) >> 2 & 1)
            x = (ix + 0.5) / nc; y = (iy + 0.5) / nc; z = (iz + 0.5) / nc
            rho_in[o, c] = 1.0 + amp * sin(2π * x) * sin(4π * y) * cos(2π * z)
            flag[o, c] = ((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2 < radius^2) ? 1.0 : 0.0
        end
        RamsesLib.set_hydro!(h, :uold, 1, levc, ck, rho_in; lib = lib)
        RamsesLib.set_field!(h, :flag1, levc, ck, flag; lib = lib)
        RamsesLib.refine_fine!(h, levc; lib = lib)
        nfo = RamsesLib.level_noct(h, levf; lib = lib)
        nfo > 0 || error("blob refinement produced no fine octs")
        RamsesLib.rho_fine!(h, levc, 0; lib = lib)
        RamsesLib.rho_fine!(h, levf, 0; lib = lib)
        RamsesLib.multigrid!(h, levc, 1; lib = lib)
        # ── ORACLE fine solve, captured on the masked bbox ────────────────────
        RamsesLib.phi_fine_cg!(h, levf, 1; lib = lib)
        po = _ramses_field_bbox_masked(h, :phi, levf; lib = lib)
        is_cuboid = (8 * nfo == prod(po.nloc))
        # ── GUEST: the masked CG (overwrites the live phi) ────────────────────
        g = ramses_ka_poisson_fine!(h; levc = levc, levf = levf,
                                    device = device, lib = lib)
        # oracle residual under OUR masked system (φ_or with zeros outside the
        # mask — the ghost contributions live in g.b)
        phio = ifelse.(po.covered, po.A, 0.0)
        Ap = zeros(Float64, size(phio))
        _masked_apply!(Ap, phio, po.covered)
        resid_oracle = maximum(abs.(ifelse.(po.covered, Ap .- g.b, 0.0)))
        scale = maximum(abs, ifelse.(po.covered, po.A, 0.0))
        dphi = maximum(abs.(ifelse.(po.covered, g.phi .- po.A, 0.0))) / scale
        return (dphi = dphi, resid_oracle = resid_oracle / (g.fact * 1.0),
                phi_scale = scale, is_cuboid = is_cuboid, n_fine_octs = nfo,
                nloc = po.nloc, cg_iters = g.iters, cg_relres = g.relres,
                handle = h, free = () -> RamsesLib.finalize(h; lib = lib))
    end
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
