# patchgrid.jl — in-process topgrid decomposition into flat sibling patches.
#
# Mirrors how Enzo (under N MPI ranks) breaks its root grid into N chunks
# (CommunicationPartitionGrid), but entirely in ONE Julia process and in OUR
# hierarchy: a uniform periodic grid (ncell³) is tiled into np[1]·np[2]·np[3]
# sibling patches, each carrying its own `ng`-deep ghost shell.  Each patch runs
# its hydro (PPMKernels) + chemistry (ChemistryKernels) INDEPENDENTLY on the GPU;
# periodicity and patch-patch coupling live entirely in the ghost EXCHANGE.
#
# Top-grid GRAVITY is global (it couples all patches) and is handled separately in
# global_gravity.jl — assemble→FFT→scatter accel; here we only apply the kick.
#
# Key design points (see /home/tabel/.claude/plans/snoopy-marinating-beacon.md):
#   * Fields are FLAT device vectors of length nd³ (nd = pdim+2ng per axis), with a
#     dims tuple — the exact layout PPMKernels._hancock_sweep_axis! consumes.
#   * The halo is a reshaped-view broadcast (`@views dst .= src`) — backend-agnostic
#     (Array or CuArray), no scalar indexing, no bespoke kernel.
#   * np=1 along an axis ⇒ a patch is its own periodic partner ⇒ the SAME exchange
#     code reproduces periodic self-wrap.  So a 1×1×1 PatchGrid is the undecomposed
#     reference and shares every code path with the decomposed run.
#   * The driver owns the directional-sweep loop and refreshes the cross-patch halo
#     BEFORE each axis sweep — honoring the split integrator's per-sweep ghost
#     contract (PPMKernels.muscl_hancock_step_3d! refills ghosts before every sweep).

"One sibling patch: flat conserved fields (length nd³) + species, ghost shell, neighbors.
`A` is the conserved-field array type; `S` the species array type — `A` (f32 ρ·xᵢ) for the default
storage, or a `UInt16` device array when `packed_species` (then the stored value is the log2-packed mass
FRACTION Xᵢ=ρxᵢ/ρ; see ChemistryKernels log2_species.jl / FiniteVolumeGodunovKA colors.jl)."
struct Patch{A,S}
    idx     ::NTuple{3,Int}              # block coords, each in 0:np[d]-1
    D       ::A                          # ρ
    S1      ::A; S2::A; S3::A            # momenta ρv
    Tau     ::A                          # total energy density ρ·etot
    Ge      ::A                          # gas energy density ρ·eint (dual energy)
    species ::Vector{S}                  # passive species: ρ·xᵢ (default) or packed UInt16 fraction Xᵢ
    nbr     ::NTuple{3,NTuple{2,Int}}    # per axis: (lo_neighbor_lin, hi_neighbor_lin)
end

"The decomposition: a uniform periodic ncell³ grid tiled into np patches, GPU-resident."
mutable struct PatchGrid{A,S}
    backend                              # PPMKernels backend handle
    besym   ::Symbol                     # backend name (:cpu/:cuda/:metal) for ChemistryKernels
    T       ::DataType                   # element type (Float64 on cpu, Float32 on gpu)
    ng      ::Int                        # ghost depth
    np      ::NTuple{3,Int}              # tiling, e.g. (2,2,2) or (1,1,1) for the reference
    ncell   ::NTuple{3,Int}              # global active cells per axis (e.g. 128)
    pdim    ::NTuple{3,Int}              # per-patch active cells (ncell ÷ np)
    nd      ::NTuple{3,Int}              # per-patch total dims = pdim + 2ng
    dx      ::Float64                    # comoving cell width (a·L/ncell); set per step if cosmo
    gamma   ::Float64
    du      ::Float64; lu::Float64; tu::Float64   # chem cgs unit factors
    deut    ::Bool                       # H+D chemistry (3 species) vs H-only (2)
    packed  ::Bool                       # species stored as packed UInt16 fractions (vs f32 ρ·xᵢ)
    patches ::Vector{Patch{A,S}}
    fvgk    ::Any                        # nothing, or per-patch FiniteVolumeGodunovKA grids (solver=:fvgk)
end

_lin(ix, iy, iz, np) = ix + np[1]*iy + np[1]*np[2]*iz + 1

"All halo-exchanged fields of a patch, in a fixed order (conserved + dual energy + species)."
_allfields(p::Patch) = (p.D, p.S1, p.S2, p.S3, p.Tau, p.Ge, p.species...)

"""
    build_patchgrid(; ng, ncell, np, dx, gamma, nspecies, backend, T, du, lu, tu, deut)

Allocate `prod(np)` GPU-resident patches tiling a uniform periodic `ncell` grid,
with periodic neighbor links.  Fields start zeroed; call `scatter_global!` to load
an initial condition.
"""
function build_patchgrid(; ng::Int, ncell::NTuple{3,Int}, np::NTuple{3,Int},
                         dx::Real, gamma::Real, nspecies::Int, besym::Symbol, T::DataType,
                         du::Real=1.0, lu::Real=1.0, tu::Real=1.0, deut::Bool=true,
                         packed_species::Bool=false)
    backend = PPMKernels.backend(besym)
    all(ncell .% np .== 0) || error("build_patchgrid: ncell $ncell not divisible by np $np")
    pdim = ncell .÷ np
    nd   = pdim .+ 2ng
    n    = prod(nd)
    zeros_dev()  = PPMKernels.device_zeros(backend, T, (n,))
    # packed species: UInt16 fractions (UInt16(0) = the log2 floor ≈ 0); else f32 ρ·xᵢ
    spec_zeros() = packed_species ? PPMKernels.device_zeros(backend, UInt16, (n,)) : zeros_dev()
    A  = typeof(zeros_dev()); SA = typeof(spec_zeros())
    patches = Vector{Patch{A,SA}}(undef, prod(np))
    for iz in 0:np[3]-1, iy in 0:np[2]-1, ix in 0:np[1]-1
        lin = _lin(ix, iy, iz, np)
        # periodic lo/hi neighbor linear ids per axis
        nbr = ( ( _lin(mod(ix-1,np[1]), iy, iz, np), _lin(mod(ix+1,np[1]), iy, iz, np) ),
                ( _lin(ix, mod(iy-1,np[2]), iz, np), _lin(ix, mod(iy+1,np[2]), iz, np) ),
                ( _lin(ix, iy, mod(iz-1,np[3]), np), _lin(ix, iy, mod(iz+1,np[3]), np) ) )
        patches[lin] = Patch((ix,iy,iz), zeros_dev(), zeros_dev(), zeros_dev(), zeros_dev(),
                             zeros_dev(), zeros_dev(), SA[spec_zeros() for _ in 1:nspecies], nbr)
    end
    PatchGrid(backend, besym, T, ng, np, ncell, pdim, nd, Float64(dx), Float64(gamma),
              Float64(du), Float64(lu), Float64(tu), deut, packed_species, patches, nothing)
end

# Hydro-solver hook for `solver=:fvgk`: overridden by `MultiCodeFVGKExt` when
# `FiniteVolumeGodunovKA` is loaded. Advances every patch one full step of `dt` with the
# FiniteVolumeGodunovKA Godunov scheme (it owns the periodic boundary internally).
function _fvgk_patch_hydro! end

# ── interior-only chemistry: strided, in-place, f32 storage / f64 network ──────────────────────
# Processes ONLY the pdim³ interior cells: maps the linear interior thread → the strided nd³ index
# and reads/writes the f32 patch arrays directly — no gather/scatter, no f64 materialization.  Each
# cell promotes to Float64, runs the stiff network (ChemistryKernels.evolve_cell — the same per-cell
# map solve_chem_device! drives), narrows back, and applies the dual-energy Tau/Ge update.  Cosmology
# args match solve_chem_device!'s defaults (hubble=71, Ωm=.27, ΩΛ=.73, fh=.76; no metals/expansion).
@kernel function _chem_interior_k!(D, Ge, Tau, sp1, sp2, sp3,
                                   du, vu2, tu, dt, z, deut, rtab, ctab,
                                   nd1::Int, nd2::Int, ng::Int, pd1::Int, pd2::Int)
    t = @index(Global)
    @inbounds begin
        TT = eltype(D)
        t0 = t - 1
        ix = t0 % pd1; q = t0 ÷ pd1
        iy = q % pd2;  iz = q ÷ pd2
        idx = (ng + ix) + nd1 * (ng + iy) + nd1 * nd2 * (ng + iz) + 1
        d  = Float64(D[idx]); e0 = Float64(Ge[idx]) / d
        h1 = Float64(sp1[idx]); h2 = Float64(sp2[idx]); hd = deut ? Float64(sp3[idx]) : 0.0
        en, hn, h2n, hdn, _ = ChemistryKernels.evolve_cell(d*du, e0*vu2, h1*du, h2*du,
                                  deut ? hd*du : 0.0, dt*tu, z;
                                  hubble=71.0, Om=0.27, OL=0.73, fh=0.76, deuterium=deut,
                                  rate_tables=rtab, cool_tables=ctab)
        enew = en / vu2
        Ge[idx]  = TT(d * enew)
        Tau[idx] = Tau[idx] + TT(d * (enew - e0))
        sp1[idx] = TT(hn / du); sp2[idx] = TT(h2n / du)
        deut && (sp3[idx] = TT(hdn / du))
    end
end

"Run the stiff chemistry on patch `p`'s interior only (no ghost work, no gather/scatter copies)."
function _chem_interior!(pg::PatchGrid, p::Patch, a_value, dt, du, lu, tu, rate_tables, cool_tables)
    deut = pg.deut && length(p.species) >= 3
    sp3  = deut ? p.species[3] : p.species[1]            # placeholder ref when no deuterium (unread)
    nd1, nd2, _ = pg.nd; pd1, pd2, pd3 = pg.pdim
    _chem_interior_k!(pg.backend)(p.D, p.Ge, p.Tau, p.species[1], p.species[2], sp3,
        Float64(du), Float64((lu/tu)^2), Float64(tu), Float64(dt), Float64(1.0/a_value - 1.0),
        deut, rate_tables, cool_tables, nd1, nd2, pg.ng, pd1, pd2; ndrange = pd1*pd2*pd3)
    return nothing
end

# ── periodic ghost exchange along one axis (the patch-patch coupling) ──────────
# For np=2 (or 1) the lo and hi neighbor along an axis are the same partner, and
# the rule is symmetric: a patch's lo ghost ← partner's last ng INTERIOR planes,
# its hi ghost ← partner's first ng interior planes.  Only interiors are READ and
# only ghosts are WRITTEN, so the per-patch loop is order-independent (no aliasing).
@inline _r3(f, nd) = reshape(f, nd[1], nd[2], nd[3])

function exchange_ghosts_axis!(pg::PatchGrid, axis::Int)
    ng = pg.ng; na = pg.nd[axis]
    lo_src = (na-2ng+1):(na-ng)       # partner's last ng interior planes  → my lo ghost
    hi_src = (ng+1):(2ng)             # partner's first ng interior planes → my hi ghost
    lo_dst = 1:ng
    hi_dst = (na-ng+1):na
    for p in pg.patches
        loN = pg.patches[p.nbr[axis][1]]
        hiN = pg.patches[p.nbr[axis][2]]
        for (fp, flo, fhi) in zip(_allfields(p), _allfields(loN), _allfields(hiN))
            d = _r3(fp, pg.nd); s_lo = _r3(flo, pg.nd); s_hi = _r3(fhi, pg.nd)
            if axis == 1
                @views d[lo_dst, :, :] .= s_lo[lo_src, :, :]
                @views d[hi_dst, :, :] .= s_hi[hi_src, :, :]
            elseif axis == 2
                @views d[:, lo_dst, :] .= s_lo[:, lo_src, :]
                @views d[:, hi_dst, :] .= s_hi[:, hi_src, :]
            else
                @views d[:, :, lo_dst] .= s_lo[:, :, lo_src]
                @views d[:, :, hi_dst] .= s_hi[:, :, hi_src]
            end
        end
    end
    return nothing
end

exchange_ghosts!(pg::PatchGrid) = (for a in 1:3; exchange_ghosts_axis!(pg, a); end; nothing)

# ── scatter a global IC into patches / gather patches back to a global array ───
_interior(pg) = (pg.ng+1):(pg.ng+pg.pdim[1]), (pg.ng+1):(pg.ng+pg.pdim[2]), (pg.ng+1):(pg.ng+pg.pdim[3])

"Octant ranges of patch `p` in the global ncell grid (1-based)."
function _octant(pg::PatchGrid, p::Patch)
    o = p.idx .* pg.pdim
    (o[1]+1:o[1]+pg.pdim[1], o[2]+1:o[2]+pg.pdim[2], o[3]+1:o[3]+pg.pdim[3])
end

"""
    scatter_global!(pg, gfields)

Load a global IC into the patches.  `gfields` is a NamedTuple of host `ncell³`
arrays with keys `D,S1,S2,S3,Tau,Ge` and `species` (a `Vector` of `ncell³` host
arrays).  Fills each patch interior from its octant, uploads to the device, and
primes the ghost shell via a full exchange.
"""
function scatter_global!(pg::PatchGrid, gfields)
    li, lj, lk = _interior(pg)
    for p in pg.patches
        gi, gj, gk = _octant(pg, p)
        hostbuf = zeros(pg.T, pg.nd)
        function load!(dst_dev, gsrc)
            fill!(hostbuf, zero(pg.T))
            @views hostbuf[li, lj, lk] .= pg.T.(gsrc[gi, gj, gk])
            copyto!(dst_dev, vec(hostbuf))
        end
        load!(p.D, gfields.D); load!(p.S1, gfields.S1); load!(p.S2, gfields.S2)
        load!(p.S3, gfields.S3); load!(p.Tau, gfields.Tau); load!(p.Ge, gfields.Ge)
        if pg.packed
            # IC species are ρ·xᵢ (host); store as the packed UInt16 mass fraction Xᵢ = ρxᵢ/ρ
            ρoct = gfields.D[gi, gj, gk]; ubuf = fill(UInt16(0), pg.nd)
            for (sd, gs) in zip(p.species, gfields.species)
                fill!(ubuf, UInt16(0))
                @views ubuf[li, lj, lk] .= ChemistryKernels.encode_log2sp.(
                    Float32.(gs[gi, gj, gk] ./ max.(ρoct, eps())))
                copyto!(sd, vec(ubuf))
            end
        else
            for (sd, gs) in zip(p.species, gfields.species); load!(sd, gs); end
        end
    end
    exchange_ghosts!(pg)
    return pg
end

"""
    gather_global(pg) -> NamedTuple

Reassemble the patch interiors into global `ncell³` host arrays (ghosts dropped),
keys `D,S1,S2,S3,Tau,Ge,species`.  Used for I/O and validation.
"""
function gather_global(pg::PatchGrid)
    li, lj, lk = _interior(pg)
    nsp = length(pg.patches[1].species)
    g = (D=zeros(pg.T, pg.ncell), S1=zeros(pg.T, pg.ncell), S2=zeros(pg.T, pg.ncell),
         S3=zeros(pg.T, pg.ncell), Tau=zeros(pg.T, pg.ncell), Ge=zeros(pg.T, pg.ncell),
         species=[zeros(pg.T, pg.ncell) for _ in 1:nsp])
    for p in pg.patches
        gi, gj, gk = _octant(pg, p)
        store!(gdst, fdev) = (h = _r3(PPMKernels.to_host(fdev), pg.nd);
                              @views gdst[gi, gj, gk] .= h[li, lj, lk])
        store!(g.D, p.D); store!(g.S1, p.S1); store!(g.S2, p.S2)
        store!(g.S3, p.S3); store!(g.Tau, p.Tau); store!(g.Ge, p.Ge)
        if pg.packed
            # stored species are packed UInt16 fractions Xᵢ; return ρ·xᵢ = unpack(Xᵢ)·ρ (host)
            Dh = _r3(PPMKernels.to_host(p.D), pg.nd)
            for s in 1:nsp
                Xh = ChemistryKernels.decode_log2sp.(pg.T, _r3(PPMKernels.to_host(p.species[s]), pg.nd))
                @views g.species[s][gi, gj, gk] .= pg.T.(Xh[li, lj, lk] .* Dh[li, lj, lk])
            end
        else
            for s in 1:nsp; store!(g.species[s], p.species[s]); end
        end
    end
    return g
end

# ── gravity kick (KE-consistent), per patch — mirrors cicass _grav_kick! ───────
@inline function _grav_kick!(S1, S2, S3, Tau, D, ax, ay, az, c::T) where {T}
    dS1 = D .* ax .* c; dS2 = D .* ay .* c; dS3 = D .* az .* c
    Tau .+= ((S1 .* dS1 .+ S2 .* dS2 .+ S3 .* dS3) .+
             T(0.5) .* (dS1 .^ 2 .+ dS2 .^ 2 .+ dS3 .^ 2)) ./ D
    S1 .+= dS1; S2 .+= dS2; S3 .+= dS3
    return nothing
end

# ── per-cycle hydro + chemistry over all patches ──────────────────────────────
"""
    patch_step!(pg, dt; a_value, order=(1,2,3), accel=nothing, chem=true)

Advance every patch one hydro step on the GPU with the cross-patch halo refreshed
before each directional sweep, then (optionally) one chemistry substep.  `accel`,
if given, is a `Vector` (per patch) of `(ax,ay,az)` device arrays (length nd³);
the gravity ½-kick is applied before and after the hydro (KDK), mirroring
`hydro_localppm!`.  `a_value` is the expansion factor for chemistry units.
"""
function patch_step!(pg::PatchGrid, dt::Real; a_value::Real, order=(1,2,3),
                     accel=nothing, chem::Bool=true, solver::Symbol=:ppm,
                     du::Real=pg.du, lu::Real=pg.lu, tu::Real=pg.tu,
                     do_hydro::Bool=true, do_chem::Bool=true, chem_backend::Symbol=pg.besym,
                     rate_tables=nothing, cool_tables=nothing)
    # `do_hydro`/`do_chem` split the step into its two GPU phases.  Chemistry does NOT
    # change the density, so the next step's top-grid gravity can be solved on the host
    # from the post-hydro density WHILE this step's chemistry runs on the GPU — a
    # lag-FREE CPU/GPU overlap (see examples/patch_cicass.jl CIC_OVERLAP).  When the
    # hydro phase runs without chemistry we synchronize before releasing the scratch pool.
    be = pg.backend; T = pg.T; chalf = T(0.5*dt)
    PPMKernels.with_pool() do
      if do_hydro
        # ½ gravity kick (pre) — `accel` is per-patch POTENTIAL blocks; g = −∇φ is
        # central-differenced inline (no stored accel field), KE-consistent KDK.
        if accel !== nothing
            for (p, φb) in zip(pg.patches, accel)
                PoissonKernels.grav_kick_from_potential!(φb, p.D, p.S1, p.S2, p.S3, p.Tau;
                    dims=pg.nd, ng=pg.ng, dx=pg.dx, halfdt=0.5*dt)
            end
        end
        # dimensionally-split hydro, halo refreshed before each axis sweep.
        # COMPUTE-WITH-OVERLAP: the PPM solver's flux at a pencil's active-EDGE interface
        # is computed with a boundary stencil that differs from an interior interface, so a
        # plain ng-halo is NOT bit-exact at patch boundaries (the edge cell gets a slightly
        # wrong flux — verified independent of ng).  We exchange the full ng-deep halo but run
        # the sweep with ngs = ng-1, so each patch UPDATES one cell into the halo (the
        # `overlap`); those overlap cells carry the boundary error and are discarded/refilled
        # by the next exchange, while every KEPT interior cell is computed with interior-only
        # interfaces ⇒ bit-identical to the undecomposed run.  (overlap=1 verified sufficient.)
        if solver === :fvgk
            # FiniteVolumeGodunovKA (unsplit single-grid Godunov). Assembles the patch interiors
            # into one global periodic grid, steps once, disperses back — decomposition-invariant
            # for any np (bit-identical to the undecomposed run), then re-derives Ge. No per-axis
            # cross-patch exchange (FVGK is unsplit). See MultiCodeFVGKExt.
            _fvgk_patch_hydro!(pg, dt)
        else
            pg.packed && error("patch_step!: packed_species (UInt16) storage requires solver=:fvgk — " *
                               "the PPM sweep advects f32 colours; rebuild without packed_species for :ppm")
            ngs = pg.ng - 1
            ngs >= 1 || error("patch_step!: need ng ≥ 2 for the compute-with-overlap halo (got ng=$(pg.ng))")
            for axis in order
                PPMKernels._pool_reset!()
                exchange_ghosts_axis!(pg, axis)
                for p in pg.patches
                    PPMKernels._hancock_sweep_axis!(p.D, p.S1, p.S2, p.S3, p.Tau, pg.nd, ngs, axis;
                        dt=dt, gamma=pg.gamma, theta=1.5, dx=pg.dx, small_rho=1e-10,
                        recon=:ppm_local, predictor=:trace, ge=p.Ge,
                        colours=isempty(p.species) ? nothing : Tuple(p.species),
                        riemann=:hll, face_periodic=false)
                end
            end
            # DEF reset (gas energy ↔ total energy) once per step, per patch
            for p in pg.patches
                PPMKernels.dual_energy_sync!(p.D, p.S1, p.S2, p.S3, p.Tau, p.Ge; gamma=pg.gamma)
            end
        end
        # ½ gravity kick (post)
        if accel !== nothing
            for (p, φb) in zip(pg.patches, accel)
                PoissonKernels.grav_kick_from_potential!(φb, p.D, p.S1, p.S2, p.S3, p.Tau;
                    dims=pg.nd, ng=pg.ng, dx=pg.dx, halfdt=0.5*dt)
            end
        end
        # hydro phase without chem: synchronize so the scratch pool isn't released
        # while the async kernels still reference it (chem's own sync covers the joint case)
        do_chem || PPMKernels.KA.synchronize(be)
      end # do_hydro
        # chemistry: per patch, per cell — INTERIOR ONLY.  The old path ran the stiff network over
        # the full nd³ array, but the ghost results are immediately overwritten by the next halo
        # exchange (pure waste; (nd/pdim)³ ≈ +42% at np=2/ng=4, worse for smaller patches).  We
        # GATHER each patch interior into contiguous pdim³ buffers, solve, and SCATTER back — the
        # interior result is bit-identical, the ghost work is gone.  ALWAYS solved in Float64 (the
        # stiff network NaNs in f32), promoting on read / narrowing on write.
        if do_chem && chem && !isempty(pg.patches[1].species)
            haveHD = pg.deut && length(pg.patches[1].species) >= 3
            oncpu = chem_backend === :cpu && pg.besym !== :cpu     # run chem on the HOST
            li, lj, lk = _interior(pg)
            gi(f)   = vec(_r3(f, pg.nd)[li, lj, lk])                          # gather interior → contiguous pdim³
            si!(f, v) = (@views _r3(f, pg.nd)[li, lj, lk] .= _r3(v, pg.pdim); nothing)  # scatter back into interior
            sa!(f, v) = (@views _r3(f, pg.nd)[li, lj, lk] .+= _r3(v, pg.pdim); nothing) # scatter-ADD (no read-gather)
            for p in pg.patches
                if pg.packed
                    Dint = gi(p.D); Geint = gi(p.Ge); r64 = Float64.(Dint)
                    e64  = Float64.(Geint) ./ r64; e0 = copy(e64)
                    s1 = gi(p.species[1]); s2 = gi(p.species[2]); hdc = haveHD ? gi(p.species[3]) : nothing
                    if oncpu
                        sp1 = Array(s1); sp2 = Array(s2); sp3 = haveHD ? Array(hdc) : nothing
                        rh = Float64.(Array(Dint)); eh = Float64.(Array(Geint)) ./ rh; eh0 = copy(eh)
                        ChemistryKernels.solve_chem_device_u16!(rh, eh, sp1, sp2, sp3;
                            a_value=a_value, dt=dt, density_units=du, length_units=lu,
                            time_units=tu, deuterium=pg.deut, backend=:cpu, precision=Float64,
                            rate_tables=rate_tables, cool_tables=cool_tables)
                        sa!(p.Tau, Dint .* PPMKernels.to_device(be, T.(eh .- eh0), T))
                        si!(p.Ge,  Dint .* PPMKernels.to_device(be, T.(eh), T))
                        si!(p.species[1], PPMKernels.to_device(be, sp1, UInt16))
                        si!(p.species[2], PPMKernels.to_device(be, sp2, UInt16))
                        haveHD && si!(p.species[3], PPMKernels.to_device(be, sp3, UInt16))
                    else
                        ChemistryKernels.solve_chem_device_u16!(r64, e64, s1, s2, hdc;
                            a_value=a_value, dt=dt, density_units=du, length_units=lu,
                            time_units=tu, deuterium=pg.deut, backend=pg.besym, precision=Float64,
                            rate_tables=rate_tables, cool_tables=cool_tables)
                        sa!(p.Tau, Dint .* T.(e64 .- e0))
                        si!(p.Ge,  Dint .* T.(e64))
                        si!(p.species[1], s1); si!(p.species[2], s2)
                        haveHD && si!(p.species[3], hdc)
                    end
                elseif oncpu
                    # GPU/CPU role flip: download the interior, solve the stiff network on the HOST
                    # (no warp divergence), then upload the cooled energy + species back.
                    Dint = gi(p.D); Geint = gi(p.Ge)
                    rh = Float64.(Array(Dint)); eh = Float64.(Array(Geint)) ./ rh; e0h = copy(eh)
                    h1 = Float64.(Array(gi(p.species[1]))); h2 = Float64.(Array(gi(p.species[2])))
                    hd = haveHD ? Float64.(Array(gi(p.species[3]))) : nothing
                    ChemistryKernels.solve_chem_device!(rh, eh, h1, h2, hd;
                        a_value=a_value, dt=dt, density_units=du, length_units=lu,
                        time_units=tu, deuterium=pg.deut, backend=:cpu, precision=Float64,
                        rate_tables=rate_tables, cool_tables=cool_tables)
                    sa!(p.Tau, Dint .* PPMKernels.to_device(be, T.(eh .- e0h), T))
                    si!(p.Ge,  Dint .* PPMKernels.to_device(be, T.(eh), T))
                    si!(p.species[1], PPMKernels.to_device(be, T.(h1), T))
                    si!(p.species[2], PPMKernels.to_device(be, T.(h2), T))
                    haveHD && si!(p.species[3], PPMKernels.to_device(be, T.(hd), T))
                else
                    # f32 species, device chem: strided in-place — no gather/scatter, no f64 materialization
                    _chem_interior!(pg, p, a_value, dt, du, lu, tu, rate_tables, cool_tables)
                end
            end
        end
    end
    return nothing
end

# total active mass Σ D·dV over all patches (diagnostic / conservation check)
function total_mass(pg::PatchGrid)
    li, lj, lk = _interior(pg); s = 0.0; dV = pg.dx^3
    for p in pg.patches
        h = _r3(PPMKernels.to_host(p.D), pg.nd)
        @views s += sum(Float64.(h[li, lj, lk]))
    end
    return s * dV
end
