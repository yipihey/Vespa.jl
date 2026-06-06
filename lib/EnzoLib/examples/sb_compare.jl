# Santa Barbara cluster: enzo-f32 CPU reference  vs  EnzoNG (Julia PPM + FFT gravity).
#
# Drives the SAME SB cosmology problem through the SAME Julia EvolveLevel
# (`evolve_level!`) twice, from identical initial conditions, swapping ONLY the
# per-grid physics:
#   REFERENCE  hydro=:enzo  gravity=:enzo      — Enzo's own certified PPM + FFT gravity
#   ENZONG     hydro=:julia gravity=:julia     — PPMKernels PPM + PoissonKernels FFT
#                                                gravity, on BACKEND (cpu|metal), f32
# Then it compares the two for
#   PARITY       relative-L2 and Linf of every root field (ρ, v1..3, TE, GE),
#   PERFORMANCE  per-slot median/total wall time (from EnzoLib's SlotProbe) and the
#                ENZONG-vs-reference speedup of the hydro and gravity slots.
#
# Same orchestration, same AMR, same particle push / comoving expansion (all Enzo);
# the diff isolates exactly the swapped kernels — which is the comparison we want.
#
# The faithful reference is enzo built p4_b4 (32-bit); select its bridge with
#   ENZOMODULES_GRID_LIB=…/libenzomodules_grid_f32.dylib
# Run (env BACKEND=cpu|metal, arg = #cycles):
#   ENZOMODULES_GRID_LIB=<f32.dylib> BACKEND=metal \
#     <julia> --project=lib/PPMKernels/test lib/EnzoLib/examples/sb_compare.jl [cycles]

using EnzoLib, PPMKernels, PoissonKernels, Printf
try; @eval using Metal; catch; end

const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster"
const NG = 4
const GAMMA = 5/3
const OMEGA_B = 0.1; const OMEGA_CDM = 0.9
const iD, iV1, iV2, iV3, iTE, iGE = 0, 1, 2, 3, 4, 5
const FIELDS = [(iD,"ρ"), (iV1,"v1"), (iV2,"v2"), (iV3,"v3"), (iTE,"TE"), (iGE,"GE")]
const BE = Symbol(get(ENV, "BACKEND", "cpu"))
const T  = Float32                                  # EnzoNG kernels are f32

active_of(flat, gd, N) = Array(reshape(Float64.(flat), gd[1], gd[2], gd[3])[NG+1:NG+N, NG+1:NG+N, NG+1:NG+N])

function pad_periodic(φ::Array{Float64,3})
    N = size(φ, 1); M = N + 2NG; full = Array{Float64,3}(undef, M, M, M)
    @inbounds for k in 1:M, j in 1:M, i in 1:M
        full[i, j, k] = φ[mod(i-NG-1, N)+1, mod(j-NG-1, N)+1, mod(k-NG-1, N)+1]
    end
    full
end
place_active(act::Array{Float64,3}, gd) = begin
    full = zeros(Float64, gd[1], gd[2], gd[3]); N = size(act, 1)
    full[NG+1:NG+N, NG+1:NG+N, NG+1:NG+N] .= act; vec(full)
end
cic!(rho, pos, N) = begin
    @inbounds for p in 1:size(pos,1)
        gx=mod(pos[p,1],1.0)*N; gy=mod(pos[p,2],1.0)*N; gz=mod(pos[p,3],1.0)*N
        i=floor(Int,gx);fx=gx-i;j=floor(Int,gy);fy=gy-j;k=floor(Int,gz);fz=gz-k
        i0=mod(i,N)+1;i1=mod(i+1,N)+1;j0=mod(j,N)+1;j1=mod(j+1,N)+1;k0=mod(k,N)+1;k1=mod(k+1,N)+1
        rho[i0,j0,k0]+=(1-fx)*(1-fy)*(1-fz);rho[i1,j0,k0]+=fx*(1-fy)*(1-fz)
        rho[i0,j1,k0]+=(1-fx)*fy*(1-fz);rho[i1,j1,k0]+=fx*fy*(1-fz)
        rho[i0,j0,k1]+=(1-fx)*(1-fy)*fz;rho[i1,j0,k1]+=fx*(1-fy)*fz
        rho[i0,j1,k1]+=(1-fx)*fy*fz;rho[i1,j1,k1]+=fx*fy*fz
    end; rho
end

# ── EnzoNG :julia slots (root grid; subgrids deferred to the composite-gravity task) ──
const _step = Ref(0)
function ng_gravity!(h, level, dt)
    level == 0 || return nothing
    bep = PoissonKernels.backend(BE)
    g = EnzoLib.problem_grid_index_on_level(h, 0, 0)
    gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, g))); N = gd[1] - 2NG
    gas = active_of(EnzoLib.read_density(h; grid=g), gd, N)
    dm  = active_of(EnzoLib.deposit_particle_density(h; grid=g), gd, N)  # Enzo CIC in C++ (no 2M-particle marshalling)
    dm .*= OMEGA_CDM/(sum(dm)/length(dm))
    gas .*= OMEGA_B/(sum(gas)/length(gas))
    δ = gas .+ dm; δ ./= (sum(δ)/length(δ)); δ .-= 1.0
    φ = Array{T,3}(undef, N, N, N)                                # host: FFT stays host-resident
    PoissonKernels.fft_poisson_root!(φ, Array{T,3}(δ); G=1.0, a=1.0, boxsize=1.0)
    φf = PoissonKernels.to_device(bep, pad_periodic(Float64.(φ)), T)
    a1 = PoissonKernels.device_zeros(bep, T, (N,N,N)); a2 = similar(a1); a3 = similar(a1)
    PoissonKernels.comp_accel!(a1, a2, a3, φf; iflag=1, start=(NG,NG,NG), del=(1.0/N,1.0/N,1.0/N))
    EnzoLib.problem_set_acceleration(h, 0, place_active(Float64.(PoissonKernels.to_host(a1)), gd); grid=g)
    EnzoLib.problem_set_acceleration(h, 1, place_active(Float64.(PoissonKernels.to_host(a2)), gd); grid=g)
    EnzoLib.problem_set_acceleration(h, 2, place_active(Float64.(PoissonKernels.to_host(a3)), gd); grid=g)
    return nothing
end
function ng_hydro!(h, level, dt)
    bep = PPMKernels.backend(BE)
    n = EnzoLib.session_num_grids_on_level(h, level)
    order = isodd(_step[]) ? (3,2,1) : (1,2,3); _step[] += 1
    for i in 0:n-1
        g = EnzoLib.problem_grid_index_on_level(h, level, i)
        gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, g)))
        f(fi) = PPMKernels.to_device(bep, EnzoLib.problem_get_field(h, fi, g), T)
        d, e, ge = f(iD), f(iTE), f(iGE); vx, vy, vz = f(iV1), f(iV2), f(iV3)
        gx = PPMKernels.to_device(bep, EnzoLib.problem_get_acceleration(h,0,g), T)
        gy = PPMKernels.to_device(bep, EnzoLib.problem_get_acceleration(h,1,g), T)
        gz = PPMKernels.to_device(bep, EnzoLib.problem_get_acceleration(h,2,g), T)
        PPMKernels.ppm_step_3d!(d, e, ge, vx, vy, vz, gx, gy, gz, gd, NG;
                                dt=dt, gamma=GAMMA, order=order, gravity=1, idual=1)
        wr(fi, a) = EnzoLib.problem_set_field(h, fi, Float64.(PPMKernels.to_host(a)); grid=g)
        wr(iD, d); wr(iTE, e); wr(iGE, ge); wr(iV1, vx); wr(iV2, vy); wr(iV3, vz)
    end
    return nothing
end

# ── one full run of `maxcyc` cycles; returns (root fields, per-cycle wall ms, probe) ──
function run_config(label, eng_builder, maxcyc)
    pf = joinpath(SB, "SB_compare.enzo")
    write(pf, replace(read(joinpath(SB, "SantaBarbaraCluster.enzo"), String),
                      r"GreensFunctionMaxNumber.*" => "GreensFunctionMaxNumber   = 30\nNumberOfGhostZones        = 4"))
    _step[] = 0
    h = EnzoLib.session_init(pf); h == C_NULL && error("session_init failed ($label)")
    probe = EnzoLib.SlotProbe()
    percyc = Float64[]
    try
        eng = eng_builder(probe)
        EnzoLib.session_rebuild(h, 0)
        for cyc in 1:maxcyc
            t0 = time_ns()
            EnzoLib.evolve_level!(h, 0, 0.0; engine=eng, regrid=true)
            push!(percyc, (time_ns() - t0) / 1e6)
            EnzoLib.session_rebuild(h, 0)
        end
        gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, 0))); N = gd[1] - 2NG
        fields = Dict(name => active_of(EnzoLib.problem_get_field(h, fi, 0), gd, N) for (fi,name) in FIELDS)
        t = EnzoLib.session_time(h)
        @printf("  %-8s ran %d cycles → t=%.5f, median %.1f ms/cyc\n", label, maxcyc, t,
                sort(percyc)[(length(percyc)+1)÷2])
        return (fields=fields, t=t, percyc=percyc, probe=EnzoLib.probe_summary(probe))
    finally
        EnzoLib.free_problem(h)
    end
end

relL2(a, b) = sqrt(sum(abs2, a .- b) / max(sum(abs2, b), eps()))
Linf(a, b)  = maximum(abs, a .- b)

function main()
    maxcyc = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6
    EnzoLib.grid_available() || error("grid dylib not built")
    @printf("\nSB comparison — REFERENCE (enzo PPM+FFT) vs ENZONG (julia PPM+FFT, backend=%s, f32), %d cycles\n",
            BE, maxcyc)
    @printf("grid bridge: %s\n\n", basename(EnzoLib.grid_libpath()))

    refb(p) = EnzoLib.EngineConfig(; hydro=:enzo, gravity=:enzo, comoving_expansion=:enzo, probe=p)
    ngb(p)  = EnzoLib.EngineConfig(; hydro=:julia, gravity=:julia, comoving_expansion=:enzo, reflux=false,
                                   probe=p,
                                   hooks=Dict{Symbol,Function}(:hydro=>ng_hydro!, :gravity=>ng_gravity!))

    cd(SB) do
        println("running configs (each from identical SB ICs):")
        ref = run_config("enzo-ref", refb, maxcyc)
        ng  = run_config("enzong", ngb, maxcyc)

        println("\n── PARITY (ENZONG vs enzo-ref, root active region) ──")
        @printf("  time:  ref t=%.6f  enzong t=%.6f  (Δt/t=%.2e)\n", ref.t, ng.t, abs(ng.t-ref.t)/ref.t)
        @printf("  %-4s %-12s %-12s %-12s\n", "fld", "relL2", "Linf", "ref‖·‖∞")
        for (_,name) in FIELDS
            a = ng.fields[name]; b = ref.fields[name]
            @printf("  %-4s %-12.3e %-12.3e %-12.3e\n", name, relL2(a,b), Linf(a,b), maximum(abs,b))
        end

        println("\n── PERFORMANCE (per-slot wall time) ──")
        @printf("  %-10s %-14s %-14s %-14s\n", "slot", "ref median ms", "enzong med ms", "speedup(ref/ng)")
        for slot in (:gravity, :hydro)
            rs = get(ref.probe, slot, nothing); ns = get(ng.probe, slot, nothing)
            rm = rs === nothing ? NaN : rs.median_ns/1e6
            nm = ns === nothing ? NaN : ns.median_ns/1e6
            @printf("  %-10s %-14.2f %-14.2f %-14.2f\n", slot, rm, nm, rm/nm)
        end
        @printf("  %-10s %-14.1f %-14.1f %-14.2f\n", "TOTAL/cyc",
                sort(ref.percyc)[(end+1)÷2], sort(ng.percyc)[(end+1)÷2],
                sort(ref.percyc)[(end+1)÷2] / sort(ng.percyc)[(end+1)÷2])
    end
end

main()
