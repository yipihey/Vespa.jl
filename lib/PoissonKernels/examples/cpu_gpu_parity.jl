# Targeted CPU↔GPU parity: every gravity kernel must give IDENTICAL results for
# IDENTICAL inputs (CPU f64 is the oracle; Metal f32 must agree to f32 round-off,
# and CPU-f32 vs Metal-f32 must agree to ~1e-6 — same algorithm, same precision).
# No physics/Enzo here — pure kernel determinism. Run:
#   <jl> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/cpu_gpu_parity.jl
using PoissonKernels, Printf
try; @eval using Metal; catch; end
const PK = PoissonKernels
_flat(x) = x isa Tuple ? vcat((vec(Float64.(y)) for y in x)...) : vec(Float64.(x))
rel(a,b) = (fa=_flat(a); fb=_flat(b); d=sqrt(sum(abs2,fa.-fb)); n=sqrt(sum(abs2,fb)); n>0 ? d/n : d)
hdr(s) = (println(); println("== $s =="))

function run_one(name, T, mk, call)
    PK.has_backend(:metal) || return println("  [metal] unavailable")
    becpu = PK.backend(:cpu); bemtl = PK.backend(:metal)
    host = mk()                                  # identical host inputs (Float64 master)
    # CPU at T, Metal at T — identical algorithm + precision ⇒ must match to round-off
    rc = call(becpu, T, host); rm = call(bemtl, T, host)
    @printf("  %-26s [%s] CPU↔Metal relL2 = %.2e\n", name, T, rel(rc, rm))
end

# ---- comp_accel ----
hdr("comp_accel  (g = -∇φ)")
for T in (Float32,)
    run_one("comp_accel", T, ()->rand(34,34,34),
        (be,T,h)->begin
            φ = PK.to_device(be, T.(h), T)
            d1=PK.device_zeros(be,T,(32,32,32));d2=similar(d1);d3=similar(d1)
            PK.comp_accel!(d1,d2,d3,φ; iflag=1, start=(1,1,1), del=(T(0.01),T(0.01),T(0.01)))
            (Array(d1),Array(d2),Array(d3))
        end)
end

# ---- mg_relax / defect / restrict / prolong (single grid) ----
hdr("multigrid kernels (single grid 34³→…)")
let d=34
    run_one("mg_relax (2 sweeps)", Float32, ()->(rand(d,d,d),rand(d,d,d)),
        (be,T,h)->begin
            sol=PK.to_device(be,T.(h[1]),T); rhs=PK.to_device(be,T.(h[2]),T)
            PK.mg_relax!(sol,rhs); PK.mg_relax!(sol,rhs); Array(sol)
        end)
    run_one("mg_calc_defect", Float32, ()->(rand(d,d,d),rand(d,d,d),rand(d,d,d)),
        (be,T,h)->begin
            df=PK.to_device(be,T.(h[1]),T); sol=PK.to_device(be,T.(h[2]),T); rhs=PK.to_device(be,T.(h[3]),T)
            PK.mg_calc_defect!(df,sol,rhs); Array(df)
        end)
    run_one("mg_restrict 34³→18³", Float32, ()->rand(34,34,34),
        (be,T,h)->begin
            src=PK.to_device(be,T.(h),T); dst=PK.device_zeros(be,T,(18,18,18))
            PK.mg_restrict!(dst,src); Array(dst)
        end)
    run_one("mg_prolong 18³→34³", Float32, ()->rand(18,18,18),
        (be,T,h)->begin
            src=PK.to_device(be,T.(h),T); dst=PK.device_zeros(be,T,(34,34,34))
            PK.mg_prolong!(dst,src); Array(dst)
        end)
end

# ---- fft_poisson_root_gpu! (whole device Poisson solve) ----
hdr("fft_poisson_root_gpu!  (64³)")
run_one("fft_poisson_root_gpu", Float32, ()->(r=rand(64,64,64); r.-=sum(r)/length(r); r),
    (be,T,h)->begin
        ρ=PK.to_device(be,T.(h),T); φ=PK.device_zeros(be,T,(64,64,64))
        PK.fft_poisson_root_gpu!(φ,ρ; G=1.0,a=1.0,boxsize=1.0); Array(φ)
    end)

# ---- vcycle_solve! (full V-cycle driver) ----
hdr("vcycle_solve!  (34³ Dirichlet)")
run_one("vcycle_solve", Float32, ()->(s=zeros(34,34,34); s[1,:,:].=1; s[end,:,:].=1; (s, rand(34,34,34).*1e-2)),
    (be,T,h)->begin
        sol=PK.to_device(be,T.(h[1]),T); rhs=PK.to_device(be,T.(h[2]),T)
        PK.vcycle_solve!(sol,rhs; cycle=:W, rtol=1e-6, maxcycles=30, dirichlet=true); Array(sol)
    end)

# ---- batched (8 same-size subgrids) ----
hdr("batched MG  (8×18³)")
run_one("vcycle_batched", Float32, ()->(s=zeros(18,18,18,8); s[1,:,:,:].=1; s[end,:,:,:].=1; (s, rand(18,18,18,8).*1e-2)),
    (be,T,h)->begin
        sol=PK.to_device(be,T.(h[1]),T); rhs=PK.to_device(be,T.(h[2]),T)
        PK.vcycle_batched!(sol,rhs; cycle=:W, ncyc=20, dirichlet=true); Array(sol)
    end)
run_one("comp_accel_batched", Float32, ()->rand(18,18,18,8),
    (be,T,h)->begin
        φ=PK.to_device(be,T.(h),T)
        d1=PK.device_zeros(be,T,(16,16,16,8));d2=similar(d1);d3=similar(d1)
        PK.comp_accel_batched!(d1,d2,d3,φ; iflag=1, start=(1,1,1), del=(T(0.01),T(0.01),T(0.01)))
        (Array(d1),Array(d2),Array(d3))
    end)
println("\nDONE — all CPU↔Metal relL2 should be ~1e-6 (f32 round-off) or smaller.")
