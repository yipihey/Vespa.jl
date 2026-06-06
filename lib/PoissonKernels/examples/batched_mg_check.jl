# Validate the batched multigrid (NB same-size subgrids in one launch-set) and
# measure the launch-overhead win vs the per-subgrid loop, on CPU and Metal.
# Each batch element is a DISTINCT manufactured quadratic φ_b = c_b·(x²+y²+z²)
# (∇²=6c_b, exact discrete Laplacian) with c_b varying ⇒ confirms no cross-talk
# AND round-off recovery. Run: <jl> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/batched_mg_check.jl
using PoissonKernels, Printf
try; @eval using Metal; catch; end
const PK = PoissonKernels

function setup(d, NB; x0=0.25, S=0.5)
    h=S/(d-1); coord(i)=x0+(i-1)*h
    φ=Array{Float64,4}(undef,d,d,d,NB); sol=zeros(Float64,d,d,d,NB); rhs=Array{Float64,4}(undef,d,d,d,NB)
    for b in 1:NB
        c=1.0+0.1*b
        @inbounds for k in 1:d,j in 1:d,i in 1:d
            q=coord(i)^2+coord(j)^2+coord(k)^2; φ[i,j,k,b]=c*q; rhs[i,j,k,b]=(d-1)*S^2*6.0*c
        end
        # Dirichlet boundary = analytic
        sol[1,:,:,b].=φ[1,:,:,b]; sol[d,:,:,b].=φ[d,:,:,b]; sol[:,1,:,b].=φ[:,1,:,b]
        sol[:,d,:,b].=φ[:,d,:,b]; sol[:,:,1,b].=φ[:,:,1,b]; sol[:,:,d,b].=φ[:,:,d,b]
    end
    sol, rhs, φ
end

function check(be_name, T; d=18, NB=64)
    PK.has_backend(be_name) || return println("  [$be_name] not available")
    be=PK.backend(be_name)
    sol,rhs,φ = setup(d,NB)
    sd=PK.to_device(be,T.(sol),T); rd=PK.to_device(be,T.(rhs),T)
    PK.vcycle_batched!(sd,rd; cycle=:W, ncyc=60, dirichlet=true)
    got=Float64.(PK.to_host(sd)); I=2:d-1
    worst=0.0
    for b in 1:NB
        a=got[I,I,I,b]; e=φ[I,I,I,b]; worst=max(worst, sqrt(sum(abs2,a.-e)/sum(abs2,e)))
    end
    # timing: batched vs per-subgrid loop (per-subgrid via vcycle_solve! fixed-ish)
    tb = @elapsed (PK.vcycle_batched!(PK.to_device(be,T.(sol),T), PK.to_device(be,T.(rhs),T); cycle=:W, ncyc=60, dirichlet=true))
    tl = @elapsed for b in 1:NB
        s=PK.to_device(be,T.(sol[:,:,:,b]),T); r=PK.to_device(be,T.(rhs[:,:,:,b]),T)
        PK.vcycle_solve!(s,r; cycle=:W, rtol=1e-12, maxcycles=60, dirichlet=true)
    end
    @printf("  [%-5s %s] batched %d×%d³: worst relL2=%.2e | batched %.3fs vs per-subgrid loop %.3fs → %.1f×\n",
            be_name, T, NB, d, worst, tb, tl, tl/tb)
end

println("Batched multigrid validation + launch-overhead win:")
check(:cpu, Float64)
check(:metal, Float32)
