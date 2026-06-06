# Task 2 crux: validate the SUBGRID multigrid solve in PHYSICAL units, with a
# parent-supplied Dirichlet boundary — the new/risky piece of composite gravity
# (root FFT + per-subgrid MG with parent-interpolated boundary φ).
#
# Enzo's mg_relax/mg_calc_defect fold the grid spacing into the RHS: on a cubic
# node grid (d nodes/axis, physical size S/axis) the operator at convergence is
#       ∇²sol = rhs / ((d-1)·S²)        ⇒   rhs = (d-1)·S²·ρ   to solve ∇²φ = ρ.
# We verify that with a manufactured solution: pick a smooth φ_a, set ρ = ∇²φ_a,
# impose φ_a on the boundary (a perfect "parent" value), MG-solve the interior,
# and check we recover φ_a. A pass confirms the FFT↔MG normalization + Dirichlet
# boundary handling that the level>0 gravity slot will rely on.
#
# Run: <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/composite_subgrid_test.jl

using PoissonKernels, Printf
const PK = PoissonKernels

# Manufactured smooth solution on a sub-box [x0,x0+S]³ (NOT periodic — Dirichlet).
# φ_a = sin(2π x) sin(2π y) sin(2π z) ⇒ ∇²φ_a = -3(2π)² φ_a.
φa(x,y,z) = sinpi(2x)*sinpi(2y)*sinpi(2z)
lapφa(x,y,z) = -3*(2π)^2 * φa(x,y,z)

function run(; d=65, x0=0.25, S=0.5, backend=:cpu)
    be = PK.backend(backend); T = Float64
    h = S/(d-1)
    coord(i) = x0 + (i-1)*h                       # node i ∈ 1:d → physical coord
    # analytic field + source on the node grid
    φ_an = Array{T,3}(undef, d,d,d); ρ = Array{T,3}(undef, d,d,d)
    @inbounds for k in 1:d, j in 1:d, i in 1:d
        x=coord(i); y=coord(j); z=coord(k)
        φ_an[i,j,k] = φa(x,y,z); ρ[i,j,k] = lapφa(x,y,z)
    end
    # sol: boundary = analytic φ (parent Dirichlet), interior = 0 (cold start)
    sol = zeros(T, d,d,d)
    sol[1,:,:].=φ_an[1,:,:]; sol[d,:,:].=φ_an[d,:,:]
    sol[:,1,:].=φ_an[:,1,:]; sol[:,d,:].=φ_an[:,d,:]
    sol[:,:,1].=φ_an[:,:,1]; sol[:,:,d].=φ_an[:,:,d]
    # rhs = (d-1)·S²·ρ  (the derived normalization)
    rhs = ((d-1)*S^2) .* ρ
    sd = PK.to_device(be, sol, T); rd = PK.to_device(be, rhs, T)
    _, nrm, rel = PK.vcycle_solve!(sd, rd; cycle=:W, rtol=1e-10, maxcycles=80, dirichlet=true)
    got = PK.to_host(sd)
    # interior relative L2 error vs analytic
    I = 2:d-1
    a = got[I,I,I]; b = φ_an[I,I,I]
    relL2 = sqrt(sum(abs2, a.-b)/sum(abs2, b))
    Linf  = maximum(abs, a.-b)
    @printf("  d=%-4d S=%.3f  MG resid rel=%.1e  → φ recovery relL2=%.3e  Linf=%.3e\n",
            d, S, rel, relL2, Linf)
    return relL2
end

# Definitive normalization check: a QUADRATIC φ has an EXACT 2nd-order discrete
# Laplacian (no truncation), so a precise normalization must recover it to round-off.
function run_quadratic(; d=49, x0=0.25, S=0.5, backend=:cpu)
    be = PK.backend(backend); T = Float64; h = S/(d-1); coord(i)=x0+(i-1)*h
    q(x,y,z) = x*x + y*y + z*z                    # ∇²q = 6 (constant)
    φ_an = Array{T,3}(undef,d,d,d)
    @inbounds for k in 1:d,j in 1:d,i in 1:d; φ_an[i,j,k]=q(coord(i),coord(j),coord(k)); end
    sol = zeros(T,d,d,d)
    sol[1,:,:].=φ_an[1,:,:]; sol[d,:,:].=φ_an[d,:,:]; sol[:,1,:].=φ_an[:,1,:]
    sol[:,d,:].=φ_an[:,d,:]; sol[:,:,1].=φ_an[:,:,1]; sol[:,:,d].=φ_an[:,:,d]
    rhs = ((d-1)*S^2*6.0) .* ones(T,d,d,d)        # ρ=6 ⇒ rhs=(d-1)·S²·6
    sd=PK.to_device(be,sol,T); rd=PK.to_device(be,rhs,T)
    _,_,rel = PK.vcycle_solve!(sd,rd; cycle=:W, rtol=1e-12, maxcycles=200, dirichlet=true)
    got=PK.to_host(sd); I=2:d-1
    relL2=sqrt(sum(abs2,got[I,I,I].-φ_an[I,I,I])/sum(abs2,φ_an[I,I,I]))
    @printf("  QUADRATIC φ (exact discrete ∇², dirichlet): resid rel=%.1e → recovery relL2=%.3e  %s\n",
            rel, relL2, relL2<1e-6 ? "(round-off ⇒ normalization EXACT)" : "(NOT round-off)")
    return relL2
end

# Hypothesis: mg_prolong! writes the full fine grid (incl boundary ring), so the
# prolong-and-add can perturb a NON-ZERO Dirichlet boundary (the certified tests
# all used zero boundaries). Re-impose the parent boundary after each μ-cycle and
# see if the quadratic recovers to round-off.
function run_quadratic_reimpose(; d=49, x0=0.25, S=0.5, ncyc=60)
    be = PK.backend(:cpu); T = Float64; h = S/(d-1); coord(i)=x0+(i-1)*h
    q(x,y,z) = x*x + y*y + z*z
    φ_an = Array{T,3}(undef,d,d,d)
    @inbounds for k in 1:d,j in 1:d,i in 1:d; φ_an[i,j,k]=q(coord(i),coord(j),coord(k)); end
    setbnd!(A) = (A[1,:,:].=φ_an[1,:,:];A[d,:,:].=φ_an[d,:,:];A[:,1,:].=φ_an[:,1,:];
                  A[:,d,:].=φ_an[:,d,:];A[:,:,1].=φ_an[:,:,1];A[:,:,d].=φ_an[:,:,d])
    sol = zeros(T,d,d,d); setbnd!(sol)
    rhs = ((d-1)*S^2*6.0) .* ones(T,d,d,d)
    dims = PK.mg_dims_schedule(size(sol)); nlev=length(dims)
    Sol=Vector{Any}(undef,nlev); RHS=Vector{Any}(undef,nlev); Def=Vector{Any}(undef,nlev)
    Sol[1]=PK.to_device(be,sol,T); RHS[1]=PK.to_device(be,rhs,T); Def[1]=PK.device_zeros(be,T,dims[1])
    for L in 2:nlev; Sol[L]=PK.device_zeros(be,T,dims[L]);RHS[L]=PK.device_zeros(be,T,dims[L]);Def[L]=PK.device_zeros(be,T,dims[L]); end
    for _ in 1:ncyc
        PK._mu_cycle!(Sol,RHS,Def,1,nlev,2,3,1)
        setbnd!(Sol[1])                      # ← re-impose parent Dirichlet boundary
    end
    got=PK.to_host(Sol[1]); I=2:d-1
    relL2=sqrt(sum(abs2,got[I,I,I].-φ_an[I,I,I])/sum(abs2,φ_an[I,I,I]))
    @printf("  QUADRATIC + boundary re-imposed: recovery relL2=%.3e  %s\n",
            relL2, relL2<1e-8 ? "(round-off ⇒ pollution CONFIRMED+FIXED)" : "(still off)")
    return relL2
end

# ── full two-level composite: root FFT → trilinear parent-φ interp → subgrid MG ──
# This is exactly what the level>0 gravity hook will do: the subgrid's Dirichlet
# boundary is the parent potential interpolated to the subgrid, NOT the analytic
# value. Periodic root source with a known analytic φ; verify the subgrid solution.
φ_periodic(x,y,z) = sinpi(2x)*sinpi(2y)*sinpi(2z)              # ∇²φ = -3(2π)²φ
ρ_periodic(x,y,z) = -3*(2π)^2 * φ_periodic(x,y,z)

# trilinear sample of a CELL-CENTERED, periodic field φr (N³, cell i center=(i-0.5)/N)
function sample_cc(φr, N, x, y, z)
    gx=x*N-0.5; gy=y*N-0.5; gz=z*N-0.5
    i0=floor(Int,gx); fx=gx-i0; j0=floor(Int,gy); fy=gy-j0; k0=floor(Int,gz); fz=gz-k0
    w(i)=mod(i,N)+1; i0w=w(i0);i1w=w(i0+1);j0w=w(j0);j1w=w(j0+1);k0w=w(k0);k1w=w(k0+1)
    (φr[i0w,j0w,k0w]*(1-fx)*(1-fy)*(1-fz) + φr[i1w,j0w,k0w]*fx*(1-fy)*(1-fz) +
     φr[i0w,j1w,k0w]*(1-fx)*fy*(1-fz)     + φr[i1w,j1w,k0w]*fx*fy*(1-fz) +
     φr[i0w,j0w,k1w]*(1-fx)*(1-fy)*fz     + φr[i1w,j0w,k1w]*fx*(1-fy)*fz +
     φr[i0w,j1w,k1w]*(1-fx)*fy*fz         + φr[i1w,j1w,k1w]*fx*fy*fz)
end

function run_twolevel(; N=64, d=65, x0=0.25, S=0.5)
    be=PK.backend(:cpu); T=Float64
    # root: cell-centered periodic source, FFT solve ∇²φ=ρ
    ρr=Array{T,3}(undef,N,N,N)
    @inbounds for k in 1:N,j in 1:N,i in 1:N
        ρr[i,j,k]=ρ_periodic((i-0.5)/N,(j-0.5)/N,(k-0.5)/N); end
    ρr .-= sum(ρr)/length(ρr)
    φr=Array{T,3}(undef,N,N,N)
    PK.fft_poisson_root!(φr, ρr; G=1.0,a=1.0,boxsize=1.0)
    # root accuracy vs analytic (zero-mean φ_periodic on the cell centers)
    φra=[φ_periodic((i-0.5)/N,(j-0.5)/N,(k-0.5)/N) for i in 1:N,j in 1:N,k in 1:N]
    rootrel=sqrt(sum(abs2,φr.-φra)/sum(abs2,φra))
    # subgrid node grid; boundary = trilinear interp of the ROOT potential
    h=S/(d-1); coord(i)=x0+(i-1)*h
    φsa=[φ_periodic(coord(i),coord(j),coord(k)) for i in 1:d,j in 1:d,k in 1:d]
    sol=zeros(T,d,d,d)
    for k in 1:d,j in 1:d,i in 1:d
        if i==1||i==d||j==1||j==d||k==1||k==d
            sol[i,j,k]=sample_cc(φr,N,coord(i),coord(j),coord(k))   # ← parent-interpolated boundary
        end
    end
    rhs=Array{T,3}(undef,d,d,d)
    @inbounds for k in 1:d,j in 1:d,i in 1:d
        rhs[i,j,k]=(d-1)*S^2*ρ_periodic(coord(i),coord(j),coord(k)); end
    sd=PK.to_device(be,sol,T); rd=PK.to_device(be,rhs,T)
    PK.vcycle_solve!(sd,rd; cycle=:W, rtol=1e-10, maxcycles=80, dirichlet=true)
    got=PK.to_host(sd); I=2:d-1
    subrel=sqrt(sum(abs2,got[I,I,I].-φsa[I,I,I])/sum(abs2,φsa[I,I,I]))
    @printf("  TWO-LEVEL: root FFT relL2=%.2e ; subgrid (parent-interp BC) relL2=%.3e  %s\n",
            rootrel, subrel, subrel<5e-3 ? "(composite chain OK)" : "(off)")
    return subrel
end

println("Subgrid MG in physical units — manufactured-solution recovery (rhs=(d-1)·S²·ρ):")
eq = run_quadratic(d=49, S=0.5)
eqr = run_quadratic_reimpose(d=49, S=0.5)
println()
e1 = run(d=33, S=0.5)
e2 = run(d=65, S=0.5)
e3 = run(d=65, S=0.25)
println()
et = run_twolevel()
@printf("\nconvergence (d 33→65, S=0.5):  relL2 %.3e → %.3e  (ratio %.2f, expect ~4 for 2nd order)\n",
        e1, e2, e1/e2)
ok = eq < 1e-8 && e2 < e1 && e1/e2 > 3.0 && et < 5e-3
println(ok ? "PASS — rhs=(d-1)·S²·ρ + dirichlet re-imposition: quadratic→round-off, sin 2nd-order convergent, two-level composite OK" :
             "CHECK — normalization or convergence off")
