# Passive colour (species) advection — rides the PPM mass flux.
#
# Properties certified here (no Fortran reference needed — this is a property
# test of the conservative colour update layered on the certified PPM sweep):
#   1. colours=() is BIT-IDENTICAL to a run with no colours (zero regression).
#   2. a uniform mass fraction stays EXACTLY uniform (consistency with density).
#   3. colour mass Σ c·dV is conserved to round-off.
#   4. the advected fraction stays bounded (no spurious new extrema).

@testset "Colour (species) advection" begin
    ng = 4
    N  = 16
    dim = N + 2ng
    dims = (dim, dim, dim)
    T = Float64

    # periodic ghost fill for any set of flat fields
    function pbc!(fields...)
        for F in fields
            A = reshape(F, dim, dim, dim)
            for k in 1:dim, j in 1:dim, i in 1:dim
                (i>ng && i<=ng+N && j>ng && j<=ng+N && k>ng && k<=ng+N) && continue
                ii = mod(i-ng-1,N)+ng+1; jj = mod(j-ng-1,N)+ng+1; kk = mod(k-ng-1,N)+ng+1
                A[i,j,k] = A[ii,jj,kk]
            end
        end
    end
    act(F) = reshape(F, dim, dim, dim)[ng+1:ng+N, ng+1:ng+N, ng+1:ng+N]

    mkstate() = begin
        d  = fill(T(1.0), dim^3)
        vx = fill(T(0.7), dim^3); vy = zeros(T, dim^3); vz = zeros(T, dim^3)
        ge = fill(T(1.0)/((5/3-1)*1.0), dim^3)
        e  = ge .+ T(0.5)*T(0.7)^2
        g0 = zeros(T, dim^3)
        (; d, e, ge, vx, vy, vz, g0)
    end

    # 1. regression — colours=() bit-identical to omitting the kwarg
    sa = mkstate(); sb = mkstate()
    bc0(d,e,ge,vx,vy,vz) = pbc!(d,e,ge,vx,vy,vz)
    PPMKernels.ppm_step_3d!(sa.d, sa.e, sa.ge, sa.vx, sa.vy, sa.vz, sa.g0, sa.g0, sa.g0,
                            dims, ng; dt=0.01, gamma=5/3, bc! = bc0)
    PPMKernels.ppm_step_3d!(sb.d, sb.e, sb.ge, sb.vx, sb.vy, sb.vz, sb.g0, sb.g0, sb.g0,
                            dims, ng; dt=0.01, gamma=5/3, bc! = bc0, colours=())
    @test sa.d == sb.d
    @test sa.e == sb.e
    @test sa.ge == sb.ge

    # 2-4. uniform fraction invariance + conservation + boundedness
    s = mkstate()
    c_unif = T(0.3) .* s.d                       # uniform fraction 0.3
    c_blob = similar(s.d)                        # Gaussian-fraction blob
    let A = reshape(c_blob, dim,dim,dim), D = reshape(s.d, dim,dim,dim)
        for k in 1:dim, j in 1:dim, i in 1:dim
            x = (i-ng-0.5)/N; A[i,j,k] = D[i,j,k]*(T(0.1) + T(0.5)*exp(-((x-0.5)^2)/T(0.01)))
        end
    end
    m_u0 = sum(act(c_unif)); m_b0 = sum(act(c_blob))
    bcA(d,e,ge,vx,vy,vz) = pbc!(d,e,ge,vx,vy,vz, c_unif, c_blob)
    for step in 1:20
        order = isodd(step) ? (3,2,1) : (1,2,3)
        PPMKernels.ppm_step_3d!(s.d, s.e, s.ge, s.vx, s.vy, s.vz, s.g0, s.g0, s.g0,
                                dims, ng; dt=0.02, gamma=5/3, order=order,
                                bc! = bcA, colours=(c_unif, c_blob))
    end
    fr = act(c_unif) ./ act(s.d)
    @test maximum(abs.(fr .- 0.3)) < 1e-12                     # uniform fraction invariant
    @test abs(sum(act(c_unif)) - m_u0) / m_u0 < 1e-13          # conservation (uniform)
    @test abs(sum(act(c_blob)) - m_b0) / m_b0 < 1e-12          # conservation (blob)
    frb = act(c_blob) ./ act(s.d)
    @test minimum(frb) >= 0.1 - 1e-9                           # bounded below by initial min
    @test maximum(frb) <= 0.6 + 1e-9                           # bounded above by initial max
end

# Same colour properties for the one-ghost Local PPM (muscl_hancock recon=:ppm_local),
# which advects colours via the per-sweep mass flux fd (not the ppm sweep's df). This is
# the path the cosmology HydroMethod-10 slot uses to carry HII/H2I/HDI species.
@testset "Colour advection — muscl-hancock (Local PPM)" begin
    ng = 4; N = 16; dim = N + 2ng; dims = (dim, dim, dim)
    act(a) = reshape(a, dims)[ng+1:dim-ng, ng+1:dim-ng, ng+1:dim-ng]
    function build(::Type{T}, be) where {T}
        n = prod(dims)
        D = zeros(T, n); v1 = zeros(T, n); v2 = zeros(T, n); v3 = zeros(T, n); TE = zeros(T, n)
        @inbounds for k in 1:dim, j in 1:dim, i in 1:dim
            x=(i-0.5)/dim; y=(j-0.5)/dim; z=(k-0.5)/dim; c=i+dim*(j-1)+dim*dim*(k-1)
            D[c]=1+0.3*sin(2π*x)*cos(2π*y)+0.1*sin(2π*z)
            v1[c]=0.2cos(2π*x); v2[c]=0.15sin(2π*y); v3[c]=0.1cos(2π*z)
            TE[c]=1.5+0.5*(v1[c]^2+v2[c]^2+v3[c]^2)
        end
        d(x)=PPMKernels.to_device(be, x, T)
        d(D),d(v1),d(v2),d(v3),d(TE)
    end
    function run(be, ::Type{T}; nsteps=8) where {T}
        D,v1,v2,v3,TE = build(T, be)
        S1=D.*v1; S2=D.*v2; S3=D.*v3; Tau=D.*TE
        Ge=D.*(TE .- T(0.5).*(v1.^2 .+ v2.^2 .+ v3.^2))
        cE = copy(D)             # colour ≡ ρ → must track ρ exactly
        cU = T(0.3) .* D         # uniform fraction
        bc!(a...) = PPMKernels.fill_periodic!(dims, ng, a...)
        for s in 1:nsteps
            bc!(D,S1,S2,S3,Tau,Ge); bc!(cE); bc!(cU)
            PPMKernels.muscl_hancock_step_3d!(D,S1,S2,S3,Tau, dims, ng; dt=0.01, gamma=5/3,
                dx=1/N, ge=Ge, order = isodd(s) ? (1,2,3) : (3,2,1),
                recon=:ppm_local, predictor=:trace, riemann=:hll, face_periodic=true,
                colours=(cE, cU))
        end
        Array(D), Array(cE), Array(cU)
    end
    be = PPMKernels.backend(:cpu)
    D, cE, cU = run(be, Float64)
    Da=act(D); cEa=act(cE); cUa=act(cU); mU0 = sum(0.3 .* Da)
    @test maximum(abs.(cEa ./ Da .- 1)) < 1e-12               # colour≡ρ tracks ρ
    @test maximum(abs.(cUa ./ Da .- 0.3)) < 1e-12             # uniform fraction invariant
    @test abs(sum(cUa) - sum(act(0.3 .* D))) < 1e-12          # cU stays = 0.3·ρ everywhere
    # conservation: re-run tracking Σ
    D2,v1,v2,v3,TE = build(Float64, be)
    S1=D2.*v1;S2=D2.*v2;S3=D2.*v3;Tau=D2.*TE;Ge=D2.*(TE .- 0.5.*(v1.^2 .+v2.^2 .+v3.^2)); cU2=0.3 .*D2
    m0=sum(act(cU2)); bc!(a...)=PPMKernels.fill_periodic!(dims,ng,a...)
    for s in 1:8
        bc!(D2,S1,S2,S3,Tau,Ge); bc!(cU2)
        PPMKernels.muscl_hancock_step_3d!(D2,S1,S2,S3,Tau,dims,ng; dt=0.01,gamma=5/3,dx=1/N,ge=Ge,
            order = isodd(s) ? (1,2,3) : (3,2,1), recon=:ppm_local,predictor=:trace,riemann=:hll,
            face_periodic=true, colours=(cU2,))
    end
    @test abs(sum(act(cU2)) - m0)/m0 < 1e-12                  # species mass conserved
    if metal_ready()
        _, cEc, cUc = run(PPMKernels.backend(:cpu), Float32)
        _, cEm, cUm = run(PPMKernels.backend(:metal), Float32)
        @test maximum(abs.(cEm .- cEc)) / (maximum(abs.(cEc))+1f-30) < 1f-3
        @test maximum(abs.(cUm .- cUc)) / (maximum(abs.(cUc))+1f-30) < 1f-3
    end
end
