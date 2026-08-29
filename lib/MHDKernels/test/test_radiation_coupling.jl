using Test

@testset "spectral radiation coupling" begin
    @testset "robust PLM limiter" begin
        L=ntuple(_->0f0,9)
        M=ntuple(_->1f0,9)
        R=ntuple(_->3f0,9)
        moncen=MHDKernels.prim_slope(L,M,R)
        limited=MHDKernels.prim_slope_minmod(L,M,R)
        @test all(limited .== 0.5f0)
        @test all(limited .<= moncen)

        turning=ntuple(_->0f0,9)
        @test all(MHDKernels.prim_slope_minmod(L,M,turning) .== 0f0)
        @test MHDKernels.recon_code(:plm_minmod) ==
              MHDKernels.RECON_PLM_MINMOD
    end

    @testset "authoritative density perturbation" begin
        delta=zeros(Float32,1)
        rho=1f0
        for _ in 1:32
            rho=MHDKernels._cube_density_update!(delta,1,rho,1f-8,Val(true))
        end
        @test delta[1] ≈ 3.2f-7 rtol=2f-6
        @test rho == 1f0

        s=allocate_state(backend(:cpu),Float32,(4,4,4);dx=0.25)
        fill!(s.U[1],1f0)
        ws=allocate_radiation_workspace(s)
        ws.density_perturbation[1]=4f-8
        reconstruct_radiation_density!(ws,s)
        @test ws.density_perturbation[1] == 4f-8
        @test s.U[1][1] == 1f0

        fill!(ws.density_perturbation,0f0)
        fill!(ws.fft_real,1f-8)
        foreach(x->fill!(x,0f0),ws.correction)
        for _ in 1:32
            MHDKernels._radiation_apply_correction!(s,ws,ws.correction)
        end
        @test ws.density_perturbation[1] ≈ 3.2f-7 rtol=2f-6
        @test s.U[1][1] == 1f0+Float32(3.2f-7)

        fill!(ws.fft_real,4f-8)
        MHDKernels._radiation_apply_correction!(s,ws,ws.correction;
                                                density_absolute=true)
        @test ws.density_perturbation[1] == 4f-8
        @test s.U[1][1] == 1f0
    end

    @testset "positivity blend scales the coupled correction" begin
        s=allocate_state(backend(:cpu),Float32,(4,4,4);dx=0.25)
        fill!(s.U[1],1f0); fill!(s.U[2],0f0); fill!(s.U[3],0f0)
        fill!(s.U[4],0f0); fill!(s.U[5],1f0)
        foreach(x->fill!(x,0f0),s.U[6:9])
        ws=allocate_radiation_workspace(s)
        fill!(ws.density_perturbation,0f0)
        fill!(ws.fft_real,-0.4f0)
        fill!(ws.correction[1],2f0)
        fill!(ws.correction[2],0f0)
        fill!(ws.correction[3],0f0)

        MHDKernels._radiation_apply_correction!(s,ws,ws.correction;
                                                density_blend=0.25f0)

        @test all(s.U[1].==0.9f0)
        @test all(ws.density_perturbation.==-0.1f0)
        @test all(s.U[2]./s.U[1].==0.5f0)
    end

    @testset "matched asymptotic coefficients" begin
        rc=RadiationClosure(inertia_ratio=0.2f0,mean_free_path=0.01f0,
                            photon_speed=20f0,gas_sound_speed=1f0)
        low=radiation_mode_coefficients(1f-3,7f0,rc)
        high=radiation_mode_coefficients(1f8,7f0,rc)
        @test low.streaming_weight < 1f-9
        @test low.inertia_response ≈ 1f0/6f0 rtol=2f-6
        @test low.radiation_sound_speed2 ≈ 400f0/3.6f0 rtol=2f-6
        @test high.streaming_weight > 0.999999
        @test high.streaming_weight == 1
        @test high.inertia_response ≈ 1 rtol=2f-6
        @test high.gamma_transverse == 7
        @test high.radiation_sound_speed2 == 0
    end

    @testset "angular transport response" begin
        q1=radiation_angular_response(1.0)
        @test q1.monopole ≈ π/4 rtol=2e-7
        @test q1.longitudinal ≈ 3*(1-π/4) rtol=2e-7
        @test q1.transverse ≈ (3π/4-q1.longitudinal)/2 rtol=2e-7

        R=0.2f0; lambda=0.01f0; c=20f0
        rc=RadiationClosure(inertia_ratio=R,mean_free_path=lambda,
                            photon_speed=c,gas_sound_speed=1f0,
                            response_model=:angular)
        gamma=c/(R*lambda)
        tight=radiation_mode_coefficients(1f0,gamma,rc)
        expected=c*lambda/(5f0*(1f0+R))
        @test tight.response_model === :angular
        @test tight.gamma_transverse ≈ expected rtol=3f-3
        @test tight.gamma_longitudinal/tight.gamma_transverse ≈ 4f0/3f0 rtol=2f-4
        @test tight.radiation_sound_speed2 ≈ c*c/(3f0*(1f0+R)) rtol=2f-3

        free=radiation_mode_coefficients(1f8,7f0,rc)
        @test free.streaming_weight > 0.998f0
        @test free.inertia_response > 0.999f0
        @test free.gamma_transverse ≈ 7f0 rtol=3f-3
        @test free.gamma_longitudinal ≈ 7f0 rtol=3f-3
        @test free.radiation_sound_speed2 < 1f-8
    end

    @testset "stiff acoustic propagator retains the slow mode in Float32" begin
        g=1f8; c2=1f0; k2=1f8; dt=0.25f0
        B32,S32=MHDKernels._radiation_acoustic_coefficients(g,c2,k2,dt)
        B64,S64=MHDKernels._radiation_acoustic_coefficients(
            Float64(g),Float64(c2),Float64(k2),Float64(dt))
        @test isfinite(B32) && isfinite(S32)
        @test B32 < 0 && S32 > 0
        @test B32 ≈ Float32(B64) rtol=2f-6
        @test S32 ≈ Float32(S64) rtol=2f-6
    end

    @testset "exponential photon-moment composition" begin
        function rhs(x,force,G,nu,stress,k,cs2,cph2)
            db,ub,dg,ug=x
            ik=im*k
            ComplexF64[-ik*ub,
                -ik*cs2*db+G*(ug-ub)+force,
                -(4/3)*ik*ug,
                -ik*cph2*dg/4+nu*ub-(nu+stress)*ug]
        end
        function rk4_reference(x,force,dt,G,nu,stress,k,cs2,cph2,n)
            h=dt/n; y=copy(x)
            for _ in 1:n
                k1=rhs(y,force,G,nu,stress,k,cs2,cph2)
                k2v=rhs(y.+0.5h.*k1,force,G,nu,stress,k,cs2,cph2)
                k3=rhs(y.+0.5h.*k2v,force,G,nu,stress,k,cs2,cph2)
                k4=rhs(y.+h.*k3,force,G,nu,stress,k,cs2,cph2)
                y .+= (h/6).*(k1.+2k2v.+2k3.+k4)
            end
            y
        end

        G=37.0; nu=11.0; stress=2.5; k=8.0; cs2=0.7; cph2=19.0
        force=0.13-0.07im
        x0=ComplexF64[0.02+0.03im,-0.04+0.01im,0.01-0.02im,0.03+0.04im]
        dt=0.08
        ref=rk4_reference(x0,force,dt,G,nu,stress,k,cs2,cph2,20_000)
        function composed(n)
            y=Tuple(x0); h=dt/n
            for _ in 1:n
                y=MHDKernels._photon_split_longitudinal(y...,force,h,G,nu,
                                                         stress,k,cs2,cph2)
            end
            collect(y)
        end
        e1=maximum(abs,composed(1).-ref)
        e2=maximum(abs,composed(2).-ref)
        e4=maximum(abs,composed(4).-ref)
        @test e1/e2 > 3.5
        @test e2/e4 > 3.5
        @test e4 < 2e-3

        # The affine drag solve must retain both the zero/slow mode and forcing
        # when Gamma*dt is far outside the range of an explicit update.
        ub,ug=MHDKernels._photon_exact_drag_affine(1f0+0f0im,-2f0+0f0im,
            0.25f0+0f0im,1f0,1f8,2f7,0f0)
        conserved=(1f8*ug+2f7*ub)/(1.2f8)
        expected=(1f8*(-2f0)+2f7*1f0)/(1.2f8)+0.25f0*(2f7/1.2f8)
        @test isfinite(real(ub)) && isfinite(real(ug))
        @test real(ub) ≈ expected rtol=3f-6
        @test real(ug) ≈ expected rtol=3f-6
        @test conserved ≈ expected rtol=3f-6
    end

    @testset "RFFT self-conjugate planes remain Hermitian" begin
        N=16
        s=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N)
        ws=allocate_radiation_workspace(s)
        real=randn(Float32,N,N,N,4)
        copyto!(ws.real_batch,real)
        MHDKernels._radiation_forward_batch!(ws.hat_batch,ws.real_batch)
        rc=MHDKernels.RadiationClosure{Float32}(2f-4,1.5f-4,3.7f4,58f0,
                                                5f0,4f0/3f0,15f0/16f0)
        MHDKernels._radiation_homogeneous_hat_k!(s.be)(
            ws.velocity_hat...,ws.density_hat,Int32(N),Float32(2pi),8.4f-8,
            1.3f12,1f0,rc,Val(true);ndrange=length(ws.density_hat))
        KernelAbstractions.synchronize(s.be)
        hat=Array(ws.hat_batch)
        for ii in (1,N÷2+1)
            mismatch=0f0
            scale=0f0
            @inbounds for d in 1:4,k in 1:N,j in 1:N
                jp=mod(-(j-1),N)+1
                kp=mod(-(k-1),N)+1
                mismatch=max(mismatch,abs(hat[ii,j,k,d]-conj(hat[ii,jp,kp,d])))
                scale=max(scale,abs(hat[ii,j,k,d]))
            end
            @test mismatch <= 2f-5*max(scale,1f0)
        end
    end

    @testset "pseudospectral correction has a Nyquist guard" begin
        N=16
        s=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N)
        ws=allocate_radiation_workspace(s)
        copyto!(ws.real_batch,randn(Float32,N,N,N,4))
        MHDKernels._radiation_forward_batch!(ws.hat_batch,ws.real_batch)
        rc=MHDKernels.RadiationClosure{Float32}(2f-4,1.5f-4,3.7f4,58f0,
                                                5f0,4f0/3f0)
        MHDKernels._radiation_homogeneous_hat_k!(s.be)(
            ws.velocity_hat...,ws.density_hat,Int32(N),Float32(2pi),8.4f-8,
            1.3f12,1f0,rc,Val(true);ndrange=length(ws.density_hat))
        KernelAbstractions.synchronize(s.be)
        hat=Array(ws.hat_batch)
        @inbounds for k in 1:N,j in 1:N,i in 1:N÷2+1
            ii=Int32(i-1); jj=Int32(j-1); kk=Int32(k-1)
            MHDKernels._radiation_mode_dealiased(
                ii,jj,kk,Int32(N),rc.nyquist_guard_fraction) && continue
            @test all(iszero,hat[i,j,k,:])
        end
    end

    @testset "transverse force uses exact damped response" begin
        N=16; s=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N)
        ws=allocate_radiation_workspace(s)
        f=ntuple(_->zeros(Float32,N,N,N),3)
        @inbounds for k in 1:N,j in 1:N,i in 1:N
            f[2][i,j,k]=sinpi(2f0*Float32(i-1)/Float32(N))
        end
        MHDKernels._radiation_forward_triplet!(ws,f)
        rc=MHDKernels.RadiationClosure{Float32}(0.2,0.01,20,1,5,4/3)
        dt=0.03f0; gamma=7f0; k=Float32(2pi)
        MHDKernels._radiation_source_hat_k!(s.be)(ws.velocity_hat...,ws.density_hat,
            Int32(N),k,dt,gamma,1f0,rc,Val(false);
            ndrange=length(ws.velocity_hat[1]))
        MHDKernels._radiation_inverse_triplet!(f,ws)
        mode=radiation_mode_coefficients(k,gamma,rc)
        expected=mode.inertia_response*dt*MHDKernels._drag_phi1(mode.gamma_transverse*dt)-
                 dt*MHDKernels._drag_phi1(gamma*dt)
        measured=2f0*sum(f[2][i,1,1]*sinpi(2f0*Float32(i-1)/Float32(N)) for i in 1:N)/N
        @test measured ≈ expected rtol=3f-5 atol=2f-7
        @test maximum(abs,f[1]) < 2f-7
        @test maximum(abs,f[3]) < 2f-7
    end

    @testset "radiation-loaded longitudinal EOS response" begin
        N=16; s=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N)
        amp=1f-3
        fill!(s.U[1],1f0); fill!(s.U[5],2f0)
        @inbounds for k in 1:N,j in 1:N,i in 1:N
            s.U[1][linidx(N,N,i,j,k)]=1f0+amp*cospi(2f0*Float32(i-1)/Float32(N))
        end
        ws=allocate_radiation_workspace(s)
        rc=MHDKernels.RadiationClosure{Float32}(0.2,0.01,20,1,5,4/3)
        dt=0.004f0; gamma=7f0
        MHDKernels._radiation_transform_velocity_density!(ws,s,s.U,dt,gamma,rc)
        vx=ws.correction[1]
        measured=2f0*sum(vx[i,1,1]*sinpi(2f0*Float32(i-1)/Float32(N)) for i in 1:N)/N
        @test measured > 0
        @test maximum(abs,ws.correction[2]) < 2f-7
        @test maximum(abs,ws.correction[3]) < 2f-7
    end

    @testset "full step follows exact longitudinal mode above radiation CFL" begin
        N=64; gamma=5f0/3f0; amp=2f-4; p0=1f0/gamma
        s=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N,
                         gamma=gamma,riemann=:hll)
        @inbounds for k in 1:N,j in 1:N,i in 1:N
            phase=2f0*Float32(pi)*Float32(i-1)/Float32(N)
            rho=1f0+amp*cos(phase)
            p=p0*rho^gamma
            c=linidx(N,N,i,j,k)
            s.U[1][c]=rho; s.U[2][c]=0; s.U[3][c]=0; s.U[4][c]=0
            s.U[5][c]=p/(gamma-1f0)
            s.U[6][c]=0; s.U[7][c]=0; s.U[8][c]=0; s.U[9][c]=0
        end
        ws=allocate_radiation_workspace(s)
        rc=MHDKernels.RadiationClosure{Float32}(0.2,0.001,20,1,5,4/3)
        dt=0.004f0; drag_gamma=7f0; k1=Float32(2pi)
        step_radiation_godunov!(s,ws,dt;drag_impulse=drag_gamma*dt,
                                closure=rc,ch=1f0,integrator=:ref)
        rho=reshape(Array(s.U[1]),N,N,N)
        delta=Array(ws.density_perturbation)
        vx=reshape(Array(s.U[2]),N,N,N)./rho
        rhoamp=2f0*sum(delta[i,1,1]*cospi(2f0*Float32(i-1)/N) for i in 1:N)/N
        vxamp=2f0*sum(vx[i,1,1]*sinpi(2f0*Float32(i-1)/N) for i in 1:N)/N
        mode=radiation_mode_coefficients(k1,drag_gamma,rc)
        c2=1f0+mode.radiation_sound_speed2
        B,S=MHDKernels._radiation_acoustic_coefficients(
            mode.gamma_longitudinal,c2,k1*k1,dt)
        exact_rho=amp*(B+mode.gamma_longitudinal*S)
        exact_v=amp*k1*c2*S
        @info "radiation longitudinal mode" rhoamp exact_rho vxamp exact_v
        @test rhoamp ≈ exact_rho rtol=3f-2 atol=2f-6
        @test vxamp ≈ exact_v rtol=4f-2 atol=2f-6
        @test abs(Float64(sum(delta)) / N^3) < 1e-6
        @test minimum(rho) > 0
    end

    @testset "Lorentz-forced mode follows exact radiation response" begin
        N=64; gamma=5f0/3f0; p0=1f0/gamma; bg=0.5f0; db=0.05f0
        s=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N,
                         gamma=gamma,riemann=:hll)
        @inbounds for k in 1:N,j in 1:N,i in 1:N
            phase=2f0*Float32(pi)*Float32(i-1)/Float32(N)
            by=bg+db*cos(phase); c=linidx(N,N,i,j,k)
            s.U[1][c]=1; s.U[2][c]=0; s.U[3][c]=0; s.U[4][c]=0
            s.U[5][c]=p0/(gamma-1f0)+0.5f0*by*by
            s.U[6][c]=0; s.U[7][c]=by; s.U[8][c]=0; s.U[9][c]=0
        end
        ws=allocate_radiation_workspace(s)
        rc=MHDKernels.RadiationClosure{Float32}(0.2,0.001,20,1,5,4/3)
        dt=0.004f0; drag_gamma=7f0; k1=Float32(2pi)
        step_radiation_godunov!(s,ws,dt;drag_impulse=drag_gamma*dt,
                                closure=rc,ch=1f0,integrator=:ref)
        rho=reshape(Array(s.U[1]),N,N,N); vx=reshape(Array(s.U[2]),N,N,N)./rho
        rhoamp=2f0*sum((rho[i,1,1]-1f0)*cospi(2f0*Float32(i-1)/N) for i in 1:N)/N
        vxamp=2f0*sum(vx[i,1,1]*sinpi(2f0*Float32(i-1)/N) for i in 1:N)/N
        mode=radiation_mode_coefficients(k1,drag_gamma,rc)
        c2=1f0+mode.radiation_sound_speed2
        B,S=MHDKernels._radiation_acoustic_coefficients(
            mode.gamma_longitudinal,c2,k1*k1,dt)
        J=MHDKernels._radiation_integral_S(
            B,S,mode.gamma_longitudinal,c2*k1*k1,dt)
        accel=bg*db*sin(k1/N)*N
        exact_rho=-mode.inertia_response*k1*J*accel
        exact_v=mode.inertia_response*S*accel
        @info "radiation Lorentz-forced mode" rhoamp exact_rho vxamp exact_v
        @test rhoamp ≈ exact_rho rtol=8f-2 atol=2f-7
        @test vxamp ≈ exact_v rtol=8f-2 atol=2f-6
        @test abs(Float64(sum(rho)) / N^3 - 1) < 1e-6
        @test minimum(rho) > 0
    end

    @testset "free-streaming limit recovers local Godunov drag" begin
        N=8
        a=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N,riemann=:hll)
        b=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N,riemann=:hll)
        init_turb_field!(a)
        for v in 1:9 copyto!(b.U[v],a.U[v]) end
        dt,smax=compute_dt(a;cfl=0.03)
        step_drag_godunov!(a,dt;drag_impulse=0.7f0,ch=smax,integrator=:ref)
        rc=RadiationClosure(inertia_ratio=0.2f0,mean_free_path=1f6,
                            photon_speed=20f0,gas_sound_speed=1f0)
        ws=allocate_radiation_workspace(b)
        step_radiation_godunov!(b,ws,dt;drag_impulse=0.7f0,closure=rc,
                                ch=smax,integrator=:ref)
        ha=fields_to_host(a); hb=fields_to_host(b)
        @test maximum(maximum(abs,hb[v].-ha[v]) for v in 1:9) < 3f-6
    end


    @testset "device energy ledger classifies the coupled step" begin
        N=8
        s=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N,riemann=:hll)
        init_turb_field!(s)
        ws=allocate_radiation_workspace(s)
        before=radiation_energy_components(ws,s)
        hbefore=fields_to_host(s)
        @test before.total ≈ before.kinetic+before.internal+before.magnetic rtol=2e-6
        rc=RadiationClosure(inertia_ratio=0.2f0,mean_free_path=0.01f0,
                            photon_speed=20f0,gas_sound_speed=1f0,
                            response_model=:angular)
        ledger=RadiationEnergyLedger()
        dt,smax=compute_dt(s;cfl=0.02)
        step_radiation_godunov!(s,ws,dt;drag_impulse=0.3f0,closure=rc,
                                ch=smax,integrator=:ref,energy_ledger=ledger)
        after=radiation_energy_components(ws,s)
        hafter=fields_to_host(s)
        function host_components(h)
            rho=Float64.(h[1]); mx=Float64.(h[2]); my=Float64.(h[3]); mz=Float64.(h[4])
            bx=Float64.(h[6]); by=Float64.(h[7]); bz=Float64.(h[8])
            ek=sum(0.5 .* (mx.^2 .+ my.^2 .+ mz.^2) ./ rho)
            eb=sum(0.5 .* (bx.^2 .+ by.^2 .+ bz.^2))
            total=sum(Float64.(h[5]))
            (total=total,kinetic=ek,internal=total-ek-eb,magnetic=eb)
        end
        hb=host_components(hbefore); ha=host_components(hafter)
        @test ledger.samples == 1
        @test ledger.total_change ≈ ha.total-hb.total rtol=2e-5 atol=2e-6
        @test ledger.kinetic_change ≈ ha.kinetic-hb.kinetic rtol=2e-5 atol=2e-6
        @test ledger.internal_change ≈ ha.internal-hb.internal rtol=1e-4 atol=2e-6
        @test ledger.magnetic_change ≈ ha.magnetic-hb.magnetic rtol=2e-5 atol=2e-6
        @test all(isfinite,(ledger.total_change,ledger.kinetic_change,
                            ledger.internal_change,ledger.magnetic_change))
        @test ledger.component_residual_abs >= abs(ledger.component_residual)
        @test ledger.component_magnitude_abs >= ledger.component_residual_abs
        component_scale=abs(ledger.kinetic_change)+abs(ledger.internal_change)+
                        abs(ledger.magnetic_change)
        @test abs(ledger.component_residual)/component_scale < 5e-4
        @test ledger.component_residual_abs/ledger.component_magnitude_abs < 1e-6
    end

    @testset "evolved photon moments close momentum and angular stress" begin
        R=0.2f0; G=7f0; nu=R*G; dt=0.4f0
        ub0=ComplexF32(0.7,-0.2); ug0=ComplexF32(-0.1,0.3)
        ub1,ug1=MHDKernels._photon_sdirk_transverse(ub0,ug0,0f0,dt,G,nu,0f0)
        @test ub1+ug1/R ≈ ub0+ug0/R rtol=3f-6

        rc=RadiationClosure(inertia_ratio=R,mean_free_path=1f-4,
                            photon_speed=20f0,gas_sound_speed=1f0,
                            bridge_q2=5f0,longitudinal_viscosity=4f0/3f0,
                            nyquist_guard_fraction=1f0,
                            response_model=:moments)
        k=3f0
        st,sl=MHDKernels._photon_stress_rates(k*k,nu,rc)
        @test st ≈ nu*(k*rc.mean_free_path)^2/5 rtol=5f-3
        @test sl/st ≈ 4f0/3f0 rtol=2f-4
        @test radiation_mode_coefficients(k,G,rc).response_model === :moments

        streaming=RadiationClosure(inertia_ratio=R,mean_free_path=1f4,
                                   photon_speed=20f0,gas_sound_speed=1f0,
                                   bridge_q2=5f0,longitudinal_viscosity=4f0/3f0,
                                   nyquist_guard_fraction=1f0,
                                   response_model=:moments)
        stfree,_=MHDKernels._photon_stress_rates(k*k,nu,streaming)
        @test stfree > 0
        @test isfinite(stfree)

        # Eq. 57: an adiabatic tight photon-baryon mode propagates with
        # c/sqrt(3(1+R)). Integrate the closed moments directly so this gate is
        # independent of finite-volume spatial error.
        Gtight=1f4; nutight=R*Gtight; lambda=20f0/nutight
        tight=RadiationClosure(inertia_ratio=R,mean_free_path=lambda,
            photon_speed=20f0,gas_sound_speed=0f0,bridge_q2=5f0,
            longitudinal_viscosity=4f0/3f0,nyquist_guard_fraction=1f0,
            response_model=:moments)
        _,sltight=MHDKernels._photon_stress_rates(1f0,nutight,tight)
        db=ComplexF32(1,0); ub=ComplexF32(0); dg=ComplexF32(4f0/3f0); ug=ComplexF32(0)
        dti=1f-4
        for _ in 1:1000
            db,ub,dg,ug=MHDKernels._photon_sdirk_longitudinal(
                db,ub,dg,ug,0f0,dti,Gtight,nutight,sltight,1f0,0f0,400f0)
        end
        ceq=20f0/sqrt(3f0*(1f0+R))
        expected=cos(ceq*0.1f0)*exp(-0.5f0*sltight/(1f0+R)*0.1f0)
        @test real(db) ≈ expected rtol=8f-3
        @test abs(imag(db)) < 2f-5

        N=16; sn=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N)
        wn=allocate_radiation_workspace(sn); pn=allocate_photon_moment_state(sn)
        fill!(wn.hat_batch,0); fill!(pn.hat_batch,0)
        pn.velocity_hat[1][N÷2+1,1,1]=ComplexF32(1)
        MHDKernels._photon_moment_homogeneous_hat_k!(sn.be)(
            wn.velocity_hat...,wn.density_hat,pn.velocity_hat...,pn.delta_hat,
            Int32(N),Float32(2pi),0.1f0,7f0,1f0,tight,Val(true),Val(true);
            ndrange=length(wn.density_hat))
        KernelAbstractions.synchronize(sn.be)
        @test pn.velocity_hat[1][N÷2+1,1,1]==ComplexF32(1)
        @test all(iszero,wn.hat_batch)
    end

    @testset "photon moment state advances with full Godunov MHD" begin
        N=16; gamma=5f0/3f0; amp=1f-4
        s=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N,
                         gamma=gamma,riemann=:hll)
        @inbounds for k in 1:N,j in 1:N,i in 1:N
            phase=2f0*Float32(pi)*Float32(i-1)/Float32(N)
            rho=1f0+amp*cos(phase); vx=amp*sin(phase); c=linidx(N,N,i,j,k)
            s.U[1][c]=rho; s.U[2][c]=rho*vx; s.U[3][c]=0; s.U[4][c]=0
            s.U[5][c]=1f0/(gamma-1f0)+0.5f0*rho*vx*vx
            s.U[6][c]=0; s.U[7][c]=0; s.U[8][c]=0; s.U[9][c]=0
        end
        ws=allocate_radiation_workspace(s)
        @test length(ws.packed_psi_correction)==8*N^3
        @test all(r -> all(iszero,r),ws.magnetic_residual)
        photons=allocate_photon_moment_state(s)
        initialize_photon_moments!(photons,ws,s)
        rc=RadiationClosure(inertia_ratio=0.2f0,mean_free_path=0.001f0,
                            photon_speed=20f0,gas_sound_speed=1f0,
                            response_model=:moments)
        e0=photon_moment_energy(photons,ws,s,rc)
        p0=copy(Array(photons.hat_batch))
        ledger=RadiationEnergyLedger()
        step_radiation_godunov!(s,ws,2f-4;drag_impulse=7f0*2f-4,
                                closure=rc,ch=1f0,integrator=:ref,
                                photon_state=photons,energy_ledger=ledger)
        e1=photon_moment_energy(photons,ws,s,rc)
        h=fields_to_host(s)
        @test all(all(isfinite,x) for x in h)
        @test minimum(h[1])>0
        @test isfinite(e0) && e0>0
        @test isfinite(e1) && e1>0
        @test maximum(abs,Array(photons.hat_batch).-p0)>0
        @test ledger.samples==1
        @test ledger.photon_change ≈ e1-e0 rtol=2f-5 atol=1f-8
        @test ledger.resolved_coupled_change ≈
              ledger.total_change+ledger.photon_change rtol=2f-6 atol=1f-9
        @test ledger.component_residual_abs >= abs(ledger.component_residual)
        @test ledger.component_magnitude_abs >= ledger.component_residual_abs

        missing=allocate_state(backend(:cpu),Float32,(8,8,8);dx=1/8)
        mw=allocate_radiation_workspace(missing)
        @test_throws ErrorException step_radiation_godunov!(missing,mw,1f-4;
            drag_impulse=1f-3,closure=rc,ch=1f0,integrator=:ref)
    end

    @testset "gated free-streaming moment path recovers spectral update" begin
        N=8
        full=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N,riemann=:hll)
        fast=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N,riemann=:hll)
        init_turb_field!(full)
        for v in 1:9
            copyto!(fast.U[v],full.U[v])
        end
        wf=allocate_radiation_workspace(full)
        wa=allocate_radiation_workspace(fast)
        pf=allocate_photon_moment_state(full)
        pa=allocate_photon_moment_state(fast)
        initialize_photon_moments!(pf,wf,full)
        copyto!(pa.hat_batch,pf.hat_batch)
        rc=RadiationClosure(inertia_ratio=0.3f0,mean_free_path=1f4,
                            photon_speed=20f0,gas_sound_speed=1f0,
                            response_model=:moments)
        dt,smax=compute_dt(full;cfl=0.01)
        impulse=0.2f0
        bound=radiation_free_streaming_bound(2f0*Float32(pi),impulse/dt,rc)
        @test bound < 1f-3
        tf=RadiationStepTiming()
        ta=RadiationStepTiming()
        la=RadiationEnergyLedger()
        step_radiation_godunov!(full,wf,dt;drag_impulse=impulse,closure=rc,
                                ch=smax,integrator=:ref,photon_state=pf,
                                timing=tf)
        step_radiation_godunov!(fast,wa,dt;drag_impulse=impulse,closure=rc,
                                ch=smax,integrator=:ref,photon_state=pa,
                                timing=ta,energy_ledger=la,
                                free_streaming_tolerance=1f-3)
        hf=fields_to_host(full)
        ha=fields_to_host(fast)
        gas_error=maximum(maximum(abs,ha[v].-hf[v]) for v in 1:9)
        photon_scale=max(maximum(abs,Array(pf.hat_batch)),1f-20)
        photon_error=maximum(abs,Array(pa.hat_batch).-Array(pf.hat_batch))/
                     photon_scale
        @test gas_error < 4f-5
        @test photon_error < 2f-3
        @test tf.spectral_steps==1 && tf.fast_steps==0
        @test ta.fast_steps==1 && ta.spectral_steps==0
        @test la.samples==1
        @test la.component_residual_abs <
              2f-5*max(abs(la.total_change),1f-8)

        coupled=RadiationClosure(inertia_ratio=0.3f0,mean_free_path=0.01f0,
                                 photon_speed=20f0,gas_sound_speed=1f0,
                                 response_model=:moments)
        @test radiation_free_streaming_bound(2f0*Float32(pi),
                                             impulse/dt,coupled)>1f-3
    end

    @testset "checkpoint fallback rejects without accepting the trial" begin
        N=8
        s=allocate_state(backend(:cpu),Float32,(N,N,N);dx=1/N,riemann=:hll)
        init_turb_field!(s)
        ws=allocate_radiation_workspace(s)
        before=fields_to_host(s)
        for v in 1:9
            copyto!(s.scratch[v],s.U[v])
        end
        fill!(s.U[1],-1f-4)
        err=try
            MHDKernels._checkpoint_fallback_rejection!(s,ws,true;
                dt=1f-7,debug_context="unit-test",trial_rho_min=-1f-4,
                trial_wave_speed=0f0,wave_limit=1f0)
            nothing
        catch caught
            caught
        end
        @test err isa ErrorException
        @test occursin("RADIATION_CHECKPOINT_CFL_FALLBACK",sprint(showerror,err))
        after=fields_to_host(s)
        @test all(after[v]==before[v] for v in 1:9)
        expected_delta=reshape(before[1],s.dims).-ws.rho_mean
        @test Array(ws.density_perturbation)==expected_delta
    end
end
