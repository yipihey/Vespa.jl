function _drag_alfven_exact(t,gamma,kva)
    half=0.5*gamma
    if gamma == 0
        return cos(kva*t)
    elseif half < kva
        mu=sqrt(kva*kva-half*half)
        return exp(-half*t)*(cos(mu*t)+half/mu*sin(mu*t))
    elseif half > kva
        mu=sqrt(half*half-kva*kva)
        rslow=-half+mu; rfast=-half-mu
        return (-rfast*exp(rslow*t)+rslow*exp(rfast*t))/(rslow-rfast)
    end
    exp(-half*t)*(1+half*t)
end

function _drag_alfven_run(N; gamma_drag,dt=0.002,tfinal=0.1,integrator=:ref)
    T=Float32; gamma_gas=T(5/3); rho=T(1); p0=T(1); B0=T(1); amp=T(1e-3)
    s=allocate_state(backend(:cpu),T,(N,4,4);dx=1/N,gamma=gamma_gas,
                     recon=:ppm,riemann=:hlld)
    for k3 in 1:4,j in 1:4,i in 1:N
        c=i+(j-1)*N+(k3-1)*4N
        x=T((i-0.5)/N); by=amp*cos(T(2pi)*x)
        s.U[1][c]=rho; s.U[2][c]=0; s.U[3][c]=0; s.U[4][c]=0
        s.U[6][c]=B0; s.U[7][c]=by; s.U[8][c]=0; s.U[9][c]=0
        s.U[5][c]=p0/(gamma_gas-one(T))+T(0.5)*(B0*B0+by*by)
    end
    t=0.0
    while t < tfinal-1e-14
        dtn=min(dt,tfinal-t)
        step_drag_godunov!(s,dtn;drag_impulse=gamma_drag*dtn,ch=1.5,
                           glm_cr=0,integrator=integrator)
        t+=dtn
    end
    by=fields_to_host(s)[7]
    measured=0.0
    for i in 1:N
        measured+=Float64(by[i])*cos(2pi*(i-0.5)/N)
    end
    measured*=2/N
    exact=Float64(amp)*_drag_alfven_exact(t,gamma_drag,2pi*Float64(B0)/sqrt(Float64(rho)))
    measured,exact,s
end

@testset "source-aware Godunov follows analytic drag-damped Alfven mode" begin
    for gamma_drag in (0.0,100.0,1000.0)
        measured,exact,s=_drag_alfven_run(128;gamma_drag=gamma_drag)
        rel=abs(measured-exact)/1e-3
        @printf("  drag-Alfven Gamma=%7.1f: By=%.8e exact=%.8e abs/A=%.3e\n",
                gamma_drag,measured,exact,rel)
        @test all(isfinite,fields_to_host(s)[5])
        @test rel < (gamma_drag == 0 ? 0.015 : 0.025)
    end
end
