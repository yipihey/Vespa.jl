export RadiationClosure, RadiationWorkspace, PhotonMomentState,
       allocate_radiation_workspace, allocate_photon_moment_state,
       initialize_photon_moments!, reconstruct_radiation_density!, photon_moment_energy,
       RadiationEnergyLedger, RadiationStepTiming, radiation_energy_components,
       step_radiation_godunov!, radiation_angular_response,
       radiation_mode_coefficients, radiation_free_streaming_bound

const _RADIATION_RESPONSE_BRIDGE = Int32(0)
const _RADIATION_RESPONSE_ANGULAR = Int32(1)
const _RADIATION_RESPONSE_MOMENTS = Int32(2)
const _RADIATION_RESPONSE_MOMENTS_EXPONENTIAL = Int32(3)
const _PHOTON_EXPONENTIAL_SUBSTEPS = 32

@inline function _radiation_response_code(model::Symbol)
    model === :bridge && return _RADIATION_RESPONSE_BRIDGE
    model in (:angular, :transport) && return _RADIATION_RESPONSE_ANGULAR
    model in (:moments, :photon_moments) && return _RADIATION_RESPONSE_MOMENTS
    model in (:moments_exponential, :photon_moments_exponential) &&
        return _RADIATION_RESPONSE_MOMENTS_EXPONENTIAL
    error("radiation response_model must be :bridge, :angular, :moments, or " *
          ":moments_exponential")
end

@inline _radiation_response_name(code::Int32) =
    code == _RADIATION_RESPONSE_MOMENTS_EXPONENTIAL ? :moments_exponential :
    code == _RADIATION_RESPONSE_MOMENTS ? :moments :
    code == _RADIATION_RESPONSE_ANGULAR ? :angular : :bridge

@inline _radiation_evolves_moments(code::Int32) =
    code == _RADIATION_RESPONSE_MOMENTS ||
    code == _RADIATION_RESPONSE_MOMENTS_EXPONENTIAL

"""
    RadiationClosure(; inertia_ratio, mean_free_path, photon_speed,
                     gas_sound_speed, bridge_q2=5, longitudinal_viscosity=4/3)

Leading-order photon-baryon response used by `step_radiation_godunov!`.
Lengths and velocities use the MHD state's code units. `inertia_ratio` is
`R=3 rho_b c^2/(4 rho_gamma)`. The closure is asymptotically matched between
the tight-coupling diffusion limit and the local Compton-drag limit. The
transition kernel is deliberately exposed because it is a calibration surface,
not an exact Boltzmann transport solution.
"""
struct RadiationClosure{T}
    inertia_ratio::T
    mean_free_path::T
    photon_speed::T
    gas_sound_speed::T
    bridge_q2::T
    longitudinal_viscosity::T
    nyquist_guard_fraction::T
    response_model::Int32
end

RadiationClosure{T}(R,l,c,cs,q2,lvisc) where {T} =
    RadiationClosure{T}(R,l,c,cs,q2,lvisc,one(T),_RADIATION_RESPONSE_BRIDGE)
RadiationClosure{T}(R,l,c,cs,q2,lvisc,guard) where {T} =
    RadiationClosure{T}(R,l,c,cs,q2,lvisc,guard,_RADIATION_RESPONSE_BRIDGE)

function RadiationClosure(; inertia_ratio::Real, mean_free_path::Real,
                          photon_speed::Real, gas_sound_speed::Real,
                          bridge_q2::Real=5, longitudinal_viscosity::Real=4/3,
                          nyquist_guard_fraction::Real=1,
                          response_model::Symbol=:bridge)
    vals = promote(float(inertia_ratio), float(mean_free_path), float(photon_speed),
                   float(gas_sound_speed), float(bridge_q2),
                   float(longitudinal_viscosity),float(nyquist_guard_fraction))
    R,lambda,c,cs,q2,lvisc,guard = vals
    R > 0 || error("radiation inertia_ratio R must be positive")
    lambda >= 0 || error("radiation mean_free_path must be non-negative")
    c > 0 || error("radiation photon_speed must be positive")
    cs >= 0 || error("radiation gas_sound_speed must be non-negative")
    q2 > 0 || error("radiation bridge_q2 must be positive")
    lvisc > 0 || error("radiation longitudinal_viscosity must be positive")
    0 < guard <= 1 || error("radiation nyquist_guard_fraction must be in (0,1]")
    return RadiationClosure(R,lambda,c,cs,q2,lvisc,guard,
                            _radiation_response_code(response_model))
end

struct RadiationWorkspace{T,A,V,H,C,B,D,L,S}
    packed_psi_correction::V
    real_batch::B
    fft_real::A
    correction::NTuple{3,A}
    density_perturbation::D
    rho_mean::T
    hat_batch::H
    velocity_hat::NTuple{3,C}
    density_hat::C
    density_limiter_active::L
    density_limiter_min_theta::S
end

@kernel function _radiation_initialize_density_k!(delta,@Const(rho),rho_mean)
    c=@index(Global)
    @inbounds delta[c]=rho[c]-rho_mean
end

@kernel function _radiation_reconstruct_density_k!(rho,@Const(delta),rho_mean,smallr)
    c=@index(Global)
    @inbounds rho[c]=max(rho_mean+delta[c],smallr)
end

"""Reconstruct the compatibility density field from the authoritative perturbation."""
function reconstruct_radiation_density!(ws::RadiationWorkspace,s::MHDState)
    _radiation_reconstruct_density_k!(s.be)(s.U[1],ws.density_perturbation,
        ws.rho_mean,s.smallr;ndrange=ncells(s))
    KA.synchronize(s.be)
    s
end

function _radiation_refresh_density!(ws::RadiationWorkspace,s::MHDState)
    _radiation_initialize_density_k!(s.be)(ws.density_perturbation,s.U[1],
        ws.rho_mean;ndrange=ncells(s))
    KA.synchronize(s.be)
    s
end

"""
    PhotonMomentState

Persistent linear photon moments in the RFFT half spectrum. `delta_hat` is the
fractional photon-energy perturbation and `velocity_hat` is the photon bulk
velocity in the same code units as the gas. The anisotropic stress is eliminated
with the angular closure in `_photon_stress_rates`; no additional full-grid
quadrupole arrays are required.
"""
struct PhotonMomentState{H,C}
    hat_batch::H
    velocity_hat::NTuple{3,C}
    delta_hat::C
    dims::NTuple{3,Int}
end

mutable struct RadiationEnergyLedger
    samples::Int
    total_change::Float64
    kinetic_change::Float64
    internal_change::Float64
    magnetic_change::Float64
    photon_change::Float64
    resolved_coupled_change::Float64
    component_residual::Float64
    component_residual_abs::Float64
end

RadiationEnergyLedger()=RadiationEnergyLedger(0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

mutable struct RadiationStepTiming
    predictor_s::Float64
    godunov_s::Float64
    corrector_s::Float64
    ledger_s::Float64
    spectral_steps::Int
    fast_steps::Int
    fallback_steps::Int
    plm_hll_fallback_steps::Int
    subdivision_steps::Int
    global_fallback_steps::Int
    density_limited_stages::Int
end

RadiationStepTiming()=RadiationStepTiming(0.0,0.0,0.0,0.0,0,0,0,0,0,0,0)

@inline function _time_radiation_stage!(f, timing::Nothing, field::Symbol)
    f()
end

@inline function _time_radiation_stage!(f, timing::RadiationStepTiming, field::Symbol)
    result=nothing
    elapsed=@elapsed result=f()
    setfield!(timing,field,getfield(timing,field)+elapsed)
    result
end

function allocate_radiation_workspace(s::MHDState{T}) where {T}
    all(==(s.dims[1]), s.dims) ||
        error("spectral radiation coupling currently requires a cubic grid")
    T === Float32 || s.be isa CPU ||
        error("device spectral radiation coupling currently supports Float32")
    n = s.dims[1]
    packed=device_zeros(s.be,T,(5*n^3,))
    real_batch=reshape(view(packed,1:4*n^3),n,n,n,4)
    fft_real=reshape(view(packed,1:n^3),n,n,n)
    corr=ntuple(d -> reshape(view(packed,d*n^3+1:(d+1)*n^3),n,n,n),3)
    density_perturbation=reshape(view(packed,4*n^3+1:5*n^3),s.dims)
    hdims = (n ÷ 2 + 1, n, n)
    hat_batch=device_zeros(s.be,Complex{T},(hdims...,4))
    density_hat=reshape(view(hat_batch,:,:,:,1),hdims)
    hats=ntuple(d -> reshape(view(hat_batch,:,:,:,d+1),hdims),3)
    rho_mean=T(sum(s.U[1])/length(s.U[1]))
    _radiation_initialize_density_k!(s.be)(density_perturbation,s.U[1],rho_mean;
                                            ndrange=ncells(s))
    KA.synchronize(s.be)
    RadiationWorkspace(packed,real_batch,fft_real,corr,density_perturbation,
                       rho_mean,hat_batch,hats,density_hat,Ref(false),Ref(1.0))
end

function _radiation_trial_admissible(s::MHDState)
    rho_min=Float64(minimum(s.U[1]))
    isfinite(rho_min) && rho_min>=max(Float64(s.smallr),Float64(s.llf_dmin))
end

function _restore_radiation_trial!(s::MHDState,ws::RadiationWorkspace,
                                   track_density::Bool)
    s.U,s.scratch=s.scratch,s.U
    if track_density
        _radiation_initialize_density_k!(s.be)(ws.density_perturbation,s.U[1],
            ws.rho_mean;ndrange=ncells(s))
        KA.synchronize(s.be)
    end
    s
end

function _checkpoint_fallback_rejection!(s::MHDState,ws::RadiationWorkspace,
        track_density::Bool;dt,debug_context,trial_rho_min,trial_wave_speed,
        wave_limit)
    _restore_radiation_trial!(s,ws,track_density)
    error("RADIATION_CHECKPOINT_CFL_FALLBACK: rejected coupled " *
          "trial dt=$dt $debug_context trial_rho_min=$trial_rho_min " *
          "trial_wave_speed=$trial_wave_speed wave_limit=$wave_limit")
end

function allocate_photon_moment_state(s::MHDState{T}) where {T}
    all(==(s.dims[1]),s.dims) ||
        error("spectral photon moments currently require a cubic grid")
    n=s.dims[1]; hdims=(n÷2+1,n,n)
    hats=device_zeros(s.be,Complex{T},(hdims...,4))
    delta=reshape(view(hats,:,:,:,1),hdims)
    vel=ntuple(d->reshape(view(hats,:,:,:,d+1),hdims),3)
    PhotonMomentState(hats,vel,delta,s.dims)
end

@kernel function _radiation_energy_components_k!(kinetic,internal,magnetic,total,
        @Const(rho),@Const(mx),@Const(my),@Const(mz),@Const(E),
        @Const(Bx),@Const(By),@Const(Bz),smallr)
    c=@index(Global)
    @inbounds begin
        T=eltype(total); r=max(rho[c],smallr)
        ek=T(0.5)*(mx[c]*mx[c]+my[c]*my[c]+mz[c]*mz[c])/r
        eb=T(0.5)*(Bx[c]*Bx[c]+By[c]*By[c]+Bz[c]*Bz[c])
        kinetic[c]=ek; magnetic[c]=eb; internal[c]=E[c]-ek-eb; total[c]=E[c]
    end
end

"""Return device-reduced MHD energy components without allocating full-grid temporaries."""
function radiation_energy_components(ws::RadiationWorkspace,s::MHDState)
    kinetic=view(ws.real_batch,:,:,:,1); internal=view(ws.real_batch,:,:,:,2)
    magnetic=view(ws.real_batch,:,:,:,3); total=view(ws.real_batch,:,:,:,4)
    _radiation_energy_components_k!(s.be)(kinetic,internal,magnetic,total,
        s.U[1],s.U[2],s.U[3],s.U[4],s.U[5],s.U[6],s.U[7],s.U[8],s.smallr;
        ndrange=ncells(s))
    KA.synchronize(s.be)
    return (total=Float64(sum(total)),kinetic=Float64(sum(kinetic)),
            internal=Float64(sum(internal)),magnetic=Float64(sum(magnetic)))
end

@kernel function _radiation_energy_changes_k!(dkinetic,dinternal,dmagnetic,dtotal,
        @Const(rho),@Const(mx),@Const(my),@Const(mz),@Const(E),
        @Const(Bx),@Const(By),@Const(Bz),
        @Const(rho0),@Const(mx0),@Const(my0),@Const(mz0),@Const(E0),
        @Const(Bx0),@Const(By0),@Const(Bz0),smallr)
    c=@index(Global)
    @inbounds begin
        T=eltype(dtotal); r=max(rho[c],smallr); r0=max(rho0[c],smallr)
        ek=T(0.5)*(mx[c]*mx[c]+my[c]*my[c]+mz[c]*mz[c])/r
        ek0=T(0.5)*(mx0[c]*mx0[c]+my0[c]*my0[c]+mz0[c]*mz0[c])/r0
        eb=T(0.5)*(Bx[c]*Bx[c]+By[c]*By[c]+Bz[c]*Bz[c])
        eb0=T(0.5)*(Bx0[c]*Bx0[c]+By0[c]*By0[c]+Bz0[c]*Bz0[c])
        dkinetic[c]=ek-ek0; dmagnetic[c]=eb-eb0
        dinternal[c]=(E[c]-ek-eb)-(E0[c]-ek0-eb0)
        dtotal[c]=E[c]-E0[c]
    end
end

@kernel function _radiation_energy_residuals_k!(signed_residual,absolute_residual,
        @Const(dkinetic),@Const(dinternal),@Const(dmagnetic),@Const(dtotal))
    c=@index(Global)
    @inbounds begin
        residual=dtotal[c]-(dkinetic[c]+dinternal[c]+dmagnetic[c])
        signed_residual[c]=residual
        absolute_residual[c]=abs(residual)
    end
end

function _radiation_energy_changes(ws::RadiationWorkspace,s::MHDState,old)
    dk=view(ws.real_batch,:,:,:,1); di=view(ws.real_batch,:,:,:,2)
    db=view(ws.real_batch,:,:,:,3); dt=view(ws.real_batch,:,:,:,4)
    _radiation_energy_changes_k!(s.be)(dk,di,db,dt,
        s.U[1],s.U[2],s.U[3],s.U[4],s.U[5],s.U[6],s.U[7],s.U[8],
        old[1],old[2],old[3],old[4],old[5],old[6],old[7],old[8],s.smallr;
        ndrange=ncells(s))
    KA.synchronize(s.be)
    change=(total=Float64(sum(dt)),kinetic=Float64(sum(dk)),
            internal=Float64(sum(di)),magnetic=Float64(sum(db)))
    # Form the identity residual per cell before reducing. On Metal, subtracting
    # four independently reduced Float32 totals loses all useful digits for a
    # nearly uniform state even though the pointwise decomposition is accurate.
    _radiation_energy_residuals_k!(s.be)(dt,dk,dk,di,db,dt;
                                       ndrange=ncells(s))
    KA.synchronize(s.be)
    merge(change,(component_residual=Float64(sum(dt)),
                  component_residual_abs=Float64(sum(dk))))
end

function _record_radiation_energy!(ledger::RadiationEnergyLedger,change;
                                   photon_change=0.0)
    component_residual=hasproperty(change,:component_residual) ?
        change.component_residual :
        change.total-(change.kinetic+change.internal+change.magnetic)
    component_residual_abs=hasproperty(change,:component_residual_abs) ?
        change.component_residual_abs : abs(component_residual)
    ledger.samples+=1
    ledger.total_change+=change.total
    ledger.kinetic_change+=change.kinetic
    ledger.internal_change+=change.internal
    ledger.magnetic_change+=change.magnetic
    ledger.photon_change+=photon_change
    ledger.resolved_coupled_change+=change.total+photon_change
    ledger.component_residual+=component_residual
    ledger.component_residual_abs+=component_residual_abs
    ledger
end

function _radiation_forward_batch!(hat_batch,real_batch)
    for d in axes(real_batch,4)
        _radiation_forward!(view(hat_batch,:,:,:,d),view(real_batch,:,:,:,d))
    end
    hat_batch
end

function _radiation_inverse_batch!(real_batch,hat_batch)
    for d in axes(real_batch,4)
        _radiation_inverse!(view(real_batch,:,:,:,d),view(hat_batch,:,:,:,d))
    end
    real_batch
end

function _radiation_forward_triplet!(ws::RadiationWorkspace,fields)
    for d in 1:3
        copyto!(ws.correction[d],fields[d])
    end
    fill!(ws.fft_real,zero(eltype(ws.fft_real)))
    _radiation_forward_batch!(ws.hat_batch,ws.real_batch)
    ws.velocity_hat
end

function _radiation_inverse_triplet!(fields,ws::RadiationWorkspace)
    _radiation_inverse_batch!(ws.real_batch,ws.hat_batch)
    for d in 1:3
        fields[d] === ws.correction[d] || copyto!(fields[d],ws.correction[d])
    end
    fields
end

const _RADIATION_FFT_CACHE = Dict{Any,Any}()

function _radiation_fft_plans(x::AbstractArray{T,3}, chat::AbstractArray{Complex{T},3}) where {T}
    get!(_RADIATION_FFT_CACHE, (T,size(x))) do
        (plan_rfft(x), plan_irfft(chat, size(x,1)))
    end
end

function _radiation_forward!(chat::AbstractArray{Complex{T},3}, x::AbstractArray{T,3}) where {T}
    fwd,_ = _radiation_fft_plans(x,chat)
    FFTW.mul!(chat,fwd,x)
    chat
end

function _radiation_inverse!(x::AbstractArray{T,3}, chat::AbstractArray{Complex{T},3}) where {T}
    _,invp = _radiation_fft_plans(x,chat)
    FFTW.mul!(x,invp,chat)
    x
end

@kernel function _photon_initial_fields_k!(delta,vx,vy,vz,
        @Const(rho),@Const(rho_perturbation),@Const(mx),@Const(my),@Const(mz),
        rho_mean,smallr,
        adiabatic::Val{ADIABATIC}) where {ADIABATIC}
    c=@index(Global)
    @inbounds begin
        r=max(rho[c],smallr)
        delta[c]=ADIABATIC ? typeof(r)(4)/typeof(r)(3)*rho_perturbation[c]/rho_mean : zero(r)
        vx[c]=mx[c]/r; vy[c]=my[c]/r; vz[c]=mz[c]/r
    end
end

"""
    initialize_photon_moments!(photons, ws, s; adiabatic_density=true,
                               comoving_velocity=true)

Initialize photons from the gas. Adiabatic initial conditions use
`delta_gamma = 4 delta_b / 3`; disabling them starts from a uniform photon
monopole. Photon and baryon velocities are initially comoving unless
`comoving_velocity=false`.
"""
function initialize_photon_moments!(photons::PhotonMomentState,
                                    ws::RadiationWorkspace,s::MHDState;
                                    adiabatic_density::Bool=true,
                                    comoving_velocity::Bool=true)
    photons.dims==s.dims || error("photon and MHD grid dimensions differ")
    rho_mean=ws.rho_mean
    _photon_initial_fields_k!(s.be)(view(ws.real_batch,:,:,:,1),
        view(ws.real_batch,:,:,:,2),view(ws.real_batch,:,:,:,3),
        view(ws.real_batch,:,:,:,4),s.U[1],ws.density_perturbation,
        s.U[2],s.U[3],s.U[4],rho_mean,s.smallr,
        Val(adiabatic_density);ndrange=ncells(s))
    KA.synchronize(s.be)
    comoving_velocity || fill!(view(ws.real_batch,:,:,:,2:4),zero(eltype(s.U[1])))
    _radiation_forward_batch!(photons.hat_batch,ws.real_batch)
    photons
end

@kernel function _photon_energy_hat_k!(out,@Const(delta),@Const(vx),@Const(vy),
        @Const(vz),n::Int32,rho_mean,R,cph)
    p=@index(Global)
    h=(n>>>1)+Int32(1); ii=Int32(p-1)%h
    @inbounds begin
        T=eltype(out)
        w=(ii==0 || (iszero(n&Int32(1)) && ii==(n>>>1))) ? one(T) : T(2)
        v2=abs2(vx[p])+abs2(vy[p])+abs2(vz[p])
        # Photon inertia is rho_b/R. The monopole coefficient follows from
        # ddelta_gamma/dt=-(4/3)div(v_gamma) and
        # dv_gamma/dt=-(c^2/4)grad(delta_gamma).
        out[p]=w*rho_mean/R*(T(0.5)*v2+T(3)/T(32)*cph*cph*abs2(delta[p]))
    end
end

"""Return the quadratic photon perturbation energy represented by the moment state."""
function photon_moment_energy(photons::PhotonMomentState,ws::RadiationWorkspace,
                              s::MHDState,closure::RadiationClosure)
    n=s.dims[1]; nh=length(photons.delta_hat)
    out=reshape(view(ws.packed_psi_correction,1:nh),size(photons.delta_hat))
    T=eltype(s.U[1]); rho_mean=ws.rho_mean
    _photon_energy_hat_k!(s.be)(out,photons.delta_hat,photons.velocity_hat...,
        Int32(n),rho_mean,T(closure.inertia_ratio),T(closure.photon_speed);
        ndrange=nh)
    KA.synchronize(s.be)
    Float64(sum(out))/Float64(n^3)
end

@inline function _radiation_acoustic_coefficients(g::T,c2::T,k2::T,dt::T) where {T}
    halfg=T(0.5)*g
    disc=halfg*halfg-c2*k2
    scale=max(one(T),halfg*halfg+c2*k2)
    if abs(disc) <= T(16)*eps(T)*scale
        e=exp(-halfg*dt)
        C=e
        S=dt*e
    elseif disc > zero(T)
        mu=sqrt(disc)
        # Form the slow root without subtracting nearly equal Float32 values.
        # The direct expression can erase diffusion damping in the stiff limit.
        omega2=c2*k2
        rslow=-omega2/(halfg+mu)
        rfast=-halfg-mu
        e1=exp(rslow*dt)
        e2=exp(rfast*dt)
        S=(e1-e2)/(T(2)*mu)
        B=(rslow*e1-rfast*e2)/(T(2)*mu)
        return B,S
    else
        omega=sqrt(-disc)
        e=exp(-halfg*dt)
        C=e*cos(omega*dt)
        S=e*sin(omega*dt)/omega
    end
    return C-halfg*S,S
end

@inline function _radiation_integral_S(B::T,S::T,g::T,omega2::T,dt::T) where {T}
    if omega2 > T(32)*eps(T)*max(one(T),g*g)
        return (one(T)-B-g*S)/omega2
    end
    x=g*dt
    if abs(x) < T(1e-3)
        return dt*dt*(T(0.5)-x/T(6)+x*x/T(24))
    elseif g > zero(T)
        return (x+expm1(-x))/(g*g)
    end
    T(0.5)*dt*dt
end

@inline function _radiation_mode_coefficients(k2::T, gamma_drag::T,
                                               rc::RadiationClosure{T}) where {T}
    q2=rc.mean_free_path*rc.mean_free_path*k2
    if rc.response_model == _RADIATION_RESPONSE_ANGULAR ||
            _radiation_evolves_moments(rc.response_model)
        a0,along,atrans=_radiation_angular_response(q2)
        stream=one(T)-atrans
        inertia=rc.inertia_ratio/(rc.inertia_ratio+atrans)
        gamma_t=inertia*gamma_drag*stream
        # The transverse response is the exact static angular integral.  The
        # longitudinal viscosity factor retains the diffusion-limit 4/3
        # coefficient and approaches local drag as photons free-stream.
        gamma_l=gamma_t*(rc.longitudinal_viscosity*atrans+stream)
        # along^2 gives the tight-coupling EOS and removes its restoring
        # frequency in the optically thin limit.  This is the remaining
        # longitudinal closure approximation, exposed by response_model.
        crad2=rc.photon_speed*rc.photon_speed*along*along/
              (T(3)*(rc.inertia_ratio+along))
        return stream,inertia,gamma_t,gamma_l,crad2
    end
    # Equation 57 of Jedamzik et al. is a diffusion-limit EOS. In the
    # free-streaming system photon pressure is absent and scattering supplies
    # only local drag. Use a C1 compact bridge so c^2 times a slowly decaying
    # rational tail cannot create a fictitious optically thin sound speed.
    x=min(q2/rc.bridge_q2,one(T))
    stream=x*x*(T(3)-T(2)*x)
    tight=one(T)-stream
    inertia_tight=rc.inertia_ratio/(one(T)+rc.inertia_ratio)
    inertia=inertia_tight+(one(T)-inertia_tight)*stream
    gdiff=rc.photon_speed*rc.mean_free_path*k2/(T(5)*(one(T)+rc.inertia_ratio))
    # Diffusion is an optically thick expansion and must switch off faster than
    # k^2 grows; a second tight-coupling weight prevents a spurious constant
    # damping tail in the free-streaming limit.
    gamma_t=tight*tight*gdiff+stream*gamma_drag
    gamma_l=tight*tight*rc.longitudinal_viscosity*gdiff+stream*gamma_drag
    crad2=tight*rc.photon_speed*rc.photon_speed/
           (T(3)*(one(T)+rc.inertia_ratio))
    return stream,inertia,gamma_t,gamma_l,crad2
end

@inline function _radiation_angular_response(q2::T) where {T}
    if q2 < T(1e-3)
        q4=q2*q2; q6=q4*q2
        a0=one(T)-q2/T(3)+q4/T(5)-q6/T(7)
        along=one(T)-T(3)*q2/T(5)+T(3)*q4/T(7)-q6/T(3)
        atrans=one(T)-q2/T(5)+T(3)*q4/T(35)-q6/T(21)
        return a0,along,atrans
    end
    q=sqrt(q2)
    a0=atan(q)/q
    along=T(3)*(one(T)-a0)/q2
    atrans=T(0.5)*(T(3)*a0-along)
    return a0,along,atrans
end

@inline function _photon_stress_rates(k2::T,nu::T,rc::RadiationClosure{T}) where {T}
    q2=rc.mean_free_path*rc.mean_free_path*k2
    _,_,atrans=_radiation_angular_response(q2)
    # Eliminating the angular stress with nu*(A_T^-1-1) reproduces the exact
    # static transverse transport response. It tends to c*lambda*k^2/5 in
    # diffusion and to O(c*k) phase mixing in free streaming. Longitudinal
    # shear has the standard 4/3 enhancement only in the diffusion limit;
    # the evolved photon monopole supplies the longitudinal pressure memory.
    # Avoid losing the q^2/5 diffusion term when A_T rounds to one in f32.
    dtrans=q2<T(1e-3) ? q2/T(5)-T(8)*q2*q2/T(175) :
                         max(inv(atrans)-one(T),zero(T))
    st=nu*dtrans
    sl=st*(rc.longitudinal_viscosity*atrans+(one(T)-atrans))
    return st,sl
end

@inline function _photon_solve_transverse(rb,rg,a,G,nu,stress)
    T=typeof(a); a11=one(T)+a*G; a22=one(T)+a*(nu+stress)
    det=a11*a22-a*a*G*nu
    return ((a22*rb+a*G*rg)/det,(a*nu*rb+a11*rg)/det)
end

@inline function _photon_solve_longitudinal(rb,ru,rg,rug,a,G,nu,stress,k,cs2,cph2)
    T=typeof(a); ik=Complex{T}(zero(T),k)
    a11=one(T)+a*G+a*a*k*k*cs2
    a22=one(T)+a*(nu+stress)+a*a*k*k*cph2/T(3)
    b1=ru-a*ik*cs2*rb
    b2=rug-a*ik*cph2/T(4)*rg
    det=a11*a22-a*a*G*nu
    ub=(a22*b1+a*G*b2)/det
    ug=(a*nu*b1+a11*b2)/det
    return rb-a*ik*ub,ub,rg-a*T(4)/T(3)*ik*ug,ug
end

@inline function _photon_sdirk_transverse(ub,ug,force,dt,G,nu,stress)
    T=typeof(dt); g=T(1)-inv(sqrt(T(2))); a=g*dt
    y1b,y1g=_photon_solve_transverse(ub+a*force,ug,a,G,nu,stress)
    f1b=G*(y1g-y1b)+force
    f1g=nu*y1b-(nu+stress)*y1g
    rhsb=ub+(one(T)-g)*dt*f1b+a*force
    rhsg=ug+(one(T)-g)*dt*f1g
    _photon_solve_transverse(rhsb,rhsg,a,G,nu,stress)
end

@inline function _photon_sdirk_longitudinal(db,ub,dg,ug,force,dt,G,nu,stress,
                                            k,cs2,cph2)
    T=typeof(dt); g=T(1)-inv(sqrt(T(2))); a=g*dt
    ik=Complex{T}(zero(T),k)
    d1,u1,e1,w1=_photon_solve_longitudinal(db,ub+a*force,dg,ug,a,G,nu,
                                           stress,k,cs2,cph2)
    f1d=-ik*u1
    f1u=-ik*cs2*d1+G*(w1-u1)+force
    f1e=-T(4)/T(3)*ik*w1
    f1w=-ik*cph2/T(4)*e1+nu*u1-(nu+stress)*w1
    rdb=db+(one(T)-g)*dt*f1d
    rub=ub+(one(T)-g)*dt*f1u+a*force
    rdg=dg+(one(T)-g)*dt*f1e
    rug=ug+(one(T)-g)*dt*f1w
    _photon_solve_longitudinal(rdb,rub,rdg,rug,a,G,nu,stress,k,cs2,cph2)
end

@inline function _photon_exact_drag_affine(ub,ug,force,dt,G,nu,stress)
    # Exact action of the affine velocity subsystem
    #   ub' = G(ug-ub) + force
    #   ug' = nu*ub - (nu+stress)*ug.
    # Projecting onto its two real eigenmodes avoids a matrix exponential and,
    # unlike forming h + sqrt(h^2-det), retains the slow diffusion root in f32.
    T=typeof(dt)
    h=-T(0.5)*(G+nu+stress)
    disc=T(0.25)*(nu+stress-G)*(nu+stress-G)+G*nu
    d=sqrt(max(disc,zero(T)))
    den=T(2)*d
    if den <= T(16)*eps(T)*max(one(T),abs(h))
        return ub+dt*force,ug
    end
    rfast=h-d
    determinant=G*stress
    rslow=iszero(rfast) ? h+d : determinant/rfast
    eslow=exp(rslow*dt); efast=exp(rfast*dt)

    slowb=((-G-rfast)*ub+G*ug)/den
    slowg=(nu*ub+(-nu-stress-rfast)*ug)/den
    fastb=((rslow+G)*ub-G*ug)/den
    fastg=(-nu*ub+(rslow+nu+stress)*ug)/den

    islow=dt*_drag_phi1(-rslow*dt)
    ifast=dt*_drag_phi1(-rfast*dt)
    sourceb=(((-G-rfast)*islow+(rslow+G)*ifast)/den)*force
    sourceg=(nu*(islow-ifast)/den)*force
    return eslow*slowb+efast*fastb+sourceb,
           eslow*slowg+efast*fastg+sourceg
end

@inline function _photon_exact_acoustic(db,ub,dt,k,continuity,pressure)
    T=typeof(dt)
    speed2=continuity*pressure
    if speed2 <= zero(T)
        ikdt=Complex{T}(zero(T),-k*dt)
        return db+continuity*ikdt*ub,ub+pressure*ikdt*db
    end
    speed=sqrt(speed2)
    theta=k*speed*dt
    co=cos(theta); si=sin(theta)
    minus_i=Complex{T}(zero(T),-one(T))
    return co*db+minus_i*(continuity/speed)*si*ub,
           co*ub+minus_i*(pressure/speed)*si*db
end

@inline function _photon_split_transverse(ub,ug,force,dt,G,nu,stress)
    _photon_exact_drag_affine(ub,ug,force,dt,G,nu,stress)
end

@inline function _photon_split_longitudinal(db,ub,dg,ug,force,dt,G,nu,stress,
                                            k,cs2,cph2)
    subdt=dt/typeof(dt)(_PHOTON_EXPONENTIAL_SUBSTEPS)
    half=typeof(dt)(0.5)*subdt
    for _ in 1:_PHOTON_EXPONENTIAL_SUBSTEPS
        db,ub=_photon_exact_acoustic(db,ub,half,k,one(typeof(dt)),cs2)
        dg,ug=_photon_exact_acoustic(dg,ug,half,k,typeof(dt)(4)/typeof(dt)(3),
                                     cph2/typeof(dt)(4))
        ub,ug=_photon_exact_drag_affine(ub,ug,force,subdt,G,nu,stress)
        db,ub=_photon_exact_acoustic(db,ub,half,k,one(typeof(dt)),cs2)
        dg,ug=_photon_exact_acoustic(dg,ug,half,k,typeof(dt)(4)/typeof(dt)(3),
                                     cph2/typeof(dt)(4))
    end
    return db,ub,dg,ug
end

@inline function _photon_moment_transverse(ub,ug,force,dt,G,nu,stress,method)
    if method == _RADIATION_RESPONSE_MOMENTS_EXPONENTIAL
        return _photon_split_transverse(ub,ug,force,dt,G,nu,stress)
    end
    _photon_sdirk_transverse(ub,ug,force,dt,G,nu,stress)
end

@inline function _photon_moment_longitudinal(db,ub,dg,ug,force,dt,G,nu,stress,
                                             k,cs2,cph2,method)
    if method == _RADIATION_RESPONSE_MOMENTS_EXPONENTIAL
        return _photon_split_longitudinal(db,ub,dg,ug,force,dt,G,nu,stress,
                                          k,cs2,cph2)
    end
    _photon_sdirk_longitudinal(db,ub,dg,ug,force,dt,G,nu,stress,k,cs2,cph2)
end

@inline function _photon_baseline_homogeneous(db,ub,dt,G,k,cs2)
    B,S=_radiation_acoustic_coefficients(G,cs2,k*k,dt)
    ikdb=Complex{typeof(dt)}(-imag(db),real(db))
    ikub=Complex{typeof(dt)}(-imag(ub),real(ub))
    return (B+G*S)*db-k*S*ikub,B*ub-k*cs2*S*ikdb
end

@inline function _photon_baseline_force(force,dt,G,k,cs2)
    B,S=_radiation_acoustic_coefficients(G,cs2,k*k,dt)
    J=_radiation_integral_S(B,S,G,cs2*k*k,dt)
    iforce=Complex{typeof(dt)}(-imag(force),real(force))
    return -k*J*iforce,S*force
end

"""
    radiation_angular_response(q)

Static angular transport response for an isotropic photon distribution.
`longitudinal` and `transverse` are the photon velocity responses to a baryon
velocity perturbation at dimensionless wavenumber `q = k lambda_mfp`.
"""
function radiation_angular_response(q::Real)
    q >= 0 || error("radiation angular response requires q >= 0")
    T=typeof(float(q))
    a0,along,atrans=_radiation_angular_response(T(q)*T(q))
    (monopole=a0,longitudinal=along,transverse=atrans)
end

function radiation_mode_coefficients(k::Real, gamma_drag::Real,
                                     rc::RadiationClosure)
    T=typeof(rc.inertia_ratio)
    stream,inertia,gt,gl,cr2=_radiation_mode_coefficients(T(k)^2,T(gamma_drag),rc)
    (streaming_weight=stream,inertia_response=inertia,gamma_transverse=gt,
     gamma_longitudinal=gl,radiation_sound_speed2=cr2,
     response_model=_radiation_response_name(rc.response_model))
end

"""
    radiation_free_streaming_bound(kmin, gamma_drag, closure)

Conservative dimensionless bound for replacing the spectral gas response by
local Compton drag when every resolved mode has `k >= kmin`. The bound is the
largest fundamental-mode correction to transverse drag, longitudinal drag, or
radiation pressure relative to gas pressure. It is monotone toward zero in the
free-streaming limit for the angular and evolved-moment closures.
"""
function radiation_free_streaming_bound(kmin::Real, gamma_drag::Real,
                                        rc::RadiationClosure)
    kmin > 0 || error("free-streaming bound requires kmin > 0")
    gamma_drag >= 0 || error("free-streaming bound requires non-negative drag")
    mode=radiation_mode_coefficients(kmin,gamma_drag,rc)
    drag_t=gamma_drag>0 ? abs(one(gamma_drag)-mode.gamma_transverse/gamma_drag) :
                         zero(gamma_drag)
    drag_l=gamma_drag>0 ? abs(one(gamma_drag)-mode.gamma_longitudinal/gamma_drag) :
                         zero(gamma_drag)
    cs2=rc.gas_sound_speed*rc.gas_sound_speed
    pressure=cs2>0 ? mode.radiation_sound_speed2/cs2 :
                     (mode.radiation_sound_speed2==0 ? zero(cs2) : oftype(cs2,Inf))
    max(drag_t,drag_l,pressure)
end

@inline function _radiation_mode_index(i::Int32,n::Int32)
    # A centered spectral derivative has no representable sign at the even-grid
    # Nyquist mode. Zeroing that component preserves the Hermitian constraints
    # required by c2r transforms; assigning either sign corrupts self-conjugate
    # RFFT planes when odd gradients or longitudinal projections are applied.
    i == (n >>> 1) ? Int32(0) : (i < (n >>> 1) ? i : i-n)
end

@inline function _radiation_mode_dealiased(ii::Int32,jj::Int32,kk::Int32,
                                            n::Int32,guard)
    # The Maxwell force entering the spectral correction is a real-space
    # product. An optional Nyquist guard keeps aliased grid-axis modes from
    # feeding back through the exponential/Godunov split.
    aj=min(jj,n-jj); ak=min(kk,n-kk)
    lim=guard*typeof(guard)(n>>>1)
    typeof(guard)(ii) <= lim && typeof(guard)(aj) <= lim &&
        typeof(guard)(ak) <= lim
end

@kernel function _radiation_homogeneous_hat_k!(hx,hy,hz,drho,
        n::Int32,kfac::T,dt::T,gamma_drag::T,inv_rho_mean::T,
        rc::RadiationClosure{T},residual::Val{RES}) where {T,RES}
    p=@index(Global)
    h=(n>>>1)+Int32(1); q=Int32(p-1)
    ii=q%h; q=q÷h; jj=q%n; kk=q÷n
    @inbounds begin
        kx=kfac*T(_radiation_mode_index(ii,n)); ky=kfac*T(_radiation_mode_index(jj,n))
        kz=kfac*T(_radiation_mode_index(kk,n)); k2=kx*kx+ky*ky+kz*kz
        if k2 == zero(T) || !_radiation_mode_dealiased(
                ii,jj,kk,n,rc.nyquist_guard_fraction)
            hx[p]=zero(eltype(hx)); hy[p]=zero(eltype(hy)); hz[p]=zero(eltype(hz))
            drho[p]=zero(eltype(drho))
        else
            k=sqrt(k2); x=hx[p]; y=hy[p]; z=hz[p]
            udot=(kx*x+ky*y+kz*z)/k
            vlx=kx*udot/k; vly=ky*udot/k; vlz=kz*udot/k
            stream,inertia,gt,gl,cr2=_radiation_mode_coefficients(k2,gamma_drag,rc)
            ed=exp(-gt*dt); eb=exp(-gamma_drag*dt)
            cbase2=rc.gas_sound_speed*rc.gas_sound_speed
            Bd,Sd=_radiation_acoustic_coefficients(gl,cbase2+cr2,k2,dt)
            Bb,Sb=_radiation_acoustic_coefficients(gamma_drag,cbase2,k2,dt)
            delta=drho[p]*inv_rho_mean
            idelta=Complex{T}(-imag(delta),real(delta))
            uld=Bd*udot-k*(cbase2+cr2)*Sd*idelta
            ulb=Bb*udot-k*cbase2*Sb*idelta
            du=uld-ulb
            hx[p]=(ed-eb)*(x-vlx)+kx*du/k
            hy[p]=(ed-eb)*(y-vly)+ky*du/k
            hz[p]=(ed-eb)*(z-vlz)+kz*du/k
            if RES
                Ad=Bd+gl*Sd; Ab=Bb+gamma_drag*Sb
                iudot=Complex{T}(-imag(udot),real(udot))
                dexact=(Ad-Ab)*delta-k*(Sd-Sb)*iudot
                Bdh,Sdh=_radiation_acoustic_coefficients(gl,cbase2+cr2,k2,T(0.5)*dt)
                Bbh,Sbh=_radiation_acoustic_coefficients(gamma_drag,cbase2,k2,T(0.5)*dt)
                uldh=Bdh*udot-k*(cbase2+cr2)*Sdh*idelta
                ulbh=Bbh*udot-k*cbase2*Sbh*idelta
                iduh=Complex{T}(-imag(uldh-ulbh),real(uldh-ulbh))
                # The Godunov flux already advances continuity with the
                # corrected midpoint velocity. Add only the exact residual.
                drho[p]=(dexact+k*dt*iduh)/inv_rho_mean
            else
                drho[p]=zero(eltype(drho))
            end
        end
    end
end

@kernel function _radiation_source_hat_k!(hx,hy,hz,drho,
        n::Int32,kfac::T,dt::T,gamma_drag::T,rho_mean::T,
        rc::RadiationClosure{T},residual::Val{RES}) where {T,RES}
    p=@index(Global)
    h=(n>>>1)+Int32(1); q=Int32(p-1)
    ii=q%h; q=q÷h; jj=q%n; kk=q÷n
    @inbounds begin
        kx=kfac*T(_radiation_mode_index(ii,n)); ky=kfac*T(_radiation_mode_index(jj,n))
        kz=kfac*T(_radiation_mode_index(kk,n)); k2=kx*kx+ky*ky+kz*kz
        if k2 == zero(T) || !_radiation_mode_dealiased(
                ii,jj,kk,n,rc.nyquist_guard_fraction)
            hx[p]=zero(eltype(hx)); hy[p]=zero(eltype(hy)); hz[p]=zero(eltype(hz))
            drho[p]=zero(eltype(drho))
        else
            k=sqrt(k2); x=hx[p]; y=hy[p]; z=hz[p]
            adot=(kx*x+ky*y+kz*z)/k
            alx=kx*adot/k; aly=ky*adot/k; alz=kz*adot/k
            stream,inertia,gt,gl,cr2=_radiation_mode_coefficients(k2,gamma_drag,rc)
            phid=dt*_drag_phi1(gt*dt); phib=dt*_drag_phi1(gamma_drag*dt)
            cbase2=rc.gas_sound_speed*rc.gas_sound_speed
            Bd,Sd=_radiation_acoustic_coefficients(gl,cbase2+cr2,k2,dt)
            Bb,Sb=_radiation_acoustic_coefficients(gamma_drag,cbase2,k2,dt)
            ct=inertia*phid-phib; cl=inertia*Sd-Sb
            hx[p]=ct*(x-alx)+kx*(cl*adot)/k
            hy[p]=ct*(y-aly)+ky*(cl*adot)/k
            hz[p]=ct*(z-alz)+kz*(cl*adot)/k
            if RES
                wd=(cbase2+cr2)*k2; wb=cbase2*k2
                Jd=_radiation_integral_S(Bd,Sd,gl,wd,dt)
                Jb=_radiation_integral_S(Bb,Sb,gamma_drag,wb,dt)
                iadot=Complex{T}(-imag(adot),real(adot))
                dexact=-k*(inertia*Jd-Jb)*iadot
                Bdh,Sdh=_radiation_acoustic_coefficients(gl,cbase2+cr2,k2,T(0.5)*dt)
                Bbh,Sbh=_radiation_acoustic_coefficients(gamma_drag,cbase2,k2,T(0.5)*dt)
                iduh=Complex{T}(-imag((inertia*Sdh-Sbh)*adot),
                                real((inertia*Sdh-Sbh)*adot))
                drho[p]=rho_mean*(dexact+k*dt*iduh)
            else
                drho[p]=zero(eltype(drho))
            end
        end
    end
end

@kernel function _photon_moment_homogeneous_hat_k!(hx,hy,hz,drho,
        pgx,pgy,pgz,pdelta,n::Int32,kfac::T,dt::T,G::T,inv_rho_mean::T,
        rc::RadiationClosure{T},residual::Val{RES},update::Val{UPDATE}) where {T,RES,UPDATE}
    p=@index(Global)
    h=(n>>>1)+Int32(1); q=Int32(p-1)
    ii=q%h; q=q÷h; jj=q%n; kk=q÷n
    @inbounds begin
        kx=kfac*T(_radiation_mode_index(ii,n)); ky=kfac*T(_radiation_mode_index(jj,n))
        kz=kfac*T(_radiation_mode_index(kk,n)); k2=kx*kx+ky*ky+kz*kz
        meanmode=ii==0 && jj==0 && kk==0
        if !_radiation_mode_dealiased(ii,jj,kk,n,rc.nyquist_guard_fraction) ||
                (k2==zero(T) && !meanmode)
            hx[p]=zero(eltype(hx)); hy[p]=zero(eltype(hy)); hz[p]=zero(eltype(hz))
            drho[p]=zero(eltype(drho))
        elseif meanmode
            eb=exp(-G*dt); nu=rc.inertia_ratio*G
            xb,xg=_photon_moment_transverse(hx[p],pgx[p],zero(eltype(hx)),dt,G,nu,
                                            zero(T),rc.response_model)
            yb,yg=_photon_moment_transverse(hy[p],pgy[p],zero(eltype(hy)),dt,G,nu,
                                            zero(T),rc.response_model)
            zb,zg=_photon_moment_transverse(hz[p],pgz[p],zero(eltype(hz)),dt,G,nu,
                                            zero(T),rc.response_model)
            hx[p]=xb-eb*hx[p]; hy[p]=yb-eb*hy[p]; hz[p]=zb-eb*hz[p]
            drho[p]=zero(eltype(drho))
            if UPDATE
                pgx[p]=xg; pgy[p]=yg; pgz[p]=zg
            end
        else
            k=sqrt(k2); x=hx[p]; y=hy[p]; z=hz[p]
            gx=pgx[p]; gy=pgy[p]; gz=pgz[p]; gd=pdelta[p]
            ub=(kx*x+ky*y+kz*z)/k; ug=(kx*gx+ky*gy+kz*gz)/k
            blx=kx*ub/k; bly=ky*ub/k; blz=kz*ub/k
            glx=kx*ug/k; gly=ky*ug/k; glz=kz*ug/k
            db=drho[p]*inv_rho_mean
            nu=rc.inertia_ratio*G; st,sl=_photon_stress_rates(k2,nu,rc)
            cs2=rc.gas_sound_speed*rc.gas_sound_speed
            cph2=rc.photon_speed*rc.photon_speed
            dbn,ubn,gdn,ugn=_photon_moment_longitudinal(db,ub,gd,ug,
                zero(eltype(ub)),dt,G,nu,sl,k,cs2,cph2,rc.response_model)
            tx,tgx=_photon_moment_transverse(x-blx,gx-glx,zero(eltype(x)),dt,G,nu,
                                             st,rc.response_model)
            ty,tgy=_photon_moment_transverse(y-bly,gy-gly,zero(eltype(y)),dt,G,nu,
                                             st,rc.response_model)
            tz,tgz=_photon_moment_transverse(z-blz,gz-glz,zero(eltype(z)),dt,G,nu,
                                             st,rc.response_model)
            dbb,ubb=_photon_baseline_homogeneous(db,ub,dt,G,k,cs2)
            ebase=exp(-G*dt)
            hx[p]=kx*ubn/k+tx-(kx*ubb/k+ebase*(x-blx))
            hy[p]=ky*ubn/k+ty-(ky*ubb/k+ebase*(y-bly))
            hz[p]=kz*ubn/k+tz-(kz*ubb/k+ebase*(z-blz))
            if RES
                if rc.response_model == _RADIATION_RESPONSE_MOMENTS_EXPONENTIAL
                    drho[p]=dbn/inv_rho_mean
                else
                    _,ubh,_,_=_photon_moment_longitudinal(db,ub,gd,ug,
                        zero(eltype(ub)),T(0.5)*dt,G,nu,sl,k,cs2,cph2,
                        rc.response_model)
                    _,ubbh=_photon_baseline_homogeneous(db,ub,T(0.5)*dt,G,k,cs2)
                    iduh=Complex{T}(-imag(ubh-ubbh),real(ubh-ubbh))
                    drho[p]=(dbn-dbb+k*dt*iduh)/inv_rho_mean
                end
            else
                drho[p]=zero(eltype(drho))
            end
            if UPDATE
                pgx[p]=kx*ugn/k+tgx; pgy[p]=ky*ugn/k+tgy; pgz[p]=kz*ugn/k+tgz
                pdelta[p]=gdn
            end
        end
    end
end

@kernel function _photon_moment_source_hat_k!(hx,hy,hz,drho,
        pgx,pgy,pgz,pdelta,n::Int32,kfac::T,dt::T,G::T,rho_mean::T,
        rc::RadiationClosure{T},residual::Val{RES},update::Val{UPDATE}) where {T,RES,UPDATE}
    p=@index(Global)
    h=(n>>>1)+Int32(1); q=Int32(p-1)
    ii=q%h; q=q÷h; jj=q%n; kk=q÷n
    @inbounds begin
        kx=kfac*T(_radiation_mode_index(ii,n)); ky=kfac*T(_radiation_mode_index(jj,n))
        kz=kfac*T(_radiation_mode_index(kk,n)); k2=kx*kx+ky*ky+kz*kz
        meanmode=ii==0 && jj==0 && kk==0
        if !_radiation_mode_dealiased(ii,jj,kk,n,rc.nyquist_guard_fraction) ||
                (k2==zero(T) && !meanmode)
            hx[p]=zero(eltype(hx)); hy[p]=zero(eltype(hy)); hz[p]=zero(eltype(hz))
            drho[p]=zero(eltype(drho))
        elseif meanmode
            nu=rc.inertia_ratio*G; phib=dt*_drag_phi1(G*dt)
            xb,xg=_photon_moment_transverse(zero(hx[p]),zero(pgx[p]),hx[p],dt,G,nu,
                                            zero(T),rc.response_model)
            yb,yg=_photon_moment_transverse(zero(hy[p]),zero(pgy[p]),hy[p],dt,G,nu,
                                            zero(T),rc.response_model)
            zb,zg=_photon_moment_transverse(zero(hz[p]),zero(pgz[p]),hz[p],dt,G,nu,
                                            zero(T),rc.response_model)
            hx[p]=xb-phib*hx[p]; hy[p]=yb-phib*hy[p]; hz[p]=zb-phib*hz[p]
            drho[p]=zero(eltype(drho))
            if UPDATE
                pgx[p]+=xg; pgy[p]+=yg; pgz[p]+=zg
            end
        else
            k=sqrt(k2); ax=hx[p]; ay=hy[p]; az=hz[p]
            al=(kx*ax+ky*ay+kz*az)/k
            alx=kx*al/k; aly=ky*al/k; alz=kz*al/k
            nu=rc.inertia_ratio*G; st,sl=_photon_stress_rates(k2,nu,rc)
            cs2=rc.gas_sound_speed*rc.gas_sound_speed
            cph2=rc.photon_speed*rc.photon_speed
            dbn,ubn,gdn,ugn=_photon_moment_longitudinal(zero(al),zero(al),
                zero(al),zero(al),al,dt,G,nu,sl,k,cs2,cph2,rc.response_model)
            tx,tgx=_photon_moment_transverse(zero(ax),zero(ax),ax-alx,dt,G,nu,st,
                                             rc.response_model)
            ty,tgy=_photon_moment_transverse(zero(ay),zero(ay),ay-aly,dt,G,nu,st,
                                             rc.response_model)
            tz,tgz=_photon_moment_transverse(zero(az),zero(az),az-alz,dt,G,nu,st,
                                             rc.response_model)
            dbb,ubb=_photon_baseline_force(al,dt,G,k,cs2)
            phib=dt*_drag_phi1(G*dt)
            hx[p]=kx*ubn/k+tx-(kx*ubb/k+phib*(ax-alx))
            hy[p]=ky*ubn/k+ty-(ky*ubb/k+phib*(ay-aly))
            hz[p]=kz*ubn/k+tz-(kz*ubb/k+phib*(az-alz))
            if RES
                if rc.response_model == _RADIATION_RESPONSE_MOMENTS_EXPONENTIAL
                    drho[p]=rho_mean*dbn
                else
                    dbh,ubh,_,_=_photon_moment_longitudinal(zero(al),zero(al),
                        zero(al),zero(al),al,T(0.5)*dt,G,nu,sl,k,cs2,cph2,
                        rc.response_model)
                    dbbh,ubbh=_photon_baseline_force(al,T(0.5)*dt,G,k,cs2)
                    iduh=Complex{T}(-imag(ubh-ubbh),real(ubh-ubbh))
                    drho[p]=rho_mean*((dbn-dbb)+k*dt*iduh)
                end
            else
                drho[p]=zero(eltype(drho))
            end
            if UPDATE
                pgx[p]+=kx*ugn/k+tgx; pgy[p]+=ky*ugn/k+tgy; pgz[p]+=kz*ugn/k+tgz
                pdelta[p]+=gdn
            end
        end
    end
end

@kernel function _radiation_prepare_velocity_k!(vx,vy,vz,
        @Const(rho),@Const(mx),@Const(my),@Const(mz),smallr)
    c=@index(Global)
    @inbounds begin
        r=max(rho[c],smallr); vx[c]=mx[c]/r; vy[c]=my[c]/r; vz[c]=mz[c]/r
    end
end

@kernel function _radiation_force_to_accel_k!(fx,fy,fz,@Const(rho),smallr)
    c=@index(Global)
    @inbounds begin
        ir=inv(max(rho[c],smallr)); fx[c]*=ir; fy[c]*=ir; fz[c]*=ir
    end
end

@kernel function _radiation_magnetic_accel_pair_k!(fx,fy,fz,
        @Const(rho0),@Const(Bx0),@Const(By0),@Const(Bz0),
        @Const(rho1),@Const(Bx1),@Const(By1),@Const(Bz1),
        n::Int,inv2dx,smallr)
    c=@index(Global)
    @inbounds begin
        i=(c-1)%n+1; j=((c-1)÷n)%n+1; k=(c-1)÷(n*n)+1
        x0,y0,z0=_tct_magnetic_force_at(Bx0,By0,Bz0,i,j,k,n,inv2dx)
        x1,y1,z1=_tct_magnetic_force_at(Bx1,By1,Bz1,i,j,k,n,inv2dx)
        h=typeof(inv2dx)(0.5)
        ir0=inv(max(rho0[c],smallr)); ir1=inv(max(rho1[c],smallr))
        fx[c]=h*(x0*ir0+x1*ir1)
        fy[c]=h*(y0*ir0+y1*ir1)
        fz[c]=h*(z0*ir0+z1*ir1)
    end
end

@kernel function _radiation_add3_k!(x,y,z,@Const(ax),@Const(ay),@Const(az))
    c=@index(Global)
    @inbounds begin x[c]+=ax[c]; y[c]+=ay[c]; z[c]+=az[c] end
end

@kernel function _radiation_speed2_k!(work,@Const(x),@Const(y),@Const(z))
    c=@index(Global)
    @inbounds work[c]=x[c]*x[c]+y[c]*y[c]+z[c]*z[c]
end

@kernel function _radiation_apply_state_correction_k!(rho,mx,my,mz,E,
        @Const(Bx),@Const(By),@Const(Bz),
        @Const(drho),@Const(cx),@Const(cy),@Const(cz),rho_perturbation,
        rho_mean,gamma,smallr,smallE,density_blend,
        absolute_density::Val{ABS}) where {ABS}
    c=@index(Global)
    @inbounds begin
        T=eltype(E); r=max(rho[c],smallr)
        delta0=rho_perturbation[c]
        target=ABS ? rho_mean+drho[c] : rho_mean+delta0+drho[c]
        raw_rn=r+density_blend*(target-r)
        rn=max(raw_rn,smallr)
        deltan=rn-rho_mean
        rho_perturbation[c]=raw_rn==rn ? deltan : rn-rho_mean
        vx=mx[c]/r+density_blend*cx[c]
        vy=my[c]/r+density_blend*cy[c]
        vz=mz[c]/r+density_blend*cz[c]
        ek0=T(0.5)*(mx[c]^2+my[c]^2+mz[c]^2)/r
        eb=T(0.5)*(Bx[c]^2+By[c]^2+Bz[c]^2)
        eint=max(E[c]-ek0-eb,smallE)
        eintn=eint*(rn/r)^gamma
        nmx=rn*vx; nmy=rn*vy; nmz=rn*vz
        ek1=T(0.5)*(nmx*nmx+nmy*nmy+nmz*nmz)/rn
        rho[c]=rn; mx[c]=nmx; my[c]=nmy; mz[c]=nmz
        E[c]=max(eintn+ek1+eb,smallE)
    end
end

function _radiation_transform_velocity_density!(ws::RadiationWorkspace,s::MHDState,
                                                state,dt,gamma_drag,rc;
                                                density_residual=false,
                                                photons=nothing,
                                                update_photons=false,
                                                rho_mean=nothing)
    x,y,z=ws.correction
    _radiation_prepare_velocity_k!(s.be)(x,y,z,state[1],state[2],state[3],state[4],
                                         s.smallr;ndrange=ncells(s))
    KA.synchronize(s.be)
    copyto!(ws.fft_real,ws.density_perturbation)
    _radiation_forward_batch!(ws.hat_batch,ws.real_batch)
    n=s.dims[1]; T=eltype(x); kfac=T(2pi)/(T(n)*s.dx)
    meanrho=rho_mean === nothing ? ws.rho_mean : T(rho_mean)
    if _radiation_evolves_moments(rc.response_model)
        photons isa PhotonMomentState ||
            error("response_model=:moments requires a PhotonMomentState")
        _photon_moment_homogeneous_hat_k!(s.be)(ws.velocity_hat...,ws.density_hat,
            photons.velocity_hat...,photons.delta_hat,Int32(n),kfac,T(dt),
            T(gamma_drag),inv(meanrho),rc,Val(density_residual),Val(update_photons);
            ndrange=length(ws.density_hat))
    else
        _radiation_homogeneous_hat_k!(s.be)(ws.velocity_hat...,ws.density_hat,
            Int32(n),kfac,T(dt),T(gamma_drag),inv(meanrho),rc,
            Val(density_residual);ndrange=length(ws.density_hat))
    end
    KA.synchronize(s.be)
    _radiation_inverse_triplet!(ws.correction,ws)
    return ws.correction
end

function _radiation_transform_force!(out,ws::RadiationWorkspace,s::MHDState,state,
                                     dt,gamma_drag,rc;density_residual=false,
                                     photons=nothing,update_photons=false,
                                     corrected_state=nothing,rho_mean=nothing)
    fx,fy,fz=out; n=s.dims[1]; inv2dx=eltype(fx)(1)/(eltype(fx)(2)*s.dx)
    if corrected_state === nothing
        _tct_magnetic_force_k!(s.be)(fx,fy,fz,state[6],state[7],state[8],n,inv2dx;
                                     ndrange=ncells(s))
        _radiation_force_to_accel_k!(s.be)(fx,fy,fz,state[1],s.smallr;
                                            ndrange=ncells(s))
    else
        _radiation_magnetic_accel_pair_k!(s.be)(fx,fy,fz,
            state[1],state[6],state[7],state[8],
            corrected_state[1],corrected_state[6],corrected_state[7],
            corrected_state[8],n,inv2dx,s.smallr;ndrange=ncells(s))
    end
    KA.synchronize(s.be)
    _radiation_forward_triplet!(ws,out)
    T=eltype(fx); kfac=T(2pi)/(T(n)*s.dx)
    meanrho=rho_mean === nothing ? ws.rho_mean : T(rho_mean)
    if _radiation_evolves_moments(rc.response_model)
        photons isa PhotonMomentState ||
            error("response_model=:moments requires a PhotonMomentState")
        _photon_moment_source_hat_k!(s.be)(ws.velocity_hat...,ws.density_hat,
            photons.velocity_hat...,photons.delta_hat,Int32(n),kfac,T(dt),
            T(gamma_drag),meanrho,rc,Val(density_residual),Val(update_photons);
            ndrange=length(ws.velocity_hat[1]))
    else
        _radiation_source_hat_k!(s.be)(ws.velocity_hat...,ws.density_hat,
            Int32(n),kfac,T(dt),T(gamma_drag),meanrho,rc,
            Val(density_residual);ndrange=length(ws.velocity_hat[1]))
    end
    KA.synchronize(s.be)
    _radiation_inverse_triplet!(out,ws)
    out
end

function _radiation_apply_correction!(s,ws,corr;density_absolute=false,
                                      density_blend=1)
    _radiation_apply_state_correction_k!(s.be)(s.U[1],s.U[2],s.U[3],s.U[4],s.U[5],
        s.U[6],s.U[7],s.U[8],ws.fft_real,corr...,ws.density_perturbation,
        ws.rho_mean,s.γ,s.smallr,eltype(s.U[1])(1e-30),
        eltype(s.U[1])(density_blend),Val(density_absolute);
        ndrange=ncells(s))
    KA.synchronize(s.be)
end

function _radiation_density_blend(ws::RadiationWorkspace,s::MHDState;
                                  absolute::Bool)
    gate=max(Float64(s.smallr),Float64(s.llf_dmin))
    rho_min=Float64(minimum(s.U[1]))
    delta_min=Float64(minimum(ws.fft_real))
    target_min=absolute ? Float64(ws.rho_mean)+delta_min : rho_min+delta_min
    if isfinite(rho_min) && isfinite(target_min) && target_min>=gate
        return 1.0
    end
    denom=rho_min-target_min
    denom>0 && isfinite(denom) || return 0.0
    clamp((rho_min-gate)/denom,0.0,1.0)
end

@kernel function _radiation_internal_energy_k!(out,@Const(rho),@Const(mx),
        @Const(my),@Const(mz),@Const(E),@Const(Bx),@Const(By),@Const(Bz),smallr)
    c=@index(Global)
    @inbounds begin
        r=max(rho[c],smallr)
        ek=eltype(out)(0.5)*(mx[c]^2+my[c]^2+mz[c]^2)/r
        eb=eltype(out)(0.5)*(Bx[c]^2+By[c]^2+Bz[c]^2)
        out[c]=E[c]-ek-eb
    end
end

function _radiation_state_bounds!(ws::RadiationWorkspace,s::MHDState)
    _radiation_internal_energy_k!(s.be)(ws.fft_real,s.U[1],s.U[2],s.U[3],
        s.U[4],s.U[5],s.U[6],s.U[7],s.U[8],s.smallr;ndrange=ncells(s))
    KA.synchronize(s.be)
    (rho_min=Float64(minimum(s.U[1])),rho_max=Float64(maximum(s.U[1])),
     eint_min=Float64(minimum(ws.fft_real)),
     eint_max=Float64(maximum(ws.fft_real)))
end

@kernel function _radiation_fallback_mask_k!(mask,@Const(trial_rho),N)
    c=@index(Global)
    @inbounds begin
        i=(c-1)%N+1; j=((c-1)÷N)%N+1; k=(c-1)÷(N*N)+1
        im=i==1 ? N : i-1; ip=i==N ? 1 : i+1
        jm=j==1 ? N : j-1; jp=j==N ? 1 : j+1
        km=k==1 ? N : k-1; kp=k==N ? 1 : k+1
        bad=trial_rho[c]<0 ||
            trial_rho[((k-1)*N+(j-1))*N+im]<0 ||
            trial_rho[((k-1)*N+(j-1))*N+ip]<0 ||
            trial_rho[((k-1)*N+(jm-1))*N+i]<0 ||
            trial_rho[((k-1)*N+(jp-1))*N+i]<0 ||
            trial_rho[((km-1)*N+(j-1))*N+i]<0 ||
            trial_rho[((kp-1)*N+(j-1))*N+i]<0
        mask[c]=bad ? one(eltype(mask)) : zero(eltype(mask))
    end
end

@kernel function _radiation_expand_fallback_mask_k!(out,@Const(mask),
        @Const(trial_rho),N)
    c=@index(Global)
    @inbounds begin
        i=(c-1)%N+1; j=((c-1)÷N)%N+1; k=(c-1)÷(N*N)+1
        im=i==1 ? N : i-1; ip=i==N ? 1 : i+1
        jm=j==1 ? N : j-1; jp=j==N ? 1 : j+1
        km=k==1 ? N : k-1; kp=k==N ? 1 : k+1
        bad=mask[c]>0 || trial_rho[c]<0 ||
            mask[((k-1)*N+(j-1))*N+im]>0 ||
            mask[((k-1)*N+(j-1))*N+ip]>0 ||
            mask[((k-1)*N+(jm-1))*N+i]>0 ||
            mask[((k-1)*N+(jp-1))*N+i]>0 ||
            mask[((km-1)*N+(j-1))*N+i]>0 ||
            mask[((kp-1)*N+(j-1))*N+i]>0 ||
            trial_rho[((k-1)*N+(j-1))*N+im]<0 ||
            trial_rho[((k-1)*N+(j-1))*N+ip]<0 ||
            trial_rho[((k-1)*N+(jm-1))*N+i]<0 ||
            trial_rho[((k-1)*N+(jp-1))*N+i]<0 ||
            trial_rho[((km-1)*N+(j-1))*N+i]<0 ||
            trial_rho[((kp-1)*N+(j-1))*N+i]<0
        out[c]=bad ? one(eltype(out)) : zero(eltype(out))
    end
end

function _radiation_prepare_fallback_mask!(ws::RadiationWorkspace,s::MHDState)
    _radiation_fallback_mask_k!(s.be)(ws.density_perturbation,s.U[1],s.dims[1];
                                           ndrange=ncells(s))
    KA.synchronize(s.be)
    nothing
end


function _radiation_expand_fallback_mask!(ws::RadiationWorkspace,s::MHDState)
    _radiation_expand_fallback_mask_k!(s.be)(ws.fft_real,
        ws.density_perturbation,s.U[1],s.dims[1];ndrange=ncells(s))
    KA.synchronize(s.be)
    copyto!(ws.density_perturbation,ws.fft_real)
    # The first packed segment normally carries the original GLM psi state.
    copyto!(view(ws.packed_psi_correction,1:ncells(s)),s.scratch[9])
    nothing
end

function _radiation_debug_rho(s, stage, threshold, context)
    threshold > 0 || return nothing
    rho_min = Float64(minimum(s.U[1]))
    rho_max = Float64(maximum(s.U[1]))
    e_min = Float64(minimum(s.U[5]))
    e_max = Float64(maximum(s.U[5]))
    if !(isfinite(rho_min) && isfinite(rho_max) && isfinite(e_min) &&
         isfinite(e_max) && rho_min >= threshold && e_min > 0)
        error("radiation stage=$stage $context crossed state gate: " *
              "rho=[$rho_min,$rho_max] E=[$e_min,$e_max] " *
              "rho_threshold=$threshold")
    end
    return nothing
end

function _radiation_correction_cfl(ws,s,dt,correction)
    _radiation_speed2_k!(s.be)(ws.fft_real,correction...;ndrange=ncells(s))
    KA.synchronize(s.be)
    vmax=sqrt(max(0.0,Float64(maximum(ws.fft_real))))
    return vmax,Float64(dt)*vmax/Float64(s.dx)
end

function _update_free_streaming_photons_velocity!(photons::PhotonMomentState,
                                                  ws::RadiationWorkspace,
                                                  s::MHDState{T},state,dt::T,
                                                  gamma_drag::T,
                                                  rc::RadiationClosure{T},
                                                  rho_mean::T) where {T}
    x,y,z=ws.correction
    _radiation_prepare_velocity_k!(s.be)(x,y,z,state[1],state[2],state[3],state[4],
                                         s.smallr;ndrange=ncells(s))
    KA.synchronize(s.be)
    copyto!(ws.fft_real,reshape(state[1],s.dims))
    _radiation_forward_batch!(ws.hat_batch,ws.real_batch)
    n=s.dims[1]
    kfac=T(2pi)/(T(n)*s.dx)
    _photon_moment_homogeneous_hat_k!(s.be)(
        ws.velocity_hat...,ws.density_hat,
        photons.velocity_hat...,photons.delta_hat,
        Int32(n),kfac,dt,gamma_drag,inv(rho_mean),rc,Val(true),Val(true);
        ndrange=length(ws.density_hat))
    KA.synchronize(s.be)
    photons
end

function _update_free_streaming_photons_force!(photons::PhotonMomentState,
                                               ws::RadiationWorkspace,
                                               s::MHDState{T},state,dt::T,
                                               gamma_drag::T,
                                               rc::RadiationClosure{T},
                                               rho_mean::T;
                                               corrected_state) where {T}
    fx,fy,fz=ws.correction
    n=s.dims[1]
    inv2dx=T(0.5)/s.dx
    _radiation_magnetic_accel_pair_k!(s.be)(fx,fy,fz,
        state[1],state[6],state[7],state[8],
        corrected_state[1],corrected_state[6],corrected_state[7],
        corrected_state[8],n,inv2dx,s.smallr;ndrange=ncells(s))
    KA.synchronize(s.be)
    _radiation_forward_triplet!(ws,ws.correction)
    kfac=T(2pi)/(T(n)*s.dx)
    _photon_moment_source_hat_k!(s.be)(
        ws.velocity_hat...,ws.density_hat,
        photons.velocity_hat...,photons.delta_hat,
        Int32(n),kfac,dt,gamma_drag,rho_mean,rc,Val(true),Val(true);
        ndrange=length(ws.density_hat))
    KA.synchronize(s.be)
    photons
end

function _step_radiation_free_streaming!(s::MHDState{T},ws,dt::T;
        drag_impulse::T,rc::RadiationClosure{T},ch::T,glm_cr::Real,
        integrator::Symbol,energy_ledger,timing,photon_state) where {T}
    photon_before=0.0
    if energy_ledger!==nothing
        _time_radiation_stage!(timing,:ledger_s) do
            photon_before=photon_moment_energy(photon_state,ws,s,rc)
        end
    end
    _time_radiation_stage!(timing,:godunov_s) do
        fill!(ws.correction[1],zero(T)); fill!(ws.correction[2],zero(T))
        fill!(ws.correction[3],zero(T))
        copyto!(view(ws.packed_psi_correction,1:ncells(s)),s.U[9])
        decay=exp(-T(glm_cr)*ch*dt/s.dx)
        if integrator===:ref
            step_ref_radiation!(s,dt;ch=ch,decay=decay,
                                drag_impulse=drag_impulse,
                                correction=ws.packed_psi_correction)
        elseif integrator===:cube
            step_cube_radiation!(s,dt;ch=ch,decay=decay,
                                 drag_impulse=drag_impulse,
                                 correction=ws.packed_psi_correction)
        else
            error("unknown radiation integrator :$integrator (have :ref, :cube)")
        end
    end
    _time_radiation_stage!(timing,:corrector_s) do
        old=s.scratch
        gamma_drag=drag_impulse/dt
        rho_mean=ws.rho_mean
        _update_free_streaming_photons_velocity!(photon_state,ws,s,old,dt,
                                                 gamma_drag,rc,rho_mean)
        _update_free_streaming_photons_force!(photon_state,ws,s,old,dt,
                                              gamma_drag,rc,rho_mean;
                                              corrected_state=s.U)
    end
    if energy_ledger!==nothing
        _time_radiation_stage!(timing,:ledger_s) do
            change=_radiation_energy_changes(ws,s,s.scratch)
            photon_change=photon_moment_energy(photon_state,ws,s,rc)-photon_before
            _record_radiation_energy!(energy_ledger,change;
                                      photon_change=photon_change)
        end
    end
    timing===nothing || (timing.fast_steps+=1)
    reconstruct_radiation_density!(ws,s)
    s
end

"""
    step_radiation_godunov!(s, ws, dt; drag_impulse, closure, ch,
                            glm_cr=GLM_CR, integrator=:cube,
                            photon_state=nothing)

Advance full GLM-MHD while embedding the nonlocal photon-baryon response in
the Godunov midpoint states. The spectral solve is a correction to the exact
local-drag predictor, so the free-streaming limit recovers
`step_drag_godunov!`. The same response corrects the conservative trial state;
gas internal and magnetic energies are retained while radiation removes or
returns only the corresponding kinetic work.
"""
function step_radiation_godunov!(s::MHDState{T},ws,dt::Real;
        drag_impulse::Real,closure::RadiationClosure,ch::Real,
        glm_cr::Real=GLM_CR,integrator::Symbol=:cube,
        energy_ledger::Union{Nothing,RadiationEnergyLedger}=nothing,
        timing::Union{Nothing,RadiationStepTiming}=nothing,
        free_streaming_tolerance::Real=0,
        photon_state=nothing,debug_rho_min::Real=0,
        debug_context::AbstractString="",
        debug_correction_cfl::Real=Inf,
        positivity_fallback::Bool=true,
        positivity_max_fallback_passes::Int=8,
        positivity_wave_growth_limit::Real=4,
        positivity_max_subdivisions::Int=8,
        checkpoint_fallback_on_rejection::Bool=false,
        _positivity_subdivision_depth::Int=0) where {T}
    all_periodic(s) || error("spectral radiation coupling requires periodic boundaries")
    dt>0 || error("radiation-coupled dt must be positive")
    drag_impulse>=0 || error("radiation drag_impulse must be non-negative")
    positivity_max_fallback_passes>0 ||
        error("positivity_max_fallback_passes must be positive")
    positivity_wave_growth_limit>1 ||
        error("positivity_wave_growth_limit must exceed one")
    positivity_max_subdivisions>=0 ||
        error("positivity_max_subdivisions must be non-negative")
    ws.density_limiter_active[]=false
    rc=RadiationClosure{T}(T(closure.inertia_ratio),T(closure.mean_free_path),
        T(closure.photon_speed),T(closure.gas_sound_speed),T(closure.bridge_q2),
        T(closure.longitudinal_viscosity),T(closure.nyquist_guard_fraction),
        closure.response_model)
    kfund=T(2pi)/(T(s.dims[1])*s.dx)
    if rc.response_model == _RADIATION_RESPONSE_BRIDGE &&
            rc.mean_free_path*rc.mean_free_path*kfund*kfund >= rc.bridge_q2
        energy_ledger === nothing || error(
            "radiation energy accounting requires an allocated spectral workspace")
        return step_drag_godunov!(s,dt;drag_impulse=drag_impulse,ch=ch,
                                  glm_cr=glm_cr,integrator=integrator)
    end
    ws isa RadiationWorkspace ||
        error("radiation diffusion modes require an allocated RadiationWorkspace")
    if _radiation_evolves_moments(rc.response_model)
        photon_state isa PhotonMomentState ||
            error("response_model=:moments requires photon_state=allocate_photon_moment_state(s)")
        photon_state.dims==s.dims || error("photon and MHD grid dimensions differ")
    end
    gamma_drag=T(drag_impulse)/T(dt)
    free_streaming_tolerance>=0 ||
        error("free_streaming_tolerance must be non-negative")
    kfund=T(2pi)/(T(s.dims[1])*s.dx)
    if _radiation_evolves_moments(rc.response_model) &&
            free_streaming_tolerance>0 &&
            radiation_free_streaming_bound(kfund,gamma_drag,rc) <=
                T(free_streaming_tolerance)
        return _step_radiation_free_streaming!(s,ws,T(dt);
            drag_impulse=T(drag_impulse),rc=rc,ch=T(ch),glm_cr=glm_cr,
            integrator=integrator,energy_ledger=energy_ledger,timing=timing,
            photon_state=photon_state)
    end
    timing===nothing || (timing.spectral_steps+=1)
    rho_mean=ws.rho_mean
    photon_energy_before=_time_radiation_stage!(timing,:ledger_s) do
        energy_ledger!==nothing && _radiation_evolves_moments(rc.response_model) ?
            photon_moment_energy(photon_state,ws,s,rc) : 0.0
    end
    _time_radiation_stage!(timing,:predictor_s) do
        half=T(0.5)*T(dt)
        _radiation_transform_velocity_density!(ws,s,s.U,half,gamma_drag,rc;
                                               density_residual=false,
                                               photons=photon_state,
                                               rho_mean=rho_mean)
        tmp=(reshape(s.scratch[1],s.dims),reshape(s.scratch[2],s.dims),
             reshape(s.scratch[3],s.dims))
        for d in 1:3
            copyto!(tmp[d],ws.correction[d])
        end
        _radiation_transform_force!(ws.correction,ws,s,s.U,half,gamma_drag,rc;
                                    density_residual=false,photons=photon_state,
                                    rho_mean=rho_mean)
        _radiation_add3_k!(s.be)(ws.correction...,tmp...;ndrange=ncells(s))
        KA.synchronize(s.be)
        if isfinite(debug_correction_cfl)
            correction_vmax,correction_cfl=
                _radiation_correction_cfl(ws,s,dt,ws.correction)
            correction_cfl<=debug_correction_cfl || error(
                "radiation predictor $debug_context exceeds correction CFL: " *
                "dt_vcorr_dx=$correction_cfl limit=$debug_correction_cfl " *
                "vmax=$correction_vmax dt=$dt dx=$(s.dx)")
        end
        copyto!(view(ws.packed_psi_correction,1:ncells(s)),s.U[9])
    end
    subdivide_step=false
    trial_wave_speed=0.0
    _time_radiation_stage!(timing,:godunov_s) do
        decay=exp(-T(glm_cr)*T(ch)*T(dt)/s.dx)
        track_density=rc.response_model!=_RADIATION_RESPONSE_MOMENTS_EXPONENTIAL
        use_fallback=positivity_fallback && integrator===:cube
        if integrator===:ref
            step_ref_radiation!(s,dt;ch=ch,decay=decay,drag_impulse=drag_impulse,
                                correction=ws.packed_psi_correction,
                                track_density=track_density)
        elseif integrator===:cube
            step_cube_radiation!(s,dt;ch=ch,decay=decay,drag_impulse=drag_impulse,
                                 correction=ws.packed_psi_correction,
                                 track_density=track_density,
                                 check_admissibility=use_fallback)
        else
            error("unknown radiation integrator :$integrator (have :ref, :cube)")
        end
        if use_fallback
            trial_admissible=_radiation_trial_admissible(s)
            wave_limit=Float64(positivity_wave_growth_limit)*
                       max(Float64(ch),eps(Float64))
            if trial_admissible
                trial_wave_speed=Float64(max_wavespeed!(ws.fft_real,s))
            end
            reject_trial=!trial_admissible ||
                !(isfinite(trial_wave_speed) && trial_wave_speed<=wave_limit)
            if reject_trial
                high_trial_wave_speed=trial_wave_speed
                high_trial_rho_min=trial_admissible ? NaN :
                    Float64(minimum(s.U[1]))
                _restore_radiation_trial!(s,ws,track_density)

                # The second trial is a complete conservative update: every face
                # uses the same minmod-PLM/HLL flux, unlike a cell-local flux swap.
                step_cube_radiation!(s,dt;ch=ch,decay=decay,
                    drag_impulse=drag_impulse,
                    correction=ws.packed_psi_correction,
                    track_density=track_density,check_admissibility=true,
                    robust_fallback=true)
                robust_admissible=_radiation_trial_admissible(s)
                robust_wave_speed=robust_admissible ?
                    Float64(max_wavespeed!(ws.fft_real,s)) : 0.0
                robust_rejected=!robust_admissible ||
                    !(isfinite(robust_wave_speed) && robust_wave_speed<=wave_limit)
                robust_trial_rho_min=robust_admissible ? NaN :
                    Float64(minimum(s.U[1]))
                if !robust_rejected
                    trial_wave_speed=robust_wave_speed
                    fallback_count=0
                    if timing!==nothing
                        timing.fallback_steps+=1
                        timing.plm_hll_fallback_steps+=1
                        fallback_count=timing.plm_hll_fallback_steps
                    end
                    ws.density_limiter_active[]=true
                    if timing===nothing || fallback_count==1 || fallback_count%1000==0
                        @warn "radiation step used conservative minmod-PLM/HLL fallback" dt debug_context fallback_count high_trial_rho_min high_trial_wave_speed robust_wave_speed wave_limit
                    end
                elseif checkpoint_fallback_on_rejection
                    _checkpoint_fallback_rejection!(s,ws,track_density;dt=dt,
                        debug_context=debug_context,
                        trial_rho_min=robust_trial_rho_min,
                        trial_wave_speed=robust_wave_speed,wave_limit=wave_limit)
                elseif _positivity_subdivision_depth<positivity_max_subdivisions
                    _restore_radiation_trial!(s,ws,track_density)
                    subdivide_step=true
                    timing===nothing || (timing.subdivision_steps+=1)
                    if _positivity_subdivision_depth==0
                        @warn "radiation PPM/HLL and minmod-PLM/HLL trials rejected; subdividing coupled step" dt debug_context high_trial_rho_min robust_trial_rho_min high_trial_wave_speed robust_wave_speed wave_limit
                    end
                else
                    _restore_radiation_trial!(s,ws,track_density)
                    error("conservative minmod-PLM/HLL radiation fallback " *
                          "remained inadmissible after " *
                          "$positivity_max_subdivisions subdivisions: dt=$dt " *
                          "$debug_context high_trial_rho_min=$high_trial_rho_min " *
                          "robust_trial_rho_min=$robust_trial_rho_min " *
                          "high_wave_speed=$high_trial_wave_speed " *
                          "robust_wave_speed=$robust_wave_speed " *
                          "wave_limit=$wave_limit ch=$ch")
                end
            end
        end
    end
    if subdivide_step
        half_dt=dt/2
        half_impulse=drag_impulse/2
        depth=_positivity_subdivision_depth+1
        step_radiation_godunov!(s,ws,half_dt;drag_impulse=half_impulse,
            closure=closure,ch=ch,glm_cr=glm_cr,integrator=integrator,
            energy_ledger=energy_ledger,timing=timing,
            free_streaming_tolerance=free_streaming_tolerance,
            photon_state=photon_state,debug_rho_min=debug_rho_min,
            debug_context=debug_context,
            debug_correction_cfl=debug_correction_cfl,
            positivity_fallback=positivity_fallback,
            positivity_max_fallback_passes=positivity_max_fallback_passes,
            positivity_wave_growth_limit=positivity_wave_growth_limit,
            positivity_max_subdivisions=positivity_max_subdivisions,
            checkpoint_fallback_on_rejection=checkpoint_fallback_on_rejection,
            _positivity_subdivision_depth=depth)
        sub_ch=max(Float64(ch),Float64(max_wavespeed!(ws.fft_real,s)))
        return step_radiation_godunov!(s,ws,half_dt;
            drag_impulse=half_impulse,closure=closure,ch=sub_ch,glm_cr=glm_cr,
            integrator=integrator,energy_ledger=energy_ledger,timing=timing,
            free_streaming_tolerance=free_streaming_tolerance,
            photon_state=photon_state,debug_rho_min=debug_rho_min,
            debug_context=debug_context,
            debug_correction_cfl=debug_correction_cfl,
            positivity_fallback=positivity_fallback,
            positivity_max_fallback_passes=positivity_max_fallback_passes,
            positivity_wave_growth_limit=positivity_wave_growth_limit,
            positivity_max_subdivisions=positivity_max_subdivisions,
            checkpoint_fallback_on_rejection=checkpoint_fallback_on_rejection,
            _positivity_subdivision_depth=depth)
    end
    _time_radiation_stage!(timing,:corrector_s) do
        _radiation_debug_rho(s,"mhd_trial",debug_rho_min,debug_context)
        old=s.scratch
        _radiation_transform_velocity_density!(ws,s,old,T(dt),gamma_drag,rc;
                                               density_residual=true,
                                               photons=photon_state,
                                               update_photons=_radiation_evolves_moments(
                                                   rc.response_model),
                                               rho_mean=rho_mean)
        density_absolute=rc.response_model==
            _RADIATION_RESPONSE_MOMENTS_EXPONENTIAL
        density_blend=ws.density_limiter_active[] ?
            _radiation_density_blend(ws,s;absolute=density_absolute) : 1.0
        _radiation_apply_correction!(s,ws,ws.correction;
            density_absolute=density_absolute,density_blend=density_blend)
        if density_blend<1
            ws.density_limiter_min_theta[]=min(ws.density_limiter_min_theta[],
                                               density_blend)
            timing===nothing || (timing.density_limited_stages+=1)
        end
        _radiation_debug_rho(s,"density_correction",debug_rho_min,debug_context)
        _radiation_transform_force!(ws.correction,ws,s,old,T(dt),gamma_drag,rc;
                                    density_residual=true,photons=photon_state,
                                    update_photons=_radiation_evolves_moments(
                                        rc.response_model),
                                    corrected_state=s.U,rho_mean=rho_mean)
        density_blend=ws.density_limiter_active[] ?
            _radiation_density_blend(ws,s;absolute=false) : 1.0
        _radiation_apply_correction!(s,ws,ws.correction;
                                     density_blend=density_blend)
        if density_blend<1
            ws.density_limiter_min_theta[]=min(ws.density_limiter_min_theta[],
                                               density_blend)
            timing===nothing || (timing.density_limited_stages+=1)
        end
        _radiation_debug_rho(s,"force_correction",debug_rho_min,debug_context)
    end
    if energy_ledger !== nothing
        _time_radiation_stage!(timing,:ledger_s) do
            change=_radiation_energy_changes(ws,s,s.scratch)
            photon_change=_radiation_evolves_moments(rc.response_model) ?
                photon_moment_energy(photon_state,ws,s,rc)-photon_energy_before : 0.0
            _record_radiation_energy!(energy_ledger,change;photon_change=photon_change)
        end
    end
    reconstruct_radiation_density!(ws,s)
    return s
end
