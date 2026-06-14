# Visual comparison of the one-ghost local PPM candidate against the wide
# DirectEuler PPM and a cheap PLM+HLL baseline.
#
# Top row: translating acoustic wave after 10 box crossings (gray = exact).
# Bottom row: matched log-density mid-plane slices of Mach-5 decaying turbulence.
#
# Run from lib/PPMKernels:
#   julia --project=test bench/plot_localppm_comparison.jl [turb_n] [t/t_cross]

using PPMKernels, KernelAbstractions, Printf, Random, LinearAlgebra, Statistics
try; @eval using Metal; catch err; @info "Metal not loadable - CPU fallback" err; end

const _P = PPMKernels
const NG = 4
const GAMMA = 1.4
const CS0 = 1.0
const WAVE_N = 128
const WAVE_NY = 8
const WAVE_K = 4
const WAVE_AMP = 1e-3
const WAVE_PERIODS = 10.0

tn = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
tfrac = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5
bkname = _P.has_backend(:metal) ? :metal : :cpu
be = _P.backend(bkname)
const T = Float32
dev(a) = _P.to_device(be, a, T)

const SOLVERS = [
    ("Local PPM + trace + 2-shock", :local),
    ("DirectEuler PPM", :direct),
    ("Hancock PLM + HLL", :plm),
]

function wave_ic()
    nx = WAVE_N; ny = WAVE_NY
    nbx = nx + 2NG; nby = ny + 2NG
    dims = (nbx, nby, nby); N = prod(dims); dx = 1.0 / nx
    d = Vector{Float64}(undef, N); u = similar(d); pr = similar(d)
    idx(i, j, k) = i + nbx*(j-1) + nbx*nby*(k-1)
    p0 = CS0^2 / GAMMA
    for k in 1:nby, j in 1:nby, i in 1:nbx
        x = (i - NG - 0.5) / nx
        s = WAVE_AMP * sinpi(2WAVE_K*x); q = idx(i, j, k)
        d[q] = 1 + s; u[q] = 1 + s; pr[q] = p0 + s
    end
    eint = pr ./ ((GAMMA-1) .* d); etot = eint .+ 0.5 .* u.^2
    (; d, u, vy=zeros(N), vz=zeros(N), eint, etot, dims, N, dx, nbx, nby)
end

function wave_line(a, ic)
    j = NG + WAVE_NY ÷ 2
    [Float64(a[i + ic.nbx*(j-1) + ic.nbx*ic.nby*(j-1)])
     for i in (NG+1):(ic.nbx-NG)]
end

function fmode(v, m)
    n = length(v); z = 0.0im
    @inbounds for j in 0:n-1
        z += v[j+1] * cis(-2pi*m*j/n)
    end
    2z/n
end

function run_wave(kind, ic, dt, nsteps)
    dims=ic.dims; N=ic.N; dx=ic.dx
    pbc5(a,b,c,d,e) = _P.fill_periodic!(dims, NG, a,b,c,d,e)
    pbc6(a,b,c,d,e,f) = _P.fill_periodic!(dims, NG, a,b,c,d,e,f)
    if kind === :direct
        d=dev(ic.d); e=dev(ic.etot); ge=dev(ic.eint)
        vx=dev(ic.u); vy=dev(ic.vy); vz=dev(ic.vz); z=dev(zeros(N))
        _P.with_pool() do
            for s in 1:nsteps
                _P.ppm_step_3d!(d,e,ge,vx,vy,vz,z,z,z,dims,NG;
                    dt, gamma=GAMMA, dx, order=isodd(s) ? (1,2,3) : (3,2,1),
                    bc! = pbc6, idual=0, iflatten=3, isteep=0, idiff=0,
                    gravity=0, eta2=0.1)
            end
        end
        return wave_line(_P.to_host(vx), ic)
    end
    D=dev(ic.d); S1=dev(ic.d.*ic.u); S2=dev(zeros(N)); S3=dev(zeros(N))
    Tau=dev(ic.d.*ic.etot)
    _P.with_pool() do
        for s in 1:nsteps
            order = isodd(s) ? (1,2,3) : (3,2,1)
            if kind === :local
                _P.muscl_hancock_step_3d!(D,S1,S2,S3,Tau,dims,NG;
                    dt, gamma=GAMMA, dx, order, bc! = pbc5, face_periodic=true,
                    recon=:ppm_local, predictor=:trace, riemann=:twoshock)
            else
                _P.muscl_hancock_step_3d!(D,S1,S2,S3,Tau,dims,NG;
                    dt, gamma=GAMMA, dx, order, bc! = pbc5,
                    recon=:plm, predictor=:hancock, riemann=:hll)
            end
        end
    end
    wave_line(_P.to_host(S1) ./ _P.to_host(D), ic)
end

function turb_ic(n; mach=5.0, seed=271, kmin=2, kmax=3, specidx=4.0)
    Random.seed!(seed); nb=n+2NG; dx=1/n
    X = [(i-NG-0.5)*dx for i in 1:nb]
    vx=zeros(nb,nb,nb); vy=zeros(nb,nb,nb); vz=zeros(nb,nb,nb)
    kmin2 = kmin * kmin
    kmax2 = kmax * kmax
    modes=[(kx,ky,kz) for kx in -kmax:kmax, ky in -kmax:kmax, kz in -kmax:kmax
           if kmin2 <= kx*kx + ky*ky + kz*kz <= kmax2]
    for (kx,ky,kz) in modes
        kk=sqrt(float(kx*kx + ky*ky + kz*kz)); amp=kk^(-specidx/2)
        kh=(kx/kk,ky/kk,kz/kk); a=randn(3); ad=sum(a[i]*kh[i] for i in 1:3)
        b=(a[1]-ad*kh[1],a[2]-ad*kh[2],a[3]-ad*kh[3])
        bn=sqrt(sum(abs2,b)); bn < 1e-6 && continue
        b=ntuple(i->b[i]*amp/bn,3); ph=2pi*rand()
        @inbounds for k in 1:nb, j in 1:nb, i in 1:nb
            c=cos(2pi*(kx*X[i]+ky*X[j]+kz*X[k])+ph)
            vx[i,j,k]+=b[1]*c; vy[i,j,k]+=b[2]*c; vz[i,j,k]+=b[3]*c
        end
    end
    ix=(NG+1):(nb-NG)
    vr=sqrt(sum(vx[i,j,k]^2+vy[i,j,k]^2+vz[i,j,k]^2
                for k in ix,j in ix,i in ix)/n^3)
    f=mach/vr; vx.*=f; vy.*=f; vz.*=f
    N=nb^3; d=ones(N); vxf=vec(vx); vyf=vec(vy); vzf=vec(vz)
    eint=fill((1/GAMMA)/(GAMMA-1),N)
    etot=eint .+ 0.5.*(vxf.^2 .+ vyf.^2 .+ vzf.^2)
    (;d,vx=vxf,vy=vyf,vz=vzf,eint,etot,dims=(nb,nb,nb),N,dx)
end

function vmax_ic(ic)
    maximum(sqrt(ic.vx[q]^2+ic.vy[q]^2+ic.vz[q]^2) +
            sqrt(GAMMA*(GAMMA-1)*ic.eint[q]) for q in 1:ic.N)
end

function run_turb(kind, ic, dt, nsteps)
    dims=ic.dims; N=ic.N; dx=ic.dx
    pbc(f...) = _P.fill_periodic!(dims, NG, f...)
    if kind === :direct
        d=dev(ic.d); e=dev(ic.etot); ge=dev(ic.eint)
        vx=dev(ic.vx); vy=dev(ic.vy); vz=dev(ic.vz); z=dev(zeros(N))
        _P.with_pool() do
            for s in 1:nsteps
                _P.ppm_step_3d!(d,e,ge,vx,vy,vz,z,z,z,dims,NG;
                    dt, gamma=GAMMA, dx, order=isodd(s) ? (1,2,3) : (3,2,1),
                    bc! = pbc, idual=1, iflatten=3, isteep=0, idiff=0,
                    gravity=0, eta2=0.1)
            end
        end
        return _P.to_host(d)
    end
    D=dev(ic.d); S1=dev(ic.d.*ic.vx); S2=dev(ic.d.*ic.vy); S3=dev(ic.d.*ic.vz)
    Tau=dev(ic.d.*ic.etot); Ge=dev(ic.d.*ic.eint)
    _P.with_pool() do
        for s in 1:nsteps
            order=isodd(s) ? (1,2,3) : (3,2,1)
            if kind === :local
                _P.muscl_hancock_step_3d!(D,S1,S2,S3,Tau,dims,NG;
                    dt, gamma=GAMMA, dx, order, bc! = pbc, ge=Ge,
                    face_periodic=true, recon=:ppm_local,
                    predictor=:trace, riemann=:twoshock)
            else
                _P.muscl_hancock_step_3d!(D,S1,S2,S3,Tau,dims,NG;
                    dt, gamma=GAMMA, dx, order, bc! = pbc, ge=Ge,
                    recon=:plm, predictor=:hancock, riemann=:hll)
            end
        end
    end
    _P.to_host(D)
end

function midslice(a, dims)
    nx,ny,nz=dims; k=NG+tn÷2
    [Float64(a[i+nx*(j-1)+nx*ny*(k-1)])
     for i in (NG+1):(nx-NG), j in (NG+1):(ny-NG)]
end

function write_ppm(path, img)
    H=size(img,2); W=size(img,3)
    open(path,"w") do io
        write(io,"P6\n$W $H\n255\n")
        for y in 1:H, x in 1:W
            write(io,img[1,y,x],img[2,y,x],img[3,y,x])
        end
    end
end

setpix!(img,x,y,c) = (1<=x<=size(img,3) && 1<=y<=size(img,2)) &&
                      (img[:,y,x] .= c)
function seg!(img,x0,y0,x1,y1,c)
    n=max(1,ceil(Int,max(abs(x1-x0),abs(y1-y0))))
    for s in 0:n
        t=s/n; x=round(Int,x0+t*(x1-x0)); y=round(Int,y0+t*(y1-y0))
        setpix!(img,x,y,c); setpix!(img,x,y+1,c)
    end
end

function wave_panel(path, exact, final)
    W=520; H=300; ml=45; mr=15; mt=18; mb=34
    img=fill(UInt8(250),3,H,W)
    X(i)=ml+(i-1)/(length(exact)-1)*(W-ml-mr)
    ymax=1.1WAVE_AMP; Y(v)=mt+(ymax-v)/(2ymax)*(H-mt-mb)
    axis=UInt8[205,205,205]; gray=UInt8[135,135,135]; blue=UInt8[25,90,205]
    seg!(img,ml,Y(0),W-mr,Y(0),axis)
    for v in (-WAVE_AMP,WAVE_AMP); seg!(img,ml,Y(v),W-mr,Y(v),UInt8[230,230,230]); end
    for i in 1:length(exact)-1
        seg!(img,X(i),Y(exact[i]-1),X(i+1),Y(exact[i+1]-1),gray)
        seg!(img,X(i),Y(final[i]-1),X(i+1),Y(final[i+1]-1),blue)
    end
    write_ppm(path,img)
end

jet(t) = (clamp(1.5-abs(4t-3),0,1),clamp(1.5-abs(4t-2),0,1),
          clamp(1.5-abs(4t-1),0,1))
function density_panel(path, sl, lo, hi)
    n=size(sl,1); scale=max(1,div(500,n)); W=n*scale; H=W
    img=Array{UInt8}(undef,3,H,W)
    for j in 1:n, i in 1:n
        t=clamp((log10(sl[i,j])-lo)/(hi-lo),0,1); c=jet(t)
        for yy in (j-1)*scale+1:j*scale, xx in (i-1)*scale+1:i*scale
            img[:,yy,xx] .= UInt8.(round.(255 .* collect(c)))
        end
    end
    write_ppm(path,img)
end

outdir=mkpath(joinpath(@__DIR__,"turb_out","localppm_visual"))

wic=wave_ic(); exact=wave_line(wic.u,wic)
w_dt=0.3*wic.dx/(2+WAVE_AMP)
w_steps=ceil(Int,WAVE_PERIODS/(2w_dt))
waves=Dict{Symbol,Vector{Float64}}()
wave_metrics=Dict{Symbol,Tuple{Float64,Float64}}()
for (label,kind) in SOLVERS
    waves[kind]=run_wave(kind,wic,w_dt,w_steps)
    p2p=(maximum(waves[kind])-minimum(waves[kind]))/(maximum(exact)-minimum(exact))
    cf=fmode(waves[kind].-1,WAVE_K)
    harm=sqrt(sum(abs2,fmode(waves[kind].-1,m) for m in (2WAVE_K,3WAVE_K,4WAVE_K)))/abs(cf)
    wave_metrics[kind]=(p2p,harm)
    @printf("%-28s wave amp=%.3f distortion=%.3f\n",label,p2p,harm)
end

tic=turb_ic(tn); t_dt=0.2*tic.dx/vmax_ic(tic)
t_steps=ceil(Int,(tfrac/5.0)/t_dt)
slices=Dict{Symbol,Matrix{Float64}}()
for (label,kind) in SOLVERS
    @printf("running %-28s turbulence (%d steps)\n",label,t_steps); flush(stdout)
    slices[kind]=midslice(run_turb(kind,tic,t_dt,t_steps),tic.dims)
end
logs=vcat([vec(log10.(s)) for s in values(slices)]...)
lo,hi=quantile(logs,0.01),quantile(logs,0.99)
@printf("shared density scale: log10(rho) = %.3f .. %.3f\n",lo,hi)

wave_png=String[]; rho_png=String[]
for (label,kind) in SOLVERS
    wp=joinpath(outdir,"wave_$(kind).ppm"); rp=joinpath(outdir,"rho_$(kind).ppm")
    wave_panel(wp,exact,waves[kind]); density_panel(rp,slices[kind],lo,hi)
    wpng=replace(wp,".ppm"=>".png"); rpng=replace(rp,".ppm"=>".png")
    run(`magick $wp $wpng`); run(`magick $rp $rpng`)
    push!(wave_png,wpng); push!(rho_png,rpng)
end

labels=[s[1] for s in SOLVERS]
wave_labeled=String[]; rho_labeled=String[]
for i in eachindex(labels)
    kind=SOLVERS[i][2]; amp,harm=wave_metrics[kind]
    w=joinpath(outdir,"wave_labeled_$i.png")
    r=joinpath(outdir,"rho_labeled_$i.png")
    run(`magick $(wave_png[i]) -background white -gravity north -splice 0x72
         -fill '#202020' -font Helvetica -pointsize 25 -annotate +0+8 $(labels[i])
         -pointsize 18 -annotate +0+42 "wave: amp=$(@sprintf("%.3f",amp)), distortion=$(@sprintf("%.3f",harm))" $w`)
    run(`magick $(rho_png[i]) -resize 520x520! -background white -gravity north -splice 0x72
         -fill '#202020' -font Helvetica -pointsize 25 -annotate +0+8 $(labels[i])
         -pointsize 18 -annotate +0+42 "Mach 5 density, t = $(tfrac) crossing" $r`)
    push!(wave_labeled,w); push!(rho_labeled,r)
end

top=joinpath(outdir,"top.png"); bottom=joinpath(outdir,"bottom.png")
run(Cmd(vcat(["magick"], wave_labeled, ["+append", "+repage", top])))
run(Cmd(vcat(["magick"], rho_labeled, ["+append", "+repage", bottom])))
out=joinpath(@__DIR__,"turb_out","localppm_wave_turbulence_comparison.png")
run(`magick $top $bottom -append +repage -background white -gravity north -splice 0x105
     -fill '#171717' -font Helvetica-Bold -pointsize 34
     -annotate +0+15 "One-ghost Local PPM versus wide PPM and PLM-HLL"
     -font Helvetica -pointsize 20
     -annotate +0+59 "Same initial conditions and timesteps; turbulence panels share log10 density scale [$(@sprintf("%.2f",lo)), $(@sprintf("%.2f",hi))]"
     $out`)
println("wrote $out")
