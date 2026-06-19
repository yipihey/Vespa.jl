# patch8_decomp.jl — DEMO: split a uniform periodic grid into np³ sibling patches,
# run each patch's HYDRO + CHEMISTRY on the GPU, and solve the top-grid GRAVITY with
# our parallel CPU FFT (gather → FFTW → scatter).  Validates that the decomposed run
# reproduces the undecomposed (np=1) reference, and reports per-patch GPU timing.
#
# Run (GPU): BACKEND=cuda CIC_NGRID=128 julia --project=lib/MultiCode/test \
#              lib/MultiCode/examples/patch8_decomp.jl
#      (CPU reference: BACKEND=cpu)

using MultiCode, Printf
import PoissonKernels

const BE    = Symbol(get(ENV, "BACKEND", "cuda"))
# Load CUDA so the PPMKernels/PoissonKernels/ChemistryKernels CUDA extensions register
# the :cuda backend (per-patch hydro+chem run on the device).
BE === :cuda && @eval import CUDA
const T     = BE === :cpu ? Float64 : Float32
const NG    = 4
const NGR   = parse(Int, get(ENV, "CIC_NGRID", "64"))
const NC    = (NGR, NGR, NGR)
const NPX   = parse(Int, get(ENV, "CIC_NP", "2"))
const NP    = (NPX, NPX, NPX)
const GAM   = 5/3
const BOX   = 1.0
const DXC   = BOX / NGR
const DT    = 0.02 * DXC
const NSP   = 3
const KCYC  = parse(Int, get(ENV, "CIC_KCYC", "4"))

function make_ic(nc, T)
    rho=Array{T,3}(undef,nc); v1=similar(rho);v2=similar(rho);v3=similar(rho);eint=similar(rho)
    sp=[similar(rho) for _ in 1:NSP]
    @inbounds for k in 1:nc[3],j in 1:nc[2],i in 1:nc[1]
        x=2π*(i-1)/nc[1];y=2π*(j-1)/nc[2];z=2π*(k-1)/nc[3]
        rho[i,j,k]=T(1.0+0.30*sin(x)*cos(y)+0.20*sin(z))
        v1[i,j,k]=T(0.05*sin(y));v2[i,j,k]=T(0.04*cos(z));v3[i,j,k]=T(0.03*sin(x))
        eint[i,j,k]=T(1.0+0.10*cos(x)*sin(y))
        sp[1][i,j,k]=rho[i,j,k]*T(0.05*(1.1+sin(x)));sp[2][i,j,k]=rho[i,j,k]*T(1e-3);sp[3][i,j,k]=rho[i,j,k]*T(1e-4)
    end
    (D=rho,S1=rho.*v1,S2=rho.*v2,S3=rho.*v3,Ge=rho.*eint,
     Tau=rho.*(eint.+T(0.5).*(v1.^2 .+v2.^2 .+v3.^2)),species=sp)
end

function run(np; label)
    pg = build_patchgrid(; ng=NG, ncell=NC, np=np, dx=DXC, gamma=GAM, nspecies=NSP, besym=BE, T=T)
    scatter_global!(pg, make_ic(NC, T))
    ρg = zeros(Float64, NC); φg = zeros(Float64, NC)
    m0 = total_mass(pg)
    twall = 0.0
    for cyc in 1:KCYC
        accel = global_gravity_accel(pg; G=1.0, a=1.0, boxsize=BOX, ρg=ρg, φg=φg)
        ord = isodd(cyc) ? (1,2,3) : (3,2,1)
        t0 = time(); patch_step!(pg, DT; a_value=1.0, order=ord, accel=accel, chem=true); twall += time()-t0
    end
    @printf("  %-8s np=%s  %d patches of %d³ on %s/%s  |  %.3f s/cyc (hydro+chem), mass drift %.2e\n",
            label, np, prod(np), NC[1]÷np[1], BE, T, twall/KCYC, abs(total_mass(pg)-m0)/m0)
    return gather_global(pg)
end

relerr(a,b) = maximum(abs.(Float64.(a) .- Float64.(b))) / (maximum(abs.(Float64.(b))) + eps())

@printf("patch8_decomp: %d³ periodic, NP=%s (%d patches), %d cycles, backend=%s T=%s\n",
        NGR, NP, prod(NP), KCYC, BE, T)
ref = run((1,1,1); label="REF")
dec = run(NP;      label="DECOMP")
@printf("DECOMP vs REF (decomposition invariance):\n")
for f in (:D,:S1,:S2,:S3,:Tau,:Ge)
    @printf("  %-4s rel err = %.3e\n", f, relerr(getfield(dec,f), getfield(ref,f)))
end
for s in 1:NSP
    @printf("  sp%d  rel err = %.3e\n", s, relerr(dec.species[s], ref.species[s]))
end
