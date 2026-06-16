# CICASS LINEAR (analytic, cosmic-variance-free) baryon & DM power spectra at each
# simulation output redshift, for comparison to the sim P(k) AT THE SAME z.
#
# Linear theory needs NO field realization: `transfer.x` (the vbc_transfer 2-fluid
# ODE solver) computes the power internally and, with PRINT_PK, writes it straight
# to stdout as  `k[1/Mpc]   k³P_DM/2π²   k³P_gas/2π²   k³P_Temp/2π²`.  We capture
# that — NOT generate() (which additionally runs the heavy makeCosICs 3D realizer
# just to throw away everything but the .pk).  ~2 s per (z, costh) vs ~20 s per z.
#
# The sim P(k) is SPHERICALLY averaged, while transfer.x's PRINT_PK is at a single
# angle costh = k̂·v̂_bc (the v_bc imprint is anisotropic — strongest along the
# stream).  So we run a few costh nodes and average Δ² uniformly in costh on [0,1]
# (= the solid-angle average for an axisymmetric, costh→−costh-symmetric field).
# At high z the anisotropy is ~0 (CICASS's anchor); it grows toward lower z.
#
# Units: transfer.x k is in 1/Mpc; the sim uses h/Mpc, so k_h = k/h and
# P[(Mpc/h)³] = 2π²·Δ²/k_h³.
#
# Run:  CIC_ZOUT="969,...,20" <julia> --project=lib/MultiCode/test \
#         lib/MultiCode/examples/cicass_linear_at_outputs.jl

using CICASSLib, Printf

const BOX     = parse(Float64, get(ENV, "CIC_BOX",   "0.128"))
const OMEGA_M = parse(Float64, get(ENV, "CIC_OMEGAM","0.27"))
const VBC     = parse(Float64, get(ENV, "CIC_VBC",   "30.0"))
const NGRID   = parse(Int,     get(ENV, "CIC_NGRID", "128"))
const HCONST  = parse(Float64, get(ENV, "CIC_HCONST","0.71"))
const NMU     = parse(Int,     get(ENV, "CIC_NMU",   "4"))      # costh nodes for the angle-average
const REPORTS = joinpath(@__DIR__, "..", "..", "..", "reports", "multicode")
const ZOUT = [parse(Float64, s) for s in split(get(ENV, "CIC_ZOUT",
    "1000.0,520.17,274.41,142.39,75.02"), ",")]

# midpoint costh nodes on [0,1]: μ_i = (i-0.5)/NMU, uniform weight
mu_nodes(M) = [(i - 0.5) / M for i in 1:M]

# Run transfer.x at one (z, costh); return (k[1/Mpc], Δ²_DM, Δ²_gas) from PRINT_PK.
function transfer_pk(z::Real, costh::Real)
    tx = CICASSLib.transfer_path()
    isfile(tx) || error("transfer.x not found at $tx")
    vbcdir = normpath(joinpath(CICASSLib.cicass_root(), "vbc_transfer"))
    args = ["-B$BOX", "-N$NGRID", "-V$VBC", "-Z$(round(Int, z))",
            "-D1", "-SinitSB_transfer_out", "-C$costh"]
    out = cd(vbcdir) do
        read(pipeline(ignorestatus(`$tx $args`); stderr = devnull), String)
    end
    k = Float64[]; Δdm = Float64[]; Δgas = Float64[]
    inpk = false
    for line in eachline(IOBuffer(out))
        if occursin("k [Mpc", line); inpk = true; continue; end
        inpk || continue
        t = split(line)
        length(t) >= 3 || continue
        kv = tryparse(Float64, t[1]); kv === nothing && continue
        d  = tryparse(Float64, t[2]); g = tryparse(Float64, t[3])
        (d === nothing || g === nothing) && continue
        push!(k, kv); push!(Δdm, d); push!(Δgas, g)
    end
    isempty(k) && error("transfer.x produced no PRINT_PK rows at z=$z costh=$costh")
    return (k = k, Δdm = Δdm, Δgas = Δgas)
end

# Per-costh AND angle-averaged linear spectra at z.  Returns the common k grid
# (h/Mpc), the per-μ P(k) for DM & gas (so the sim's μ-binned P(k,costh) can be
# compared slice-by-slice — the direct test of the v_bc anisotropy), and the
# uniform-in-μ average (the spherical P(k) for the isotropic comparison).
function linear_pk(z::Real)
    μs = mu_nodes(NMU)
    P2 = a -> 2π^2 .* a                         # Δ² → P numerator (×1/k_h³ below)
    perμ = NamedTuple[]
    kh = Float64[]
    sumdm = Float64[]; sumgas = Float64[]
    for (i, μ) in enumerate(μs)
        r = transfer_pk(z, μ)
        if i == 1
            kh = r.k ./ HCONST                  # 1/Mpc → h/Mpc
            sumdm = zeros(length(kh)); sumgas = zeros(length(kh))
        else
            length(r.k) == length(kh) || error("k-grid mismatch across costh at z=$z")
        end
        Pdm = P2(r.Δdm) ./ kh.^3
        Pb  = P2(r.Δgas) ./ kh.^3
        push!(perμ, (mu = μ, Pdm = Pdm, Pb = Pb))
        sumdm .+= r.Δdm; sumgas .+= r.Δgas
    end
    Pdm_avg = P2(sumdm ./ NMU) ./ kh.^3
    Pb_avg  = P2(sumgas ./ NMU) ./ kh.^3
    return (k = kh, Pdm = Pdm_avg, Pb = Pb_avg, perμ = perμ)
end

function main()
    mkpath(REPORTS)
    out = joinpath(REPORTS, "cicass_linear_pk.dat")
    open(out, "w") do io
        println(io, "# CICASS analytic linear P(k) at sim output z (box=$BOX Mpc/h, Om=$OMEGA_M, vbc=$VBC, h=$HCONST)")
        println(io, "# transfer.x PRINT_PK; P=2pi^2 Delta^2/k_h^3.  Blocks: angle-averaged")
        println(io, "# 'dm'/'baryon' (over $NMU costh nodes) PLUS per-costh 'dm_mu<costh>',")
        println(io, "# 'baryon_mu<costh>' (costh=k.vbc_hat) so the sim's P(k,costh) anisotropy")
        println(io, "# can be compared slice-by-slice.  block: '@ z=<z> <tag>' then k[h/Mpc] P[(Mpc/h)^3]")
        for z in ZOUT
            lin = linear_pk(z)
            nb = min(6, length(lin.k))
            rbd = sum(@view lin.Pb[2:nb]) / sum(@view lin.Pdm[2:nb])
            # gas anisotropy diagnostic: small-scale P_gas(mu=max)/P_gas(mu=min)
            ns = max(1, length(lin.k) - 6)
            aniso = sum(@view lin.perμ[end].Pb[ns:end]) / sum(@view lin.perμ[1].Pb[ns:end])
            @printf("z=%-7.1f baryon/dm(ls)=%.4f  gas P(mu=%.2f)/P(mu=%.2f) small-scale=%.3f  (%d k-bins)\n",
                    z, rbd, lin.perμ[end].mu, lin.perμ[1].mu, aniso, length(lin.k))
            writeblk(tag, P) = begin
                println(io, "@ z=$(round(z, digits=3)) $tag")
                for i in eachindex(lin.k)
                    @printf(io, "%.6e %.6e\n", lin.k[i], P[i])
                end
            end
            writeblk("dm", lin.Pdm); writeblk("baryon", lin.Pb)
            for s in lin.perμ
                ms = @sprintf("%.3f", s.mu)
                writeblk("dm_mu$ms", s.Pdm); writeblk("baryon_mu$ms", s.Pb)
            end
        end
    end
    @printf("\nwrote CICASS linear spectra at %d redshifts -> %s\n", length(ZOUT), out)
end

main()
