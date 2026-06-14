# Generate CICASS's LINEAR (analytic, cosmic-variance-free) baryon & DM power
# spectra at each simulation output redshift, so the sim P(k) can be compared to
# linear theory AT THE SAME z (not to a single z=1000 spectrum grown by D(a)²).
# CICASS evolves the CAMB transfer functions forward with the 2-fluid (baryon vs
# CDM, v_bc) physics, so the baryon catch-up after recombination is captured.
#
# CICASS <BASEOUT>.pk columns (from makeCosICs/main.c printInputPower + the G[]
# assignments at generateDisplacements lines 603-619): k, Δ²(G1), Δ²(G5),
# Δ²(G3), Δ²(G7).  The G assignments show G[1]=DM density, G[5]=BARYON density
# (the file header "Deltak_baryons Deltak_dm" is SWAPPED).  Δ²=k³P/2π² →
# P = 2π²Δ²/k³.
#
# Run:  <julia> --project=lib/MultiCode/test lib/MultiCode/examples/cicass_linear_at_outputs.jl

using CICASSLib, Printf

const BOX     = parse(Float64, get(ENV, "CIC_BOX",   "0.128"))
const OMEGA_M = parse(Float64, get(ENV, "CIC_OMEGAM","0.27"))
const VBC     = parse(Float64, get(ENV, "CIC_VBC",   "30.0"))
const NGRID   = parse(Int,     get(ENV, "CIC_NGRID", "128"))
const REPORTS = joinpath(@__DIR__, "..", "..", "..", "reports", "multicode")
# the simulation output redshifts (from the evolve run's checkpoints)
const ZOUT = [parse(Float64, s) for s in split(get(ENV, "CIC_ZOUT",
    "1000.0,520.17,274.41,142.39,75.02"), ",")]

# CICASS .pk: col1=k[h/Mpc], col2=Δ²_DM(G1), col3=Δ²_baryon(G5)  → P=2π²Δ²/k³
function read_pk(path)
    k=Float64[]; Pdm=Float64[]; Pb=Float64[]
    for line in eachline(path)
        startswith(strip(line), "#") && continue
        t = split(line); length(t) >= 3 || continue
        kv = parse(Float64, t[1]); kv > 0 || continue
        c = 2π^2 / kv^3
        push!(k, kv); push!(Pdm, c*parse(Float64,t[2])); push!(Pb, c*parse(Float64,t[3]))
    end
    return (k=k, Pdm=Pdm, Pb=Pb)
end

function main()
    CICASSLib.available() || error("libcicass_capi not found")
    mkpath(REPORTS)
    out = joinpath(REPORTS, "cicass_linear_pk.dat")
    open(out, "w") do io
        println(io, "# CICASS analytic linear P(k) at the sim output redshifts (box=$BOX Mpc/h, Om=$OMEGA_M, vbc=$VBC)")
        println(io, "# col2=DM(G1), col3=baryon(G5); P=2pi^2 Delta^2/k^3.  block: '@ z=<z> <dm|baryon>' then k P")
        for z in ZOUT
            wd = mktempdir()
            spec = CICASSSpec(boxlength=BOX, zstart=z, ngrid=NGRID, vbc=VBC,
                              Omega_m=OMEGA_M, filename="cic_lin")
            CICASSLib.generate(spec; workdir=wd)
            pkf = joinpath(wd, "cic_lin.pk")
            if !isfile(pkf)
                @warn "no .pk produced at z=$z"; continue
            end
            lin = read_pk(pkf)
            # report the baryon/dm ratio at large scale (catch-up diagnostic)
            nb = min(6, length(lin.k))
            rbd = sum(lin.Pb[2:nb])/sum(lin.Pdm[2:nb])
            @printf("z=%-8.1f  baryon/dm (large scale) = %.4f   kmin=%.1f h/Mpc\n", z, rbd, lin.k[1])
            for (tag, P) in (("dm", lin.Pdm), ("baryon", lin.Pb))
                println(io, "@ z=$(round(z,digits=3)) $tag")
                for i in eachindex(lin.k)
                    @printf(io, "%.6e %.6e\n", lin.k[i], P[i])
                end
            end
        end
    end
    @printf("\nwrote CICASS linear spectra at %d redshifts -> %s\n", length(ZOUT), out)
end

main()
