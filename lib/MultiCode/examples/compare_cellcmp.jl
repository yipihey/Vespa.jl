#!/usr/bin/env julia
using Statistics: median
using Printf

# compare_cellcmp.jl — cell-by-cell comparison of RAMSES vs Enzo cellcmp outputs.
# Binary format (both codes): Int64 N, then N³ Float64 each of: rho, xHII, fH2, fHD, T
# Usage:
#   julia compare_cellcmp.jl
# Env vars:
#   CIC_TAG=_t64_me3   (default)
#   CIC_REPORTS=<dir>
#   CIC_ZOUT=z1,z2,...  redshifts to compare (default: 1000,680,460,315,215,145,100,65,45,30,20)
#   CIC_CMP_RTOL=0.01   relative tolerance (1%)
#
# Matches ramses_cellcmp$(TAG)_z<Z>.bin vs enzo_cellcmp$(TAG)_z<Z>.bin by EXACT z.
# For each redshift, reports median and max relative error per field.
# Exit 0 if all within RTOL; exit 1 on divergence; exit 2 on missing files.

REPORTS = get(ENV, "CIC_REPORTS",
    joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
TAG  = get(ENV, "CIC_TAG", "_t64_me3")
RTOL = parse(Float64, get(ENV, "CIC_CMP_RTOL", "0.01"))
ZOUT = [parse(Int, s) for s in split(get(ENV, "CIC_ZOUT", "1000,680,460,315,215,145,100,65,45,30,20"), ",")]

function read_cellcmp(path)
    open(path, "r") do io
        N     = read(io, Int64)
        ncell = N^3
        rho   = read!(io, Vector{Float64}(undef, ncell))
        xHII  = read!(io, Vector{Float64}(undef, ncell))
        fH2   = read!(io, Vector{Float64}(undef, ncell))
        fHD   = read!(io, Vector{Float64}(undef, ncell))
        T     = read!(io, Vector{Float64}(undef, ncell))
        return (; N, rho, xHII, fH2, fHD, T)
    end
end

function relrr(a, b)
    denom = 0.5 .* (abs.(a) .+ abs.(b)) .+ 1e-30
    return abs.(a .- b) ./ denom
end

fields = (:rho, :xHII, :fH2, :fHD, :T)

println("Cell-by-cell comparison: RAMSES$(TAG) vs Enzo$(TAG)")
println("RTOL=$(RTOL*100)%  |  comparing $(length(ZOUT)) redshifts: $(ZOUT)")
println()
@printf("%-8s  %-10s %-10s %-10s %-10s %-10s  %-10s %-10s %-10s %-10s %-10s\n",
        "z",
        "rho_med", "xHII_med", "fH2_med", "fHD_med", "T_med",
        "rho_max", "xHII_max", "fH2_max", "fHD_max", "T_max")
println(repeat("-", 115))

any_fail = false
nmissing = 0

for z in ZOUT
    rpath = joinpath(REPORTS, "ramses_cellcmp$(TAG)_z$(z).bin")
    epath = joinpath(REPORTS, "enzo_cellcmp$(TAG)_z$(z).bin")
    if !isfile(rpath) || !isfile(epath)
        @printf("  z=%-6d  MISSING: %s\n", z,
            !isfile(rpath) ? basename(rpath) : basename(epath))
        global nmissing += 1
        continue
    end

    R = read_cellcmp(rpath)
    E = read_cellcmp(epath)
    if R.N != E.N
        @printf("  z=%-6d  N mismatch R=%d E=%d\n", z, R.N, E.N)
        continue
    end

    meds = Float64[]; maxs = Float64[]
    for f in fields
        rr = relrr(getfield(R, f), getfield(E, f))
        push!(meds, median(rr))
        push!(maxs, maximum(rr))
    end

    fail = any(m > RTOL for m in maxs)
    if fail; global any_fail = true; end
    marker = fail ? " ← FAIL" : ""
    @printf("%-8d  %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f  %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f%s\n",
            z,
            meds[1], meds[2], meds[3], meds[4], meds[5],
            maxs[1], maxs[2], maxs[3], maxs[4], maxs[5], marker)
end

println()
if nmissing > 0
    println("$nmissing file(s) missing — run both simulations first")
    exit(2)
elseif any_fail
    println("RESULT: DIVERGENCE — max relative error > $(RTOL*100)% in at least one field/redshift")
    exit(1)
else
    println("RESULT: All outputs agree within $(RTOL*100)%  ✓")
    exit(0)
end
