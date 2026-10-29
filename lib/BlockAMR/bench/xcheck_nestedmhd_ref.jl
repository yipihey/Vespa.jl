# Reference half of the BlockAMR-vs-NestedMHD scheme cross-check (bench/xcheck_nestedmhd.jl).
# Run in NestedMHD's OWN environment (its KernelAbstractions pin differs from BlockAMR's):
#   git clone git@github.com:yipihey/NestedTurbulence.git
#   julia --project=NestedTurbulence/core lib/BlockAMR/bench/xcheck_nestedmhd_ref.jl [outdir]
# Writes nested_ot_{glm,ct}_{hlld,hll}.jls: single-level 64² Orszag–Tang (Float64) at t = 0.1.
using NestedMHD, Serialization
const NM = NestedMHD
outdir = get(ARGS, 1, tempdir())
N = 64; tend = 0.1; b0 = 1 / sqrt(4π)
ot(x, y, z) = (25/(36π), -sin(2π*y), sin(2π*x), 0.0, 5/(12π), -b0*sin(2π*y), b0*sin(4π*x), 0.0)
aot(x, y, z) = (0.0, 0.0, b0 * (cos(4π*x)/(4π) + cos(2π*y)/(2π)))
for mode in (:glm, :ct), riemann in (:hlld, :hll)
    H = NM.Hierarchy(; N0 = (N, N, 1), mode, riemann, T = Float64)
    mode === :ct ? NM.init!(H, ot; vecpot = aot) : NM.init!(H, ot)
    NM.evolve!(H, tend)
    P = H.patches[1]
    serialize(joinpath(outdir, "nested_ot_$(mode)_$(riemann).jls"),
              (U = Array(P.U), bx = Array(P.bx), by = Array(P.by), nsteps = H.nsteps[1]))
    println(mode, " ", riemann, ": ", H.nsteps[1], " steps → ", outdir)
end
