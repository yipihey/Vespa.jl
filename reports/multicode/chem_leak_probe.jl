# Isolated repro: does ChemistryKernels.solve_chem! leak RSS per call on Metal?
# Mirrors the Arepo driver's per-step chem call on 64³ = 262144 gas cells.
import ChemistryKernels
using Printf
if Symbol(get(ENV, "PROBE_BE", "metal")) === :metal
    using Metal
end
const BE = Symbol(get(ENV, "PROBE_BE", "metal"))
const P  = BE === :metal ? Float32 : Float64
const N  = 262144
rssMB() = Sys.maxrss() / 2^20

base_rho = fill(1.0e-3, N); base_e = fill(2.0, N)
base_HII = fill(1.0e-4, N); base_H2 = fill(1.0e-6, N); base_HD = fill(1.0e-8, N)

@printf("probe backend=%s precision=%s N=%d\n", BE, P, N); flush(stdout)
for it in 1:80
    rho = copy(base_rho); e = copy(base_e)
    HII = copy(base_HII); H2 = copy(base_H2); HD = copy(base_HD)
    ChemistryKernels.solve_chem!(rho, e, HII, H2, HD;
        a_value=0.001, dt=1.0e12, density_units=1.0e-28,
        length_units=3.085678e21, time_units=3.085678e16,
        hubble=71.0, Om=0.27, OL=0.73, fh=0.76,
        deuterium=true, backend=BE, precision=P)
    if it == 1 || it % 5 == 0
        @printf("iter %3d  rss=%.0fMB\n", it, rssMB()); flush(stdout)
    end
end
@printf("DONE backend=%s final_rss=%.0fMB\n", BE, rssMB())
