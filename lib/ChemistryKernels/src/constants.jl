# Physical constants in CGS — byte-identical to grackle's src/clib/phys_constants.h
# so the ported analytic formulas reproduce grackle's `kN_rate(T, 1.0, cd)` values
# to floating-point round-off.  Stored as Float64 literals; convert with `T(...)`
# at the point of use (never round the literal itself).

const KBOLTZ   = 1.3806504e-16      # Boltzmann constant [erg/K]
const MH       = 1.67262171e-24     # hydrogen mass [g]
const ME       = 9.10938215e-28     # electron mass [g]
const PI_G     = 3.14159265358979323846
const CLIGHT   = 2.99792458e10      # speed of light [cm/s]
const GRAVCONST = 6.67428e-8        # gravitational constant [cm^3 g^-1 s^-2]
const SOLARMASS = 1.9891e33         # solar mass [g]
const MPC      = 3.0857e24          # megaparsec [cm]
const KPC      = 3.0857e21
const PC       = 3.0857e18

# eV ↔ K conversion used pervasively in the rate fits (T_ev = T / TEV_PER_K).
# grackle uses the literal 11605.0 in rate_functions.c (NOT kboltz/eV) — match it.
const TEV_PER_K = 11605.0

# Hydrogen mass fraction default (grackle HydrogenFractionByMass); the reduced
# network uses fh passed from the host, but this is the cosmic default.
const FH_DEFAULT = 0.76

# Deuterium-to-hydrogen seeding ratio used by the reduced wrapper (2 * 3.4e-5).
const DTOH_SEED = 6.8e-5

# A tiny floor mirroring grackle's `tiny` (1e-20) for species/abundances.
const TINY = 1e-20
