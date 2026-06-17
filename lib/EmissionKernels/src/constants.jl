# Minimal CGS constants needed by the radiative-channel physics. EmissionKernels is
# the FOUNDATION layer (ChemistryKernels depends on it), so it cannot import
# ChemistryKernels' constants — these are the self-contained subset the moved cooling
# code references. Stored as Float64 literals; convert with `T(...)` at use.
const KBOLTZ = 1.3806504e-16      # Boltzmann constant [erg/K]
const TINY   = 1.0e-20            # species/abundance floor (HD low-density guard)
