#!/bin/bash
# Build the ChemistryKernels verification oracle (libchem_oracle.dylib).
# Links grackle's own analytic rate/cooling functions (units=1.0 -> exact CGS).
# Default = the f64 grackle install (bit-tight per-rate oracle); override
# GRACKLE_INSTALL=$HOME/grackle_install_f32 for the f32 cross-check, or point it
# at a high-temperature-bin build for the tight one-zone integration test.
set -e
cd "$(dirname "$0")"
GR="${GRACKLE_INSTALL:-$HOME/grackle_install}"
HDF5="${HDF5_INSTALL:-/opt/homebrew/opt/hdf5}"
GFLIB="${GFLIB:-/opt/homebrew/lib/gcc/current}"
g++-15 -O2 -fPIC -dynamiclib chem_oracle.c \
  -I"$GR/include" -L"$GR/lib" -lgrackle \
  -I"$HDF5/include" -L"$HDF5/lib" -lhdf5 -L"$GFLIB" -lgfortran \
  -Wl,-rpath,"$GR/lib" -Wl,-rpath,"$HDF5/lib" -Wl,-rpath,"$GFLIB" \
  -o libchem_oracle.dylib
echo "built $(pwd)/libchem_oracle.dylib  (grackle: $GR)"
