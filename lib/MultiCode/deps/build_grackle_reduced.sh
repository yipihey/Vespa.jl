#!/bin/bash
# Build the reduced-chemistry service dylib (links the installed f64 Grackle fork).
set -e
cd "$(dirname "$0")"
GR="${GRACKLE_INSTALL:-$HOME/grackle_install}"
HDF5="${HDF5_INSTALL:-/opt/homebrew/opt/hdf5}"
GFLIB="${GFLIB:-/opt/homebrew/lib/gcc/current}"
g++-15 -O2 -fPIC -dynamiclib grackle_reduced.c \
  -I"$GR/include" -L"$GR/lib" -lgrackle \
  -I"$HDF5/include" -L"$HDF5/lib" -lhdf5 -L"$GFLIB" -lgfortran \
  -Wl,-rpath,"$GR/lib" -Wl,-rpath,"$HDF5/lib" -Wl,-rpath,"$GFLIB" \
  -o libgrackle_reduced.dylib
echo "built $(pwd)/libgrackle_reduced.dylib"
