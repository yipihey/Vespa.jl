#!/bin/bash
# Build the reduced-chemistry service dylib.  grackle_reduced.c is now
# PRECISION-AGNOSTIC (converts double<->gr_float at the boundary), so it links
# against EITHER the f32 or f64 Grackle fork.  Default is f32 (grackle_install_f32)
# to match the rest of the EnzoNG stack; override with GRACKLE_INSTALL=.../grackle_install.
# IMPORTANT: the runtime DYLD_LIBRARY_PATH must point at the SAME precision install
# this was linked against (ABI: gr_float size) — f32 lib + f64 grackle = garbage.
set -e
cd "$(dirname "$0")"
GR="${GRACKLE_INSTALL:-$HOME/grackle_install_f32}"
HDF5="${HDF5_INSTALL:-/opt/homebrew/opt/hdf5}"
GFLIB="${GFLIB:-/opt/homebrew/lib/gcc/current}"
g++-15 -O2 -fPIC -dynamiclib grackle_reduced.c \
  -I"$GR/include" -L"$GR/lib" -lgrackle \
  -I"$HDF5/include" -L"$HDF5/lib" -lhdf5 -L"$GFLIB" -lgfortran \
  -Wl,-rpath,"$GR/lib" -Wl,-rpath,"$HDF5/lib" -Wl,-rpath,"$GFLIB" \
  -o libgrackle_reduced.dylib
echo "built $(pwd)/libgrackle_reduced.dylib"
