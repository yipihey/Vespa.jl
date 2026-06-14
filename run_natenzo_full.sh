#!/bin/zsh
setopt NULL_GLOB
cd /Users/tabel/Projects/enzo-dev/EnzoNG.jl
JL=~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia
export BACKEND=metal
export ENZOMODULES_GRID_LIB=/Users/tabel/Projects/enzo-dev/EnzoModules/deps/libenzomodules_grid_f32.dylib
export DYLD_LIBRARY_PATH=$HOME/grackle_install_f32/lib:/opt/homebrew/opt/hdf5/lib
export CIC_HYDRO=enzo CIC_GRAV=enzo
export CIC_ZEND=20 CIC_NOUT=7 CIC_TAG=_natenzo
LOG=/Users/tabel/Projects/enzo-dev/EnzoNG.jl/reports/multicode/natenzo_full.log
$JL --project=lib/MultiCode/test lib/MultiCode/examples/cicass_highz_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
