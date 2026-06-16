#!/bin/zsh
setopt NULL_GLOB
cd /Users/tabel/Projects/enzo-dev/EnzoNG.jl
JL=~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia
export BACKEND=metal
export RAMSES_LIB_COSMO=/Users/tabel/Projects/mini-ramses/bin64sc_chem_metal/libramses3d_metal.dylib
export RAMSES_LIB=$RAMSES_LIB_COSMO
export RAMSES_METALLIB=/Users/tabel/Projects/mini-ramses/bin64sc_chem_metal/ramses_kernels.metallib
export CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0
export CIC_ZSTART=1000.0 CIC_ZEND=20.0 CIC_NOUT=7
export CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_DUAL_ENERGY=1 CIC_COMPTON_DRAG=1 CIC_XSPEC=1 CIC_CELLCMP=1 CIC_TAG=_metal
LOG=/Users/tabel/Projects/enzo-dev/EnzoNG.jl/reports/multicode/cmp_ramses_metal.log
$JL --project=lib/MultiCode/test lib/MultiCode/examples/cicass_ramses_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
