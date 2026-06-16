#!/bin/zsh
setopt NULL_GLOB; cd /Users/tabel/Projects/enzo-dev/EnzoNG.jl
export BACKEND=metal
export RAMSES_LIB_COSMO=/Users/tabel/Projects/mini-ramses/bin64sc_chem/libramses3d.dylib
export RAMSES_LIB=$RAMSES_LIB_COSMO
export CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_DUAL_ENERGY=1 CIC_COMPTON_DRAG=1 CIC_XSPEC=1 CIC_CELLCMP=1 CIC_TAG=_c64
LOG=reports/multicode/c64_ramses.log
/Users/tabel/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_ramses_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
