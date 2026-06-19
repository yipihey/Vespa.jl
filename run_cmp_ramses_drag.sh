#!/bin/zsh
setopt NULL_GLOB
cd /Users/tabel/Projects/Vespa.jl
JL=~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia
export BACKEND=metal
# cosmo (supercomoving) RAMSES, no chem — DM+baryon density growth is gravitational
# bin64sc_chem carries the full capi (ramses_get_units etc.); chem stays OFF (CIC_CHEM=0)
export RAMSES_LIB=/Users/tabel/Projects/mini-ramses/bin64sc_chem/libramses3d.dylib
export RAMSES_LIB_COSMO=/Users/tabel/Projects/mini-ramses/bin64sc_chem/libramses3d.dylib
export CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0
export CIC_ZSTART=1000.0 CIC_ZEND=20.0 CIC_NOUT=7 CIC_NEARLY=4
export CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_XSPEC=1 CIC_CELLCMP=1 CIC_COMPTON_DRAG=1 CIC_TAG=""
LOG=/Users/tabel/Projects/Vespa.jl/reports/multicode/cmp_ramses_drag.log
$JL --project=lib/MultiCode/test lib/MultiCode/examples/cicass_ramses_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
