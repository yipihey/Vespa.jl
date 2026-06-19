#!/bin/zsh
setopt NULL_GLOB
cd /Users/tabel/Projects/Vespa.jl
JL=~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia
export AREPO_LIB=$HOME/Projects/arepo/libarepo3d_cosmo.dylib
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$HOME/grackle_install_f32/lib
export BACKEND=metal
export CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0
export CIC_ZSTART=1000.0 CIC_ZEND=20.0 CIC_NOUT=7
export CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=metal CIC_COMPTON_DRAG=1 CIC_TAG=""
LOG=/Users/tabel/Projects/Vespa.jl/reports/multicode/cmp_arepo_drag.log
$JL --project=lib/MultiCode/test lib/MultiCode/examples/cicass_arepo_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
