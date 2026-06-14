#!/bin/zsh
setopt NULL_GLOB
cd /Users/tabel/Projects/enzo-dev/EnzoNG.jl
JL=~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia
export AREPO_LIB=$HOME/Projects/arepo/libarepo3d_cosmo.dylib
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$HOME/grackle_install_f32/lib   # f32 grackle ≡ f32-linked reduced lib (ABI match)
export BACKEND=metal
export CIC_NGRID=128 CIC_SUB=2 CIC_ZEND=${CIC_ZEND:-50} CIC_NOUT=${CIC_NOUT:-6}
export CIC_CHEM=${CIC_CHEM:-1} CIC_CHEM_ZMAX=${CIC_CHEM_ZMAX:-2000} CIC_TAG=${CIC_TAG:-_sub2chem}
LOG=/Users/tabel/Projects/enzo-dev/EnzoNG.jl/reports/multicode/arepo_cic${CIC_TAG}.log
$JL --project=lib/MultiCode/test lib/MultiCode/examples/cicass_arepo_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
