#!/bin/zsh
setopt NULL_GLOB; cd /Users/tabel/Projects/Vespa.jl
export BACKEND=metal
export ENZOMODULES_GRID_LIB=/Users/tabel/Projects/enzo-dev/EnzoModules/deps/libenzomodules_grid_f32.dylib
export DYLD_LIBRARY_PATH=$HOME/grackle_install_f32/lib:/opt/homebrew/opt/hdf5/lib
export CIC_HYDRO=enzo CIC_GRAV=enzo CIC_PARTICLES=enzo
export CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20" CIC_XSPEC=1 CIC_CELLCMP=1 CIC_COMPTON_DRAG=1 CIC_TAG=_c64
LOG=reports/multicode/c64_enzo.log
/Users/tabel/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_highz_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
