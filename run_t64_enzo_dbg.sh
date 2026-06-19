#!/usr/bin/env bash
cd /home/tabel/Projects/Vespa.jl
export BACKEND=cuda
export ENZOMODULES_GRID_LIB=/home/tabel/Projects/enzo-dev/EnzoModules/deps/libenzomodules_grid.so
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GRACKLE_DATA_FILE=/home/tabel/Projects/enzo-dev/run/Cooling/CoolingTest_Grackle/CloudyData_noUVB.h5
export CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=680.0
export CIC_MAXEXP=0.1
export CIC_ZOUT="1000,680" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_USE_GRACKLE=0 CIC_COMPTON_DRAG=1 CIC_XSPEC=1 CIC_CELLCMP=0 CIC_TAG=_t64_dbg CIC_CHEM_INIT_MATCH=1
export CHEM_DEBUG=1
LOG=reports/multicode/t64_enzo_dbg.log
mkdir -p reports/multicode
~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_highz_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
