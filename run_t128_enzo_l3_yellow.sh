#!/usr/bin/env bash
# CICASS cosmology, 128^3 root + 3 levels of AMR on the GPU (Vespa/Enzo), z=1000 -> 20.
# NOTE: AMR here is PROJECTION-ONLY (FluxCorrection=0) — the conservative coarse-fine
# reflux for the GPU split-MUSCL hydro is not bit-exact yet (see docs/adr/0003).
cd /home/tabel/Projects/Vespa.jl
export BACKEND=cuda
export ENZOMODULES_GRID_LIB=/home/tabel/Projects/enzo-dev/EnzoModules/deps/libenzomodules_grid.so
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GRACKLE_DATA_FILE=/home/tabel/Projects/enzo-dev/run/Cooling/CoolingTest_Grackle/CloudyData_noUVB.h5
export CIC_BOX=0.128 CIC_NGRID=128 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
export CIC_MAXLEVEL=3 CIC_MAXEXP=0.1
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_USE_GRACKLE=0 CIC_COMPTON_DRAG=1 CIC_XSPEC=1 CIC_CELLCMP=1 CIC_TAG=_t128_l3
LOG=reports/multicode/t128_enzo_l3.log
mkdir -p reports/multicode
~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_highz_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
