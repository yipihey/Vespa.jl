#!/usr/bin/env bash
# Enzo GPU mode @ 64^3 fixed resolution (no AMR): Enzo hierarchy is the substrate
# (ghost fill + storage) but hydro/gravity/chem run on the A6000 via the Julia kernel
# paths (CIC_HYDRO=julia=PPMKernels-CUDA, CIC_GRAV=julia=PoissonKernels-CUDA,
# CIC_CHEM_BACKEND defaults to the backend = cuda).  Tag _c64g keeps it distinct from
# the CPU native-Enzo run (_c64) so the two can be compared.
cd /home/tabel/Projects/Vespa.jl
export BACKEND=cuda
export ENZOMODULES_GRID_LIB=/home/tabel/Projects/enzo-dev/EnzoModules/deps/libenzomodules_grid.so
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GRACKLE_DATA_FILE=/home/tabel/Projects/enzo-dev/run/Cooling/CoolingTest_Grackle/CloudyData_noUVB.h5
export CIC_BOX=0.128 CIC_NGRID=256 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
# Canonical Enzo for the comparison = GPU (PPMKernels f32 dual-energy) — like-for-like vs RAMSES-GPU.
# CIC_CHEM_INIT_MATCH=1 → identical species/T IC as RAMSES; CIC_CELLCMP=0 → cell-by-cell T dump.
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_USE_GRACKLE=0 CIC_CHEM_INIT_MATCH=1 CIC_XSPEC=1 CIC_CELLCMP=0 CIC_COMPTON_DRAG=1 CIC_TAG=_c256
LOG=reports/multicode/c256_enzo_gpu.log
mkdir -p reports/multicode
~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_highz_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
