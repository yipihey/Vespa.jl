#!/usr/bin/env bash
cd /home/tabel/Projects/enzo-dev/EnzoNG.jl
export BACKEND=cuda
# CUDA GPU build (NVHPC/nvfortran, sm_86, hydro+gravity on A6000) — set as the :cosmo flavor
export RAMSES_LIB_COSMO=/home/tabel/Projects/mini-ramses-metal/bin64sc_cuda/libramses3d.so
export RAMSES_LIB=$RAMSES_LIB_COSMO
# Keep the CPU cosmo build accessible as :cuda_cosmo (for CPU/GPU diff if needed)
export RAMSES_LIB_CUDA_COSMO=/home/tabel/Projects/mini-ramses-metal/bin64sc_chem/libramses3d.so
# NVHPC runtime libs needed at dlopen time
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cuda CIC_DUAL_ENERGY=1 CIC_COMPTON_DRAG=1 CIC_XSPEC=1 CIC_CELLCMP=1 CIC_TAG=_c64
LOG=reports/multicode/c64_ramses.log
mkdir -p reports/multicode
~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_ramses_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
