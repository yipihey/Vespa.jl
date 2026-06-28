#!/usr/bin/env bash
cd /home/tabel/Projects/Vespa.jl
export BACKEND=cuda
# CUDA GPU build (NVHPC/nvfortran, sm_86, hydro+gravity on A6000) — set as the :cosmo flavor.
# f32 (NPRE=4) is the GPU build we use. The cold-gas low-z T problem is the KDKD leapfrog
# half-step velocity staggering of the injected streaming bulk (see driver workaround), NOT
# a precision issue to be papered over with f64.
export RAMSES_LIB_COSMO=/home/tabel/Projects/mini-ramses-metal/bin64sc_cuda/libramses3d.so
export RAMSES_LIB=$RAMSES_LIB_COSMO
# Keep the CPU cosmo build accessible as :cuda_cosmo (for CPU/GPU diff if needed)
export RAMSES_LIB_CUDA_COSMO=/home/tabel/Projects/mini-ramses-metal/bin64sc_chem/libramses3d.so
# NVHPC runtime libs needed at dlopen time
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CIC_BOX=0.128 CIC_NGRID=256 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
# UNBOOSTED (gas streams at +v_bc, DM at rest) — matches Enzo's frame and gives the verified
# streaming anisotropy (RAMSES gas P(k,cosθ) z=20 0.263 vs CICASS-lin 0.220). The boosted
# frame was tried for the low-z cold-T but did NOT help (T cold even with gas at rest) and
# regressed the anisotropy (boost_particles! left DM bulk=0), so the cold-T is frame-
# independent (local-infall f32 dual-energy staggering), not the bulk-velocity offset.
# Cosmic-expansion da/a cap (mini-ramses newdt_fine.f90 reads CIC_MAXEXP; default 0.1 too coarse
# → DM under-grows: 0.1→0.68, 0.02→0.80 ls z=20). 0.01 converges the DM growth toward linear.
export CIC_MAXEXP=0.01
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cuda CIC_DUAL_ENERGY=1 CIC_COMPTON_DRAG=1 CIC_XSPEC=1 CIC_CELLCMP=0 CIC_TAG=_c256
LOG=reports/multicode/c256_ramses.log
mkdir -p reports/multicode
~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_ramses_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
