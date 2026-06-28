#!/usr/bin/env bash
cd /home/tabel/Projects/Vespa.jl
export BACKEND=cuda
export RAMSES_LIB_COSMO=/home/tabel/Projects/mini-ramses-metal/bin64sc_cuda/libramses3d.so
export RAMSES_LIB=$RAMSES_LIB_COSMO
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CIC_BOX=0.128 CIC_NGRID=16 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=460.0
export CIC_ZOUT="1000,680,460" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cuda CIC_DUAL_ENERGY=1 CIC_COMPTON_DRAG=1 CIC_MAXEXP=0.01 CIC_CHEMDBG=1 CIC_TAG=_dbg16
~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_ramses_pk.jl > reports/multicode/dbg16_ramses.log 2>&1
echo "===== EXIT $? =====" >> reports/multicode/dbg16_ramses.log
