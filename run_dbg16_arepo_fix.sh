#!/usr/bin/env bash
cd /home/tabel/Projects/Vespa.jl
export AREPO_LIB=/home/tabel/Projects/arepo/libarepo3d_cosmo.so
export BACKEND=cpu
export CIC_BOX=0.128 CIC_NGRID=16 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=460.0
export CIC_ZOUT="1000,680,460" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cpu CIC_COMPTON_DRAG=1 CIC_MAXDT=0.01 CIC_SOFT_DIV=1 CIC_CHEMDBG=1 CIC_TAG=_dbg16fix
mpiexec -n 1 ~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_arepo_pk.jl > reports/multicode/dbg16_arepo_fix.log 2>&1
echo "===== EXIT $? =====" >> reports/multicode/dbg16_arepo_fix.log
