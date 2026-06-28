#!/usr/bin/env bash
# Arepo @ 64^3 fixed-resolution (moving mesh, no new zones), DISTRIBUTED over MPI ranks.
# Arepo's hydro/gravity/mesh parallelize natively under mpiexec; the driver realizes the
# CICASS IC on rank 0 into a SHARED run dir, broadcasts the cosmology scalars, every rank
# boots from the shared IC, and the P(k) readback gathers to rank 0 (ArepoLib gather +
# the arepo_bridge MPI helpers). Set NRANKS to taste (default 64).
NRANKS=${NRANKS:-64}
cd /home/tabel/Projects/Vespa.jl
export AREPO_LIB=/home/tabel/Projects/arepo/libarepo3d_cosmo.so
export BACKEND=cpu
export CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cpu CIC_COMPTON_DRAG=1 CIC_TAG=_c64
LOG=reports/multicode/c64_arepo.log
mkdir -p reports/multicode
# NOTE: ArepoLib must be precompiled BEFORE this (64 simultaneous precompiles would race).
mpiexec -n "$NRANKS" ~/.juliaup/bin/julia --project=lib/MultiCode/test \
    lib/MultiCode/examples/cicass_arepo_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
