#!/usr/bin/env bash
# Arepo c128 UNIFORM-BARYON test (CIC_UNIFORM_BARYONS=1): δ_b=0 at IC, DM keeps full
# CICASS structure. Isolates the large-scale over-growth — if Arepo DM still over-grows
# vs the CICASS-linear D(a) (which Enzo/RAMSES match to ~1%) with smooth baryons, the
# cause is the gravity/IC (TreePM/softening/comoving), NOT baryon back-reaction.
# Separate TAG → separate workdir + output files (concurrent with the cellcmp run).
NRANKS=${NRANKS:-16}
cd /home/tabel/Projects/Vespa.jl
export AREPO_LIB=/home/tabel/Projects/arepo/libarepo3d_cosmo.so
export BACKEND=cpu
export CIC_BOX=0.128 CIC_NGRID=128 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20"
export CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cpu CIC_COMPTON_DRAG=1
export CIC_UNIFORM_BARYONS=1 CIC_XSPEC=1 CIC_MAXDT=0.08 CIC_TAG=_c128ub
LOG=reports/multicode/c128ub_arepo.log
mkdir -p reports/multicode
mpiexec -n "$NRANKS" ~/.juliaup/bin/julia --project=lib/MultiCode/test \
    lib/MultiCode/examples/cicass_arepo_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
