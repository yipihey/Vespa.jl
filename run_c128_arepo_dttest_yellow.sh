#!/usr/bin/env bash
# QUICK timestep test (Arepo): z=1000→460 only, RAMSES-matched step (MAXDT=0.01 → Δln a 0.01 vs the
# canonical 0.03), + the _dt_phys radiation fix. Tests whether the high-z recombination overshoot
# (x_e ~23% low at z=680 → weak Compton → over-cool) is caused by the coarse chem step holding the
# CMB a_value fixed over too-large Δz. Compare arepo_cellcmp_dttest_z{680,460} x_e/T to Enzo/RAMSES.
NRANKS=${NRANKS:-16}
cd /home/tabel/Projects/Vespa.jl
export AREPO_LIB=/home/tabel/Projects/arepo/libarepo3d_cosmo.so
export BACKEND=cpu
export CIC_BOX=0.128 CIC_NGRID=128 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=460.0
export CIC_ZOUT="1000,680,460"
export CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cpu CIC_COMPTON_DRAG=1
export CIC_XSPEC=0 CIC_CELLCMP=1 CIC_MAXDT=0.01 CIC_SOFT_DIV=1 CIC_TAG=_dttest
LOG=reports/multicode/c128_arepo_dttest.log
mkdir -p reports/multicode
mpiexec -n "$NRANKS" ~/.juliaup/bin/julia --project=lib/MultiCode/test \
    lib/MultiCode/examples/cicass_arepo_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
