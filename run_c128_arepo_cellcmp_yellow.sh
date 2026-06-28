#!/usr/bin/env bash
# Arepo c128 rerun WITH the cell-by-cell dump (CIC_CELLCMP=1) + xspec, to join
# plot_cicass_cellcmp.py (Enzo/RAMSES/Arepo). 16 ranks (over-decomposition fastest),
# MAXDT=0.08 (the accuracy-validated canonical step). Regenerates the canonical _c128 data.
NRANKS=${NRANKS:-16}
cd /home/tabel/Projects/Vespa.jl
export AREPO_LIB=/home/tabel/Projects/arepo/libarepo3d_cosmo.so
export BACKEND=cpu
export CIC_BOX=0.128 CIC_NGRID=128 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20"
export CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cpu CIC_COMPTON_DRAG=1
# CIC_SOFT_DIV=1: gravitational softening = mean particle spacing = 1 cell, MATCHING the ~1-cell
# gravity resolution of Enzo/RAMSES. The default /5 makes Arepo's softening 5× finer → huge excess
# small-scale clustering (Arepo/Enzo P(k) up to 13× at high k in this 128 kpc/h box, where every
# resolved scale sits above the softening) → spurious "over-growth". /1 = fair cross-code gravity.
export CIC_XSPEC=1 CIC_CELLCMP=1 CIC_MAXDT=0.08 CIC_SOFT_DIV=1 CIC_TAG=_c128
LOG=reports/multicode/c128_arepo_cellcmp.log
mkdir -p reports/multicode
mpiexec -n "$NRANKS" ~/.juliaup/bin/julia --project=lib/MultiCode/test \
    lib/MultiCode/examples/cicass_arepo_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
