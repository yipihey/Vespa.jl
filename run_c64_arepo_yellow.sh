#!/usr/bin/env bash
cd /home/tabel/Projects/Vespa.jl
export AREPO_LIB=/home/tabel/Projects/arepo/libarepo3d_cosmo.so
# NOTE: do NOT prepend /usr/lib64 — libarepo's deps (gsl/hdf5/gmp) already resolve via
# the loader's default /lib64, and /usr/lib64 on LD_LIBRARY_PATH shadows Julia's bundled
# libpcre2-8 with the system one → "PCRE compilation error: unrecognised compile-time option".
export BACKEND=cpu
export CIC_BOX=0.128 CIC_NGRID=64 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20" CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cpu CIC_COMPTON_DRAG=1 CIC_TAG=_c64
LOG=reports/multicode/c64_arepo.log
mkdir -p reports/multicode
~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_arepo_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
