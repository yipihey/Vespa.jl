#!/usr/bin/env bash
cd /home/tabel/Projects/Vespa.jl
export BACKEND=cpu
export ENZOMODULES_GRID_LIB=/home/tabel/Projects/enzo-dev/EnzoModules/deps/libenzomodules_grid.so
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GRACKLE_DATA_FILE=/home/tabel/Projects/enzo-dev/run/Cooling/CoolingTest_Grackle/CloudyData_noUVB.h5
export CIC_HYDRO=enzo CIC_GRAV=enzo CIC_PARTICLES=enzo
export CIC_BOX=0.128 CIC_NGRID=128 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
# CIC_CHEM_INIT_MATCH=1: force Enzo's species IC (HII,H2I,HDI) + T_gas(μ=1.22) to the SAME
# explicit values RAMSES/Arepo use. WITHOUT it Enzo's MultiSpecies equilibrium init starts the
# gas 75% ionized with HD≈0 (vs x_e=0.047, HD=3e-6) — a different high-z recombination/thermal
# history that makes the low-z cross-code T comparison apples-to-oranges.
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20" CIC_USE_GRACKLE=0 CIC_CHEM_INIT_MATCH=1 CIC_XSPEC=1 CIC_CELLCMP=1 CIC_COMPTON_DRAG=1 CIC_TAG=_c128
LOG=reports/multicode/c128_enzo.log
mkdir -p reports/multicode
~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_highz_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
