#!/usr/bin/env bash
# CONTROLLED single-step test (Enzo): identical CICASS realization, FORCE the species IC to
# match RAMSES (CIC_CHEM_INIT_MATCH=1 → fields 9/14/18 = ρ·{x_e,1e-6,6.8e-5·x_e}, T_gas μ=1.22),
# dump full cell state at the IC (z=1000) and after ONE fixed step (z=990). Lets us confirm
# Enzo & RAMSES start identical and see exactly where one step diverges.
cd /home/tabel/Projects/Vespa.jl
export BACKEND=cpu
export ENZOMODULES_GRID_LIB=/home/tabel/Projects/enzo-dev/EnzoModules/deps/libenzomodules_grid.so
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export GRACKLE_DATA_FILE=/home/tabel/Projects/enzo-dev/run/Cooling/CoolingTest_Grackle/CloudyData_noUVB.h5
export CIC_HYDRO=enzo CIC_GRAV=enzo CIC_PARTICLES=enzo
export CIC_BOX=0.128 CIC_NGRID=128 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=990.0
export CIC_ZOUT="1000,990" CIC_USE_GRACKLE=0 CIC_CHEM_INIT_MATCH=1 CIC_XSPEC=1 CIC_CELLCMP=1 CIC_COMPTON_DRAG=1 CIC_TAG=_1s
LOG=reports/multicode/c128_enzo_1step.log
mkdir -p reports/multicode
~/.juliaup/bin/julia --project=lib/MultiCode/test lib/MultiCode/examples/cicass_highz_pk.jl > "$LOG" 2>&1; echo "===== EXIT $? =====" >> "$LOG"
