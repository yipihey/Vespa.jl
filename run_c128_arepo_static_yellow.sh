#!/usr/bin/env bash
# Arepo with a FIXED (static) Voronoi mesh — built once in comoving coords, gas flows through it
# Eulerian-style (libarepo3d_cosmo_static.so = VORONOI_STATIC_MESH +
# VORONOI_STATIC_MESH_DO_DOMAIN_DECOMPOSITION; same source = has the radiation H(a) fix).
# Identical settings to the moving canonical (SOFT_DIV=1, MAXDT=0.08, chem, radiation) so the
# ONLY difference is the mesh — isolates moving-mesh effects on P(k) AND the thermal history.
NRANKS=${NRANKS:-16}
cd /home/tabel/Projects/Vespa.jl
export AREPO_LIB=/home/tabel/Projects/arepo/libarepo3d_cosmo_static.so
export BACKEND=cpu
export CIC_BOX=0.128 CIC_NGRID=128 CIC_OMEGAM=0.27 CIC_VBC=30.0 CIC_ZSTART=1000.0 CIC_ZEND=20.0
export CIC_ZOUT="1000,680,460,315,215,145,100,65,45,30,20"
export CIC_CHEM=1 CIC_CHEM_ENGINE=kernels CIC_CHEM_BACKEND=cpu CIC_COMPTON_DRAG=1
# CIC_STATIC_MESH=1 omits the moving-mesh-only params (CellShapingSpeed/CellMaxAngleFactor)
# that the VORONOI_STATIC_MESH build rejects.
export CIC_STATIC_MESH=1
export CIC_XSPEC=1 CIC_CELLCMP=1 CIC_MAXDT=0.08 CIC_SOFT_DIV=1 CIC_TAG=_c128static
LOG=reports/multicode/c128_arepo_static.log
mkdir -p reports/multicode
mpiexec -n "$NRANKS" ~/.juliaup/bin/julia --project=lib/MultiCode/test \
    lib/MultiCode/examples/cicass_arepo_pk.jl > "$LOG" 2>&1
echo "===== EXIT $? =====" >> "$LOG"
