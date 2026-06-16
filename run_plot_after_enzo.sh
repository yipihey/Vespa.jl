#!/bin/zsh
setopt NULL_GLOB; cd /Users/tabel/Projects/enzo-dev/EnzoNG.jl
EL=reports/multicode/c64_enzo.log
PY=$HOME/Projects/disco-dj-fem/.venv/bin/python
# wait for Enzo's sentinel (file may not exist yet)
until grep -q "===== EXIT" "$EL" 2>/dev/null; do sleep 20; done
sleep 3
echo "[plot] Enzo finished:"; grep -E "wrote [0-9]+ outputs|===== EXIT" "$EL" | tail -2
echo "[plot] data files:"
for f in cicass_enzo_pk_c64.dat cicass_ramses_pk_c64.dat cicass_arepo_pk_c64.dat cicass_linear_pk.dat; do
  n=$(grep -c "^@ z=" reports/multicode/$f 2>/dev/null); echo "  $f: $n blocks"
done
echo "[plot] rendering 3-code comparison ..."
$PY lib/MultiCode/examples/plot_threecode_catchup_c64.py
echo "[plot] EXIT $?"
