#!/bin/zsh
setopt NULL_GLOB; cd /Users/tabel/Projects/Vespa.jl
AL=reports/multicode/c64_arepo.log
# wait for Arepo's sentinel
until grep -q "===== EXIT" "$AL" 2>/dev/null; do sleep 15; done
sleep 3
# make sure the Arepo julia is gone (free the GPU) before starting Enzo
pkill -f cicass_arepo_pk.jl 2>/dev/null; sleep 3
echo "[chain] Arepo finished:"; grep -E "wrote [0-9]+ Arepo|===== EXIT" "$AL" | tail -2
echo "[chain] launching Enzo c64 ..."
./run_c64_enzo.sh
echo "[chain] Enzo c64 done."
