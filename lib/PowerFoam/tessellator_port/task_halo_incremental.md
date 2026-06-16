# Task Packet: Ghost Halo And Incremental Rebuild

Output file: `lib/PowerFoam/tessellator_port/halo_incremental.md`

Read only:

- `arepo_tessellator_port_brief.md`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_ghost_search.c`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_dynamic_update.c`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_exchange.c`
- `/Users/tabel/Projects/arepo/src/main/allvars.h`

Deliverable:

- Summarize AREPO ghost/halo search and connectivity update semantics.
- Identify what matters for single-rank periodic GPU first, and what can wait.
- Propose dirty-cell/dirty-face halo rules for hierarchical timestepping.
- Max 160 lines.

Do not edit code.

