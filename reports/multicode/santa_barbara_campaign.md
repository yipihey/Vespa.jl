# Santa Barbara campaign harness

Track C1 now has a repeatable Vespa-local runner:

```sh
BACKEND=cpu SB_MAXCYC=64 SB_STOP_ON_REFINEMENT=1 \
  <julia> --project=lib/PPMKernels/test lib/EnzoLib/examples/sb_metal_amr.jl
```

The same command with `BACKEND=metal` runs the Metal-f32 path.  Both paths use
`Float32`; CPU-f32 is the faithful reference for the Metal trajectory.

Each run writes a campaign directory under
`lib/EnzoLib/examples/sb_campaign_out/` unless `SB_CAMPAIGN_OUT` is set.  The
directory contains:

- `diagnostics.csv` — cycle, time, grid counts on levels 0..2, max density,
  mass drift, refinement-onset flag, and wall-clock seconds per cycle.
- `summary.md` — the same diagnostics in a compact table for reports.

The intended certification sequence is:

1. CPU-f32 with `SB_STOP_ON_REFINEMENT=1`, to identify the first live AMR
   cycle and confirm mass drift stays controlled.
2. Metal-f32 with identical settings, comparing cycle-by-cycle grid counts,
   density extrema, and mass drift against CPU-f32.
3. Longer CPU-f32 and Metal-f32 runs with `SB_STOP_ON_REFINEMENT=0`, once the
   refinement-onset pair agrees.

The measured refinement-onset pair is now archived under:

- `reports/multicode/sb_campaign_cpu_f32/`
- `reports/multicode/sb_campaign_metal_f32/`
- `reports/multicode/santa_barbara_campaign_comparison.md`

Both CPU-f32 and Metal-f32 hit the first level-1 grid at cycle 42 with matching
grid counts (`[1, 1, 0]`).  The 43 overlapping rows have matching cycle numbers,
matching refinement state, zero reported `rho_max` drift at printed precision,
and a 3.142× Metal speedup including warmup.

After two runs finish, compare them with:

```sh
<julia> lib/EnzoLib/examples/sb_compare_campaigns.jl \
  cpu-run/diagnostics.csv metal-run/diagnostics.csv comparison.md
```

The comparator reports cycle/grid agreement, maximum time drift, maximum
relative `rho_max` drift, mass-drift differences, and candidate speedup.
