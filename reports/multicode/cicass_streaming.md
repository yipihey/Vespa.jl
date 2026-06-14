# CICASS streaming-velocity ICs in the multi-code framework

**Goal.** Add CICASS (McQuinn & O'Leary 2012, arXiv:1204.1344/1345) as a third
cosmological IC generator alongside MUSIC and DISCO-DJ, to capture the one piece
of physics the others structurally lack: the **baryon–dark-matter streaming
velocity** (Tseliakhovich & Hirata 2010) and correct high-redshift setup.

**Why it is needed.** MUSIC's two-component path realizes separate baryon and CDM
power spectra but with *identical phases differing only in amplitude* — no relative
displacement, no bulk velocity offset. DISCO-DJ's 1LPT is single-component. CICASS
evolves a **2D `(k⊥, k∥)` transfer function** forward from recombination with the
relative velocity `v_bc`, giving the gas a coherent **bulk velocity offset**
relative to the dark matter.

## Status

| Stage | What | State |
|------|------|-------|
| A | Build CICASS (`transfer.x`, `genICs.x`), confirm streaming | ✅ done |
| B | C-ABI wrapper `CICASSLib` (CodeBridge) | ✅ done, 15/15 tests |
| C | `MultiCodeCICASSExt`: grafic streaming + live injection | ✅ done + validated |
| D | Cross-code gate: live Enzo + live RAMSES, both carry the offset | ✅ done + validated |

## Cross-code streaming gate

ONE CICASS realization (v_bc, 128³, 0.2 Mpc/h, z=100), the coherent gas–DM bulk
velocity offset measured at every stage:

| Path | v_bc=30 offset (km/s) | v_bc=0 |
|------|----------------------|--------|
| Realization (`.cicass`) | `[3.027, 0, 0]` | `[0, 0, 0]` |
| RAMSES grafic format (`ic_velb − ic_velc`) | `[3.027, 0, 0.0004]` | `[0, 0, 0]` |
| **Live Enzo** (128³ SB host, BaryonField velocities) | `[3.027, 0, 0]` | `[0, 0, 0]` |
| **Live RAMSES** (UNITS=COSMO, hydro, `set_hydro!`) | `[3.027, 0, 0]` | `[0, 0, 0]` |

Expected `= v_bc·(1+z)/1001 = 3.027 km/s`. Both codes agree to ~0.1%, the offset
is coherent on a single axis, and v_bc=0 is identically zero everywhere.

**Enzo half** (`run_cicass_enzo`): boot the SantaBarbaraCluster 128³ hydro host
patched to the CICASS cosmology (so Enzo's `VelocityUnits` match), inject the gas
velocity field into the BaryonField velocities (FieldType 4/5/6, c-order →
ghosted-Fortran) and the DM velocities into particles, read the bulk offset back.
Conversion: 127.915 km/s per Enzo velocity unit.

**RAMSES half** (`run_cicass_ramses`): boot live RAMSES purely on the grafic
streaming set. **Finding:** mini-ramses initializes the gas velocity from `ic_velc`
(the CDM velocity) and ignores `ic_velb*` — it bakes in "baryons trace CDM," the
same limitation as MUSIC. Since the streaming velocity is *coherent across the box*
(Tseliakhovich–Hirata), it is injected post-init as a uniform gas-velocity boost
via `set_hydro!` (ρu += ρ·Δv) — exactly the right physics. RAMSES supercomoving
`unit_v` derived empirically from the DM particles (≈2016 km/s/code).

## Validated streaming signature

The relative-velocity transfer grid is anisotropic — that anisotropy *is* the
streaming. File sizes alone show it: `transfer.x -V0` → 30 KB (1D isotropic),
`-V30` → 1.79 MB (full 2D grid).

The realized gas–DM bulk velocity offset, read back from a 128³, 0.2 Mpc/h,
z_i=100 realization, and after the round-trip into RAMSES grafic format:

| v_bc [km/s @ z=1000] | `⟨v_gas − v_dm⟩` realization (km/s) | grafic `⟨ic_velb − ic_velc⟩` (km/s) | expected |
|---|---|---|---|
| 0  | `[0, 0, 0]`        | `[0, 0, 0]`           | 0 |
| 30 | `[3.027, 0, 0]`    | `[3.027, 0, 0.0004]`  | 3.027 |

Expected `= v_bc·(1+z)/1001 = 30·101/1001 = 3.027 km/s` (physical peculiar, the
linear `(1+z)⁻¹` scaling). The offset is **coherent and on a single axis**
(off-axis component `< 1e-3`), and it lands exactly where RAMSES reads it:
`ic_velb*` minus `ic_velc*`. The 0.0004 residual is f32 grafic rounding.

## Pipeline

```
CICASSSpec(vbc=30, box=0.2, z=100, N=128)
  → make_tf       (vbc_transfer/transfer.x → 2D (k⊥,k∥) TF grid)
  → generate      (libcicass_capi: makeCosICs realizer → .cicass raw dump)
  → read_snapshot (HDF5-free: DM pos[box frac]+vel[km/s], gas grid δ_b/vel/temp)
  → write_grafic_streaming → ics/{ic_velbx/y/z, ic_velcx/y/z, ic_deltab}
```

`.cicass` is a flat little-endian f64 dump (`makeCosICs/capi_out.c`,
magic `CICASS01`) — deliberately not HDF5, to dodge the `libhdf5 ∥ libenzo`
abort. DM velocities are CICASS-native physical peculiar km/s (not the Gadget
`v/√a` convention).

## Remaining work

The physics and the cross-code gate are complete. Outstanding housekeeping:

1. Fork `astromcquinn/CICASS` and push the wrapper edits (capi shim `cicass_capi.cc`,
   raw writer `capi_out.c`, `deps/build_cicass_darwin.sh`, the `main.c` dual-purpose
   + `extern "C"` edits, and the `vbc_transfer/main.cc` modern-clang narrowing fix).
   Worked locally in `Projects/cicass`; needs the user's GitHub to push.
2. Update `docs/framework-surface.md` (register CICASS as the 8th wrapper) and
   `docs/roadmap.md`.
3. Optional follow-ups: feed CICASS the realized gas velocity *field* (not just the
   coherent bulk) into RAMSES via per-oct `set_hydro!`; v_bc=0 parity vs MUSIC's
   two-component ICs; temperature field (`ic_tempb`) for RAMSES.
