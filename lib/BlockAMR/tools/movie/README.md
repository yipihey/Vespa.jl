# BlockAMR movies

Movies are a first-class output of BlockAMR (`src/movie.jl`). Derived fields are
rasterized on the device onto a uniform image at a chosen AMR level, stored
losslessly as 16-bit codes in RGB PNGs, and viewed in a browser player
(`player.html`, WebGL2) where min/max, lin/log/symlog, colormap, playback speed
and panels are all adjustable after the fact.

## Writing

```julia
mw = MovieWriter("RUN/movie", hier; level = 4, cooling = cool,           # T, t_cool from the table
                 length_unit = ("pc", 100.0),
                 time_labels = [("t", "t_cc", 1 / TCC), ("t", "Myr", TU / MYR)],
                 units = Dict(:p => ("K cm⁻³", fac_p), :vx => ("km/s", 100.0)))
write_frame!(mw, hier, t)            # whenever a frame is due (fills ghosts itself)
```

Fields (`MOVIE_CATALOG`): `rho p T B2 absJ Jz divv absw wz vmag vx mach beta entropy
tcool level`. The default list includes the magnetic fields only for MHD and
`tcool` only with a cooling table. `examples/hvc_cloud_2d.jl` writes a movie
with `HVC_MOVIE=<frames per t_cc>` (plus `HVC_MOVIE_LEVEL` and `HVC_MOVIE_FIELDS`).

On disk, each frame is one `RUN/movie/<field>/NNNNN.png` per field. A code in
1–65535 maps linearly onto that frame's `[lo, hi]` of log₁₀(value) for positive
fields, or of the value itself for signed ones (`J_z`, `∇·v`, `ω_z`, `v_x`).
Code 0 means no data. `manifest.json` holds the per-frame ranges, 0.1/99.9
percentiles, times and units. It is rewritten after every frame.

Gradients are 2nd-order centred differences on ghost-filled blocks. Where a
stencil reads prolonged coarse→fine ghosts, the coarse truncation error divided
by the fine Δx shows as a faint seam on the fine side of the C/F face. The
solver sees the same ghosts.

## Watching

```sh
python3 -m http.server 8000 -d RUN/movie     # then open http://localhost:8000/
```

`index.html` is installed into every movie directory. **follow run** re-reads
the manifest every 5 s, so a run can be watched while it is still writing.
`#f40` opens at frame 40. Locally,
`?panels=4&fields=rho,T,B2,absw&autoplay` sets the panels and starts playback.
Press `?` in the player for its keyboard shortcuts.

## Sharing

```julia
movie_bundle("RUN/movie", "RUN/share"; maxfiles = 250, every = 1, downsample = 1,
             fields = nothing, title = "…")
```

This tiles several frames into each PNG sheet, so the whole bundle has at most
`maxfiles` images. A claude.ai Artifact holds 511 files per version, 255 per
publish and ≤ 15 MB per file. You can also subsample frames (`every`), average
pixel blocks (`downsample`, exact in the stored quantity), or keep a subset of
`fields`. Publish `RUN/share/player.html` as the page, with `manifest.json` and
the `<field>/*.png` files beside it. Any static web host works too: copy
`RUN/share` and open `index.html`.
