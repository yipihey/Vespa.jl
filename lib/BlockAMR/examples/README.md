# BlockAMR examples: a high-velocity cloud in a wind

`hvc_cloud_2d.jl` runs a high-velocity cloud in a Mach-2 hot halo wind: 2-D MHD with
constrained transport and dual energy, tabulated Grackle cooling, in an elongated wind
tunnel, on a CPU or an NVIDIA GPU. Every setting is an environment variable.

The problem setup was reconstructed from the first-Grackle-runs plan (04.05, 2026-10-28);
its companion initial-condition documents were not available. The header of
`hvc_cloud_2d.jl` lists every assumption.

## Setup

You need Julia 1.10 or later (tested on 1.12), and Python 3 with `numpy` and `h5py` for the
cooling table. `ffmpeg` with libvpx-vp9 is only needed to make shareable movies.

Clone the code and install the packages. `ChemistryKernels` and `EmissionKernels` come from our
own Julia registry, so add that first:

```sh
git clone -b blockamr-mhd-dualenergy git@github.com:yipihey/Vespa.jl.git
cd Vespa.jl
julia -e 'using Pkg; Pkg.Registry.add("General"); Pkg.Registry.add(RegistrySpec(url = "https://github.com/yipihey/VespaRegistry"))'
julia --project=lib/BlockAMR/examples -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Make the cooling table (FG2011 UV background, z = 0), which the driver looks for in
`~/data/grackle_tables/FG2011_z0` unless `HVC_TABLE` says otherwise:

```sh
mkdir -p ~/data/grackle_tables && cd ~/data/grackle_tables
curl -LO 'https://raw.githubusercontent.com/grackle-project/grackle_data_files/main/input/CloudyData_UVB=FG2011.h5'
cd -
python3 lib/BlockAMR/tools/cloudy_table_export.py \
    ~/data/grackle_tables/CloudyData_UVB=FG2011.h5 ~/data/grackle_tables/FG2011_z0
```

## Running

Run these from the top of the `Vespa.jl` checkout. Start with a short test.

**CPU** (use a thread count that suits your machine):

```sh
HVC_BACKEND=cpu HVC_OUT=runs/test-cpu HVC_NX=256 HVC_LMAX=2 HVC_BETA=100 HVC_SMALLR=1e-2 HVC_TEND=0.1 \
  julia -t 16 --project=lib/BlockAMR/examples lib/BlockAMR/examples/hvc_cloud_2d.jl
```

**GPU** (NVIDIA, through CUDA; this is the default backend):

```sh
HVC_OUT=runs/test-gpu HVC_NX=256 HVC_LMAX=4 HVC_BETA=100 HVC_SMALLR=1e-2 HVC_TEND=0.1 \
  julia --project=lib/BlockAMR/examples lib/BlockAMR/examples/hvc_cloud_2d.jl
```

A run ends with a `done:` line and leaves `timeseries.csv`, `params.toml`, `snap/` and
`final.ckpt`. On our machine these tests take about 1 to 1.5 minutes, mostly compilation.
Installing the packages takes about 2.5 minutes the first time.

### The runs we have done

The AMR run, with a movie, reaching 4 t_cc:

```sh
HVC_OUT=runs/amr-b100 HVC_NX=256 HVC_LMAX=4 HVC_BETA=100 HVC_SMALLR=1e-2 HVC_TEND=4 HVC_MOVIE=20 \
  julia --project=lib/BlockAMR/examples lib/BlockAMR/examples/hvc_cloud_2d.jl
```

The 8192×2048 uniform grid, four 2048² blocks, 6 t_cc. It took 14.9 hours and about 24 GB
on one RTX A6000:

```sh
HVC_OUT=runs/uni-b100 HVC_NX=8192 HVC_B=2048 HVC_LMAX=0 HVC_BETA=100 HVC_SMALLR=1e-2 \
HVC_TEND=6 HVC_MOVIE=20 HVC_MOVIE_LEVEL=0 HVC_GRAPH=0 HVC_REGRID_EVERY=1000000 HVC_WALL=36 \
  julia --project=lib/BlockAMR/examples lib/BlockAMR/examples/hvc_cloud_2d.jl
```

For the hydro twin, set `HVC_BETA=Inf`, which took 2.4 hours. It uses the same MHD solver
with zero field (HLLD reduces to HLLC at B = 0).

### Settings you are most likely to change

| Variable | Default | Meaning |
|---|---|---|
| `HVC_BACKEND` | `cuda` | `cpu` or `cuda` |
| `HVC_OUT` | a temp directory | output directory |
| `HVC_NX` | 256 | base cells along x; the number of y cells follows from the box shape |
| `HVC_LMAX` | 4 | refinement levels; 0 is a uniform grid |
| `HVC_B` | 16 | block size; must divide both grid sides |
| `HVC_BETA` | `Inf` | plasma β of the transverse field; `Inf` is B = 0 |
| `HVC_LX`, `HVC_LY` | 32, 8 | box size in cloud radii |
| `HVC_TEND` | 6 | end time in cloud-crushing times t_cc |
| `HVC_SMALLR` | 1e-6 | density floor in halo densities; magnetized runs need about 1e-2 |
| `HVC_T` | `Float32` | precision; use `Float64` only for a specific reason |
| `HVC_MOVIE` | 0 | movie frames per t_cc |
| `HVC_WALL` | none | stop and save after this many hours |

The header of `hvc_cloud_2d.jl` lists the rest.

**Choosing a block size.** On a uniform grid, use the largest block that divides both sides:
results are bitwise identical at any block size, and large blocks are about twice as fast at
about 40% less memory than B = 16. A 4:1 box with B = NY is four blocks, and sides need not be
powers of two. With AMR, B = 16 to 64 gave the same physics to about 0.2% at 1 t_cc, with
larger blocks refining more volume.

**Memory, uniform grid.** Budget about 0.35 GB per million cells with large blocks, or about
0.5 GB per million with B = 16.

## Movies

`HVC_MOVIE` writes a browser movie to `HVC_OUT/movie`. Watch it with

```sh
python3 -m http.server 8000 -d runs/uni-b100/movie     # then open http://localhost:8000/
```

To make a small copy for sharing (about 300× smaller, but lossy):

```julia
using BlockAMR
movie_video("runs/uni-b100/movie", "runs/uni-b100/share")
```

Details are in `../tools/movie/README.md`.

## Cooling and chemistry

**What this problem uses.** The driver uses tabulated Grackle (Cloudy) cooling and UV heating,
in ionization equilibrium. You can change the metallicity (`HVC_Z`, in solar units) and the
table. For the self-shielded table, run the export on `CloudyData_UVB=FG2011_shielded.h5` from
the same URL and point `HVC_TABLE` at the result.

**Non-equilibrium chemistry does not work with this driver yet. Do not pass `chem =` to a
2-D or MHD run.** The H + H₂ network from `ChemistryKernels` is connected to BlockAMR, but only
for 3-D hydro runs:

- `chem_level!` assumes 3-D blocks, so on a 2-D level it would address the wrong cells, and it
  would not raise an error.
- The MHD and constrained-transport kernels do not advect species.
- The network has no tabulated metal cooling above 10⁴ K, which this problem depends on in the
  1e6 K halo and the mixing layers.
- It is far slower: about 1 Mcell/s in Float32 on our GPU, against hundreds of Mcell/s for the
  table.

**What you can run today:**

- One-zone chemistry on its own, which is a good way to compare the network with the table on
  HVC conditions: see the quick start in the
  [ChemistryKernels README](https://github.com/yipihey/ChemistryKernels.jl) and its
  `docs/integration.md`.
- Chemistry in a 3-D hydro hierarchy: create it with `nsp = 2` (HII, H₂) and pass
  `chem = (; a_value, density_units, length_units, time_units)` to `advance_hierarchy!`.
  `test/test_chem.jl` is a working example of `chem_level!`. The `advance_hierarchy_graph!` path
  does not run chemistry.

**What this problem would need:** species advection in `ct.jl` and `mhd.jl`, including the
coarse/fine flux corrections; a 2-D `chem_level!`; and a way to combine the network with
tabulated metal cooling without counting primordial cooling twice.

## Known issues

- **A standing shock in the default box.** With `HVC_LY=8` the cloud blocks 25% of a periodic
  channel, and every run so far has a full-width normal shock upstream of the cloud (near
  x = 2.5 R at β = 100, x = 3.3 R at B = 0; the density jump matches Mach 2). The likely cause
  is choking. The cloud is then probably in subsonic, compressed gas rather than the intended
  Mach-2 wind, so survival results from this box may not apply to an isolated cloud. Wider
  boxes (`HVC_LY=16`, `32`) have not been tested yet.
- **β = 100 runs slow down after about 1 t_cc.** The time step collapses in low-density
  draped-field cells: the uniform-grid run went from about 22,000 to about 500,000 steps per
  t_cc. B = 0 runs do not do this.
- **∇·B grows in Float32.** It reached about 1.7e-4 (relative to the peak field) after 2.4M
  steps, and is about 1e-14 in Float64. It grows like rounding error would.
- **No restart.** The driver only saves a checkpoint at the end, so a crash loses the run apart
  from the snapshots and movie frames already written.
- **Tested on** Linux with an NVIDIA RTX A6000 and on CPU. Apple GPUs have a backend in
  BlockAMR, but this driver does not load it.
