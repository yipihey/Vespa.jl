"""
    BlockAMR

Lean block-based adaptive mesh refinement for the Vespa patch-grid stack —
fully KernelAbstractions-driven (one source, CPU + CUDA), built for extreme
dynamic range (60+ levels) gravitational collapse at GPU speed.

Design contract:

  * **Exact integer topology.** A block is B³ active cells (+2 ghosts/side) whose
    position is defined ONLY by an integer offset inside its parent block; global
    origins are `UInt128` triples in level-`l` cell units (exact far beyond 60
    levels), `dx_l = box/(nbase·2^l)` is always derived. No float grid coordinates
    exist anywhere in the topology; kernels see only `Int32` tables built at regrid.

  * **One global λ = dt/dx.** Strict 2:1 binary subcycling (the finest level's CFL
    sets dt; every coarser level takes dt×2 per level) makes `dt_l/dx_l` the SAME
    fraction on every level: fluxes are computed length-unit-free and the
    conservative update multiplies by the single f32 scalar λ.

  * **Per-level fused launches.** All blocks of a level advance in ONE kernel
    launch (workgroup = tile, group id decodes (slot, tile)); ghost fill,
    prolongation, restriction and reflux are batched rectangle-job kernels driven
    by precomputed `Int32` tables. Throughput comes from batching, not per-block
    dispatch.

  * **f16 storage, f32 compute, per-block power-of-two scales.** Physical
    variables are stored `Float16` scaled per block per class (`Dsc`, `Ssc`,
    `Esc`), `stored = phys / 2^e` with max|stored| windowed into [1,2); power-of-two
    scales make every rescale bit-exact. Species are UInt16 log₂ mass fractions
    (`ChemistryKernels` codec, absolute range — no scales needed).

  * **Backend by name.** `backend(:cpu)` always works; `backend(:cuda)` after
    `using CUDA` (extension). Allocation via `device_zeros`/`to_device`/`to_host`.
"""
module BlockAMR

using KernelAbstractions
const KA = KernelAbstractions
using Printf: @printf
import ChemistryKernels

export backend, has_backend, device_zeros, to_device, to_host
export BlockMeta, Level, AMRHierarchy
export add_block!, remove_block!, init_base_level!, overlapping_blocks, check_nesting
export build_sibling_jobs, build_prolong_jobs, build_restrict_jobs, RectJobTable
export fill_ghosts!, restrict_level!, blockview
export stage_level!, max_signal, hierarchy_rk2_step!, compute_dt, total_conserved
export ctu_level!
export build_level_tables!, build_cf_register!, capture_cf!, reflux_apply!
export BlockRefinementPolicy, regrid!, compact_level!
export advance_hierarchy!, advance_level_w!, compute_lambda!, capture_fine!, capture_coarse!
export update_scales!, encode_from_host!
export grav_kick_level!, sync_block_geometry!
export chem_level!
export solve_gravity_level!, phi_from_global!, grav_kick_level_pool!
export deposit_particles_level!, gather_accel_particles!, particles_kick!, particles_drift!, build_block_lookup!
export global_from_level0!, compton_drag!
export save_checkpoint, load_checkpoint
export memory_mode, set_memory_mode!, memory_report
export prefetch_level!, advise_level_host!, advise_all_host!

# ── backend registry (house pattern — PoissonKernels/ChemistryKernels) ────────
const _BACKENDS = Dict{Symbol,Any}(:cpu => CPU())

# Backends on which the tiled single-pass CTU kernel (`_ctu_step_k!`, ~42 KB
# threadgroup memory at NS=2) is the validated default.  A GPU extension adds
# its symbol here ONLY after confirming the launch fits the device's shared-
# memory limit (CUDA: yes; Metal: pending threadgroup-size validation on the
# target Apple GPU — until then the per-cell CTU oracle runs, correct but
# slower, and `ctu_level!(...; tiled=true)` force-enables tiling for testing).
const _TILING = Set{Symbol}()

"Register a KernelAbstractions backend under `name` (used by the GPU extensions)."
register_backend!(name::Symbol, be) = (_BACKENDS[name] = be)

"Mark backend `name` as validated for the tiled CTU kernel (see `_TILING`)."
enable_tiling!(name::Symbol) = (push!(_TILING, name); name)

"True when backend `name` is available (`:cuda` needs `using CUDA` first)."
has_backend(name::Symbol) = haskey(_BACKENDS, name)

"""
    backend(name::Symbol = :cpu)

The KernelAbstractions backend registered under `name`. `:cpu` is always
available; `:cuda` requires `using CUDA` to have loaded `BlockAMRCUDAExt`.
"""
function backend(name::Symbol = :cpu)
    return get(_BACKENDS, name) do
        error("BlockAMR backend :$name is not available. " *
              (name === :cuda ? "Run `using CUDA` first (NVIDIA only)." :
               "Known backends: $(collect(keys(_BACKENDS)))."))
    end
end

# ── memory backing mode (out-of-core Phase 0) ─────────────────────────────────
# `:device`  — pools in ordinary GPU memory (default; the fully-resident path).
# `:managed` — pools in CUDA Unified memory: the driver pages block data over
#              PCIe on demand, so the host's RAM backs a working set far larger
#              than the GPU. Set before building a hierarchy; the CUDA extension
#              reads it in `device_zeros`. No effect on the CPU backend.
const _MEMMODE = Ref(:device)
memory_mode() = _MEMMODE[]
set_memory_mode!(m::Symbol) =
    (m in (:device, :managed) || error("memory mode must be :device or :managed");
     _MEMMODE[] = m)

# Migration hints for unified pools — no-ops unless a GPU extension overrides
# them (CPU arrays and plain device arrays ignore them).  `advise_host!` marks a
# cold pool as host-preferred + device-accessible (paged in on touch);
# `prefetch_device!` pulls a hot pool to the GPU up front.
advise_host!(a) = a
prefetch_device!(a) = a

"A zero-filled array of element type `T` and shape `dims` on backend `be`."
device_zeros(::CPU, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

"Copy host array `a` onto backend `be`, converting to element type `T`."
function to_device(be, a::AbstractArray, ::Type{T} = eltype(a)) where {T}
    d = device_zeros(be, T, size(a))
    copyto!(d, convert(Array{T}, a))
    return d
end

"`to_host(a)` — a plain host `Array` copy of a device array; synchronizes first."
function to_host(a::AbstractArray)
    KA.synchronize(KA.get_backend(a))
    return Array(a)
end

include("topology.jl")     # exact-integer origins, wrapped intervals, BlockMeta
include("pool.jl")         # Level block pool, slot alloc/free/grow, AMRHierarchy
include("tables.jl")       # RectJobTable + sibling/prolong/restrict builders
include("transfer.jl")     # batched copy/prolong/restrict kernels + fill_ghosts!
include("kernels.jl")      # fused batched SSP-RK2 PLM+HLLC dual-energy hydro
include("kernels_ctu.jl")  # tiled single-pass CTU (GPU) + per-cell oracle
include("reflux.jl")       # C/F face registry + recompute capture/apply
include("stepping.jl")     # hierarchy integrator + conservation diagnostics
include("scales.jl")       # per-block power-of-two f32 scale maintenance (f16)
include("regrid.jl")       # dynamic refinement: flag → cluster → rebuild
include("gravity.jl")      # KDK half-kicks from the topgrid potential
include("chem.jl")         # fast analytic H+H2 chemistry over block batches
include("particles.jl")    # DM particles: per-level deposit + finest-level accel gather
include("checkpoint.jl")   # root-boundary save/restore (topology+fields+scales+phi)

end # module BlockAMR
