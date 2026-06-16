using Printf
using PowerFoam

hier_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const HIER_OUTBASE = joinpath(@__DIR__, "out", "arepo_hierarchy_gate_3d")
const HIER_N = hier_arg(1, 12, Int)
const HIER_DT = hier_arg(2, 0.001, Float64)
const HIER_RIEMANN = Symbol(length(ARGS) >= 3 ? ARGS[3] : "hll")
const HIER_NSTEPS = hier_arg(4, 1, Int)
const HIER_GAMMA = 5 / 3
const HIER_FIXTURE = Symbol(lowercase(get(ENV, "POWERFOAM_HIERARCHY_FIXTURE", "decay")))
const HIER_FIXTURE_TAG = HIER_FIXTURE == :decay ? "" : "_$(HIER_FIXTURE)"
const HIER_RUN_TAG = HIER_NSTEPS == 1 ?
                     string(@sprintf("N%d_dt%.3g_%s", HIER_N, HIER_DT,
                                     HIER_RIEMANN), HIER_FIXTURE_TAG) :
                     string(@sprintf("N%d_dt%.3g_n%d_%s", HIER_N, HIER_DT,
                                     HIER_NSTEPS, HIER_RIEMANN),
                            HIER_FIXTURE_TAG)
const HIER_OUTDIR = joinpath(HIER_OUTBASE, replace(HIER_RUN_TAG, "." => "p"))
const HIER_AREPOLIB_IMPORT_ERROR = Ref{Any}(nothing)

try
    @eval import ArepoLib
catch err
    HIER_AREPOLIB_IMPORT_ERROR[] = err
end

function _hier_bridge_available()
    return isdefined(Main, :ArepoLib) && isdefined(ArepoLib, :get_hydro_timebins)
end

function _hier_arepo_libpath()
    if isdefined(Main, :ArepoLib)
        return ArepoLib.libpath()
    end
    return "unavailable: ArepoLib package is not in the active Julia environment"
end

function _apply_multirung_fixture!(dir)
    HIER_FIXTURE == :decay && return
    HIER_FIXTURE == :multirung ||
        error("unsupported POWERFOAM_HIERARCHY_FIXTURE=$(HIER_FIXTURE); use decay or multirung")
    py = Base.invokelatest(getfield(Main, :python_cmd))
    code = """
import h5py
import numpy as np
import sys
path = sys.argv[1]
with h5py.File(path, "r+") as f:
    u = f["PartType0/InternalEnergy"]
    arr = u[...]
    idx = np.arange(arr.shape[0])
    factors = np.ones_like(arr)
    factors[idx % 7 == 0] = 64.0
    factors[idx % 7 == 1] = 16.0
    factors[idx % 7 == 2] = 4.0
    factors[idx % 7 == 3] = 0.25
    u[...] = arr * factors
"""
    run(pipeline(Cmd([py, "-c", code, joinpath(dir, "IC.hdf5")]);
                 stdout = devnull))
end

function _hier_stats(exported, arepo_bins)
    pf_bins = arepo_hydro_timebins_3d(exported.vol, exported.pressure, exported.rho;
                                      gamma = HIER_GAMMA, courant = 0.3,
                                      max_dt = 0.05, min_dt = 1e-6,
                                      timebase_interval = arepo_bins.timebase_interval,
                                      velocity = exported.vel,
                                      mesh_velocity = exported.velvertex)
    gravity_bins = hasproperty(arepo_bins, :gravity_bins) ?
                   Int.(arepo_bins.gravity_bins) :
                   fill(typemax(Int), length(pf_bins.bins))
    effective_bins = min.(pf_bins.bins, gravity_bins)
    active_from_arepo_bins = arepo_active_cells_3d(arepo_bins.bins,
                                                  arepo_bins.ti_current)
    active_from_pf_bins = arepo_active_cells_3d(effective_bins,
                                                arepo_bins.ti_current)
    active_list = hasproperty(arepo_bins, :active_list) ?
                  sort(Int.(arepo_bins.active_list)) :
                  findall(arepo_bins.active)
    active_from_mask = findall(arepo_bins.active)
    active_list_delta = length(setdiff(active_list, active_from_mask)) +
                        length(setdiff(active_from_mask, active_list))
    next_arepo_bins = arepo_next_sync_step_3d(arepo_bins.bins,
                                             arepo_bins.ti_current)
    next_pf_bins = arepo_next_sync_step_3d(effective_bins,
                                          arepo_bins.ti_current)
    return (;
        cells = exported.ng,
        ti_current = arepo_bins.ti_current,
        timebase_interval = arepo_bins.timebase_interval,
        occupied_bins = length(unique(arepo_bins.bins)),
        active_arepo = count(arepo_bins.active),
        active_list_count = length(active_list),
        active_from_arepo_bins = count(active_from_arepo_bins),
        active_from_pf_bins = count(active_from_pf_bins),
        active_list_mismatches = active_list_delta,
        raw_hydro_bin_mismatches = count(pf_bins.bins .!= arepo_bins.bins),
        bin_mismatches = count(effective_bins .!= arepo_bins.bins),
        active_mask_mismatches = count(active_from_pf_bins .!= arepo_bins.active),
        arepo_active_self_mismatches = count(active_from_arepo_bins .!= arepo_bins.active),
        pf_raw_min_bin = minimum(pf_bins.bins),
        pf_raw_max_bin = maximum(pf_bins.bins),
        gravity_min_bin = minimum(gravity_bins),
        gravity_max_bin = maximum(gravity_bins),
        pf_min_bin = minimum(effective_bins),
        pf_max_bin = maximum(effective_bins),
        arepo_min_bin = minimum(arepo_bins.bins),
        arepo_max_bin = maximum(arepo_bins.bins),
        next_arepo_bins,
        next_pf_bins,
        next_step_mismatch = next_arepo_bins != next_pf_bins)
end

function _write_hier_table(io, title, stats)
    println(io, "## $title")
    println(io)
    println(io, "| check | value |")
    println(io, "| --- | ---: |")
    @printf(io, "| cells | %d |\n", stats.cells)
    @printf(io, "| ti_current | %d |\n", stats.ti_current)
    @printf(io, "| timebase interval | %.12g |\n", stats.timebase_interval)
    @printf(io, "| occupied hydro bins | %d |\n", stats.occupied_bins)
    @printf(io, "| active cells from AREPO export | %d |\n", stats.active_arepo)
    @printf(io, "| active cells from AREPO active list | %d |\n",
            stats.active_list_count)
    @printf(io, "| active cells from AREPO bins | %d |\n",
            stats.active_from_arepo_bins)
    @printf(io, "| active cells from PowerFoam bins | %d |\n",
            stats.active_from_pf_bins)
    @printf(io, "| raw hydro bin mismatches | %d |\n",
            stats.raw_hydro_bin_mismatches)
    @printf(io, "| bin mismatches | %d |\n", stats.bin_mismatches)
    @printf(io, "| active mask mismatches | %d |\n",
            stats.active_mask_mismatches)
    @printf(io, "| active list mismatches | %d |\n",
            stats.active_list_mismatches)
    @printf(io, "| AREPO active self mismatches | %d |\n",
            stats.arepo_active_self_mismatches)
    @printf(io, "| AREPO min/max bin | %d / %d |\n",
            stats.arepo_min_bin, stats.arepo_max_bin)
    @printf(io, "| raw PowerFoam hydro min/max bin | %d / %d |\n",
            stats.pf_raw_min_bin, stats.pf_raw_max_bin)
    @printf(io, "| AREPO gravity min/max bin | %d / %d |\n",
            stats.gravity_min_bin, stats.gravity_max_bin)
    @printf(io, "| effective PowerFoam min/max bin | %d / %d |\n",
            stats.pf_min_bin, stats.pf_max_bin)
    @printf(io, "| next sync step AREPO bins / PF bins | %d / %d |\n",
            stats.next_arepo_bins, stats.next_pf_bins)
    println(io)
end

function _hier_stats_passed(stats)
    return stats.bin_mismatches == 0 &&
           stats.active_mask_mismatches == 0 &&
           stats.active_list_mismatches == 0 &&
           stats.arepo_active_self_mismatches == 0 &&
           !stats.next_step_mismatch
end

function _write_hier_report(path; status, initial_stats = nothing,
                            step_stats = NamedTuple[],
                            arepo_step_statuses = String[])
    open(path, "w") do io
        println(io, "# AREPO Hierarchy Gate")
        println(io)
        println(io, "This gate compares PowerFoam's AREPO-style hydro timebin")
        println(io, "quantization and active-cell mask against AREPO's live scheduler")
        println(io, "at every checked synchronization point.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", _hier_arepo_libpath())
        @printf(io, "- N: %d^3\n", HIER_N)
        @printf(io, "- Riemann solver: %s\n", HIER_RIEMANN)
        @printf(io, "- hierarchy fixture: %s\n", HIER_FIXTURE)
        @printf(io, "- requested AREPO steps: %d\n", HIER_NSTEPS)
        @printf(io, "- status: %s\n", status)
        println(io)
        if initial_stats === nothing
            if HIER_AREPOLIB_IMPORT_ERROR[] !== nothing
                println(io, "The active Julia environment does not expose")
                println(io, "`ArepoLib`, so AREPO scheduler state could not be queried.")
            else
                println(io, "The AREPO bridge does not yet expose")
                println(io, "`get_hydro_timebins`.")
            end
            println(io)
            println(io, "See")
            println(io, "`lib/PowerFoam/external_patches/arepo_bridge_face_trace_contract.md`.")
            return
        end
        _write_hier_table(io, "Initial Synchronized State", initial_stats)
        for (i, stats) in enumerate(step_stats)
            step_status = i <= length(arepo_step_statuses) ?
                          arepo_step_statuses[i] : "unknown"
            @printf(io, "- AREPO native step %d status: %s\n\n",
                    i, string(step_status))
            _write_hier_table(io, @sprintf("Post-Step %d Scheduler State", i),
                              stats)
        end
    end
end

function _scheduler_fields_from_arepo(h)
    return (;
        ng = ArepoLib.info(h).numgas,
        vol = ArepoLib.get_cell_field(h, :volume),
        rho = ArepoLib.get_cell_field(h, :rho),
        pressure = ArepoLib.get_cell_field(h, :pressure),
        vel = ArepoLib.get_particle_field(h, :vel)[1:ArepoLib.info(h).numgas, :],
        velvertex = ArepoLib.get_cell_field(h, :velvertex))
end

function main_hierarchy()
    mkpath(HIER_OUTDIR)
    report = joinpath(HIER_OUTDIR, "README.md")
    if !_hier_bridge_available()
        _write_hier_report(report; status = "skipped: missing AREPO timebin bridge")
        @printf("wrote %s\n", report)
        @printf("skipped: ArepoLib.get_hydro_timebins is not available\n")
        return
    end
    include(joinpath(@__DIR__, "arepo_geometry_gate_3d.jl"))
    dir = Base.invokelatest(getfield(Main, :stage_arepo_case), HIER_N;
                            riemann = HIER_RIEMANN)
    _apply_multirung_fixture!(dir)
    exported = Base.invokelatest(getfield(Main, :arepo_initial_export), dir)
    try
        initial_bins = ArepoLib.get_hydro_timebins(exported.h)
        initial_stats = _hier_stats(exported, initial_bins)
        step_stats = NamedTuple[]
        step_statuses = String[]
        for _ in 1:HIER_NSTEPS
            push!(step_statuses, string(ArepoLib.run_step!(exported.h)))
            post_fields = _scheduler_fields_from_arepo(exported.h)
            post_bins = ArepoLib.get_hydro_timebins(exported.h)
            push!(step_stats, _hier_stats(post_fields, post_bins))
        end
        all_passed = all(_hier_stats_passed, step_stats)
        reached_multirung = any(s -> s.occupied_bins > 1, step_stats)
        status = all_passed ?
                 (reached_multirung || HIER_NSTEPS == 1 ?
                  "passed" : "passed: no multi-rung occupancy reached") :
                 "failed"
        _write_hier_report(report; status, initial_stats, step_stats,
                           arepo_step_statuses = step_statuses)
        @printf("wrote %s\n", report)
        last_stats = isempty(step_stats) ? initial_stats : step_stats[end]
        @printf("hierarchy %s: steps=%d occupied_bins=%d bin mismatches=%d active mismatches=%d next %d/%d\n",
                status, HIER_NSTEPS, last_stats.occupied_bins,
                last_stats.bin_mismatches, last_stats.active_mask_mismatches,
                last_stats.next_arepo_bins, last_stats.next_pf_bins)
    finally
        ArepoLib.finalize(exported.h)
    end
end

main_hierarchy()
