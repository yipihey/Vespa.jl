# Instrumentation wrapper (ADR-0001, P10).
#
# `Instrumented{B}` satisfies the same interface as the backend it wraps and
# times the coarse, meaningful units of work — `for_each_cell`, restrict/prolong,
# refine/coarsen. It is a *distinct type specialization*, not a runtime flag, so
# when a simulation runs on a bare backend there is zero overhead: the timing
# code is simply never compiled in.
#
# Because the same wrapper wraps every backend with the same span names and
# units, measurements are directly comparable across RefMesh / HGBackend / a
# future Rust or GPU backend. Measurement never reshapes the interface: only the
# operations already on the seam are timed. Per-cell/-face queries (`neighbor`,
# geometry, field views) are hot and called from inside kernels, so they pass
# through untimed — their cost shows up inside the `for_each_cell` span that
# drives them, which is the meaningful unit.

"Running accumulator for one timed operation."
mutable struct SpanStat
    count::Int
    total_ns::Int
end
SpanStat() = SpanStat(0, 0)

"""
    Instrumented(backend)

Wrap `backend` so that interface calls accumulate per-operation timing. Read the
report with [`span_report`](@ref); clear it with [`reset_spans!`](@ref).
"""
struct Instrumented{B<:AbstractMeshBackend} <: AbstractMeshBackend
    inner::B
    spans::Dict{Symbol,SpanStat}
end
Instrumented(b::AbstractMeshBackend) = Instrumented(b, Dict{Symbol,SpanStat}())

@inline function _record!(w::Instrumented, name::Symbol, t0::UInt)
    dt = Int(time_ns() - t0)
    s = get!(w.spans, name, SpanStat())
    s.count += 1
    s.total_ns += dt
    return nothing
end

"Per-operation cost report: `(name, calls, total_ms, mean_µs)` sorted by total time."
function span_report(w::Instrumented)
    rows = Tuple{Symbol,Int,Float64,Float64}[]
    for (name, s) in w.spans
        total_ms = s.total_ns / 1e6
        mean_us = s.count == 0 ? 0.0 : s.total_ns / s.count / 1e3
        push!(rows, (name, s.count, total_ms, mean_us))
    end
    sort!(rows; by = r -> r[3], rev = true)
    return rows
end

"Clear accumulated spans."
reset_spans!(w::Instrumented) = (empty!(w.spans); w)

# -- timed operations: wrap, record, delegate --

for_each_cell(f, w::Instrumented; kw...) =
    (t0 = time_ns(); r = for_each_cell(f, w.inner; kw...); _record!(w, :for_each_cell, t0); r)

for_each_face(f, w::Instrumented; kw...) =
    (t0 = time_ns(); r = for_each_face(f, w.inner; kw...); _record!(w, :for_each_face, t0); r)

restrict!(w::Instrumented, args...) =
    (t0 = time_ns(); r = restrict!(w.inner, args...); _record!(w, :restrict!, t0); r)

prolong!(w::Instrumented, args...) =
    (t0 = time_ns(); r = prolong!(w.inner, args...); _record!(w, :prolong!, t0); r)

refine!(w::Instrumented, cells) =
    (t0 = time_ns(); r = refine!(w.inner, cells); _record!(w, :refine!, t0); r)

coarsen!(w::Instrumented, parents) =
    (t0 = time_ns(); r = coarsen!(w.inner, parents); _record!(w, :coarsen!, t0); r)

# -- untimed pass-throughs (cheap topology + hot per-cell/-face queries) --

rank(w::Instrumented) = rank(w.inner)
domain(w::Instrumented) = domain(w.inner)
n_cells(w::Instrumented) = n_cells(w.inner)
level_of(w::Instrumented, c) = level_of(w.inner, c)
max_level(w::Instrumented) = max_level(w.inner)
cell_center(w::Instrumented, c) = cell_center(w.inner, c)
cell_width(w::Instrumented, c) = cell_width(w.inner, c)
cell_volume(w::Instrumented, c) = cell_volume(w.inner, c)
face_area(w::Instrumented, c, axis) = face_area(w.inner, c, axis)
neighbor(w::Instrumented, c, axis, side; kw...) = neighbor(w.inner, c, axis, side; kw...)
allocate_fields(w::Instrumented, spec; kw...) = allocate_fields(w.inner, spec; kw...)
field_eltype(w::Instrumented) = field_eltype(w.inner)
coord_eltype(w::Instrumented) = coord_eltype(w.inner)
field_view(w::Instrumented, store, name) = field_view(w.inner, store, name)
