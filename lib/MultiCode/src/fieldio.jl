# fieldio.jl — one writer for the CICASS field dumps + a self-describing side-car.
#
# The legacy dumps are heterogeneous: most lead with an Int64 header (grid size N, or
# a particle/cell count) followed by column-major Float64 columns; `*_fields*.bin`
# has no header.  `write_grid` reproduces those byte streams EXACTLY (it just
# `write`s `vec(col)` for each column passed, in order, behind an optional Int64
# header) and, as a side effect, appends a row to `<run-dir>/fields.toml` describing
# the file (kind, columns, dtype, order, header, ndim, n).  That side-car lets the
# Python `RunDir` (lib/MultiCode/examples/vespa_io.py) load any dump without
# re-encoding the format — a single source of truth instead of ~20 copies.
#
# Byte-identity is by construction: pass the SAME expressions the old `open(...) do
# io; write(...) end` block wrote, and the bytes are unchanged.  The recorded dtype
# is the column's true `eltype`, so the reader is correct even for non-Float64 dumps.

"""
    write_grid(path; kind, columns, n, ndim=3, header=true)

Write the field dump `columns` (a vector of `name => array` pairs) to `path` and
append a descriptor to `fields.toml` in the same directory.

  * `header=true` writes `Int64(n)` before the columns (the legacy convention); for
    grid dumps `n` is the per-axis size, for particle/phase dumps it is the count.
  * `header=false` writes no header (the `*_fields*.bin` case); `n` is still recorded
    so the reader can reshape.
  * `ndim` is 3 for full grids, 2 for slices, 1 for flat particle/phase columns.

Columns are written as `vec(array)` in the order given — pass the exact expressions
the legacy writer used (including any `Float64.(…)` conversions) for byte-identity.
"""
# `columns` is any iterable of `name => array` pairs.  Pass a TUPLE when columns have
# mixed element types (e.g. the RAMSES pdump's Int64 idp + Float64 coords) — a Vector
# literal would promote them to a common eltype and silently re-encode the bytes.
function write_grid(path::AbstractString; kind::AbstractString,
                    columns, n::Integer, ndim::Integer=3,
                    header::Bool=true)
    open(path, "w") do io
        header && write(io, Int64(n))
        for (_, a) in columns
            write(io, vec(a))
        end
    end
    _record_fields(path, kind, columns, Int(n), Int(ndim), header)
    return path
end

function _record_fields(path, kind, columns, n, ndim, header)
    cols   = join(("\"$(first(c))\"" for c in columns), ", ")
    # per-column element type (e.g. "Float64", or "Int64" for the RAMSES idp column)
    dtypes = join(("\"$(string(eltype(last(c))))\"" for c in columns), ", ")
    entry = """
    [[field]]
    file = "$(basename(path))"
    kind = "$(kind)"
    columns = [$(cols)]
    dtypes = [$(dtypes)]
    order = "F"
    has_header = $(header)
    ndim = $(ndim)
    n = $(n)
    """
    open(joinpath(dirname(path), "fields.toml"), "a") do io
        write(io, entry)
    end
end
