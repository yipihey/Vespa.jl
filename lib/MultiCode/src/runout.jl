# runout.jl — where a CICASS run writes its outputs.
#
# Single source of truth for the output root, so the physics drivers never write
# inside the git repo.  Resolution priority:
#
#   1. ENV["VESPA_RUN_DIR"]  — an explicit run dir (set by the `vrun` manager, or
#                              pinned by hand).  Used verbatim.
#   2. ENV["VESPA_SCRATCH"]  — a scratch BASE (default the NVMe zpool); a unique
#      (default below)         run-id subdir is created under it.
#
# The repo is never an output target.  The auto run-id is
# `<UTCstamp>-<code>-<tag>-<sha7>` (e.g. 20260627T142233-arepo-dbg16-a1b9f3c) so
# concurrent runs — even with an empty CIC_TAG — never collide.

using Dates

const _VESPA_SCRATCH_DEFAULT = "/zpool/nvme/data/tabel_scratch/vespa"

"""
    run_dir(code) -> String

Absolute directory this run writes into (created if needed).  `code` is the
simulation code name (`"enzo"`, `"ramses"`, `"arepo"`) — used only to label the
auto run-id.  See the file header for the resolution order.
"""
function run_dir(code::AbstractString)
    if haskey(ENV, "VESPA_RUN_DIR")
        d = ENV["VESPA_RUN_DIR"]
        mkpath(d)
        return d
    end
    base = get(ENV, "VESPA_SCRATCH", _VESPA_SCRATCH_DEFAULT)
    tag  = let t = get(ENV, "CIC_TAG", ""); isempty(t) ? "untagged" : lstrip(t, '_') end
    sha  = try
        readchomp(`git -C $(pkgdir(MultiCode)) rev-parse --short HEAD`)
    catch
        "nogit"
    end
    stamp = Dates.format(now(UTC), "yyyymmdd\\THHMMSS")
    d = joinpath(base, "$(stamp)-$(code)-$(tag)-$(sha)")
    mkpath(d)
    return d
end
