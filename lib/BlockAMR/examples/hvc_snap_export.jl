# hvc_snap_export.jl — convert hvc_cloud_2d.jl snapshots (Julia-serialized) into
# .npy maps + a small meta file per snapshot, for hvc_plots.py.
#   julia --project=lib/BlockAMR/test lib/BlockAMR/examples/hvc_snap_export.jl RUN_DIR...
using BlockAMR, Serialization, Printf

function write_npy(path, a::AbstractMatrix{Float32})
    hdr = "{'descr': '<f4', 'fortran_order': True, 'shape': ($(size(a, 1)), $(size(a, 2))), }"
    pad = 64 - (10 + length(hdr) + 1) % 64
    hdr = hdr * " "^pad * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93]); write(io, "NUMPY"); write(io, UInt8[1, 0])
        write(io, UInt16(length(hdr))); write(io, hdr); write(io, a)
    end
end

for run in ARGS
    sd = joinpath(run, "snap"); od = joinpath(run, "npy"); mkpath(od)
    files = sort(filter(f -> endswith(f, ".jls"), readdir(sd)))
    for f in files
        k = f[6:9]
        isfile(joinpath(od, "meta_$k.txt")) && continue
        s = deserialize(joinpath(sd, f))
        for (nm, m) in (("full", s.full), ("zoom", s.zoom)), fld in (:R, :T, :By, :Vx)
            write_npy(joinpath(od, "$(nm)_$(fld)_$k.npy"), Float32.(getfield(m, fld)))
        end
        open(joinpath(od, "meta_$k.txt"), "w") do io
            @printf(io, "t_code %.8g\nt_cc %.8g\nt_myr %.8g\n", s.t, s.t_cc, s.t_myr)
            @printf(io, "full_x0 %.8g\nfull_y0 %.8g\nfull_dx %.8g\n", s.full.x0, s.full.y0, s.full.dx)
            @printf(io, "zoom_x0 %.8g\nzoom_y0 %.8g\nzoom_dx %.8g\n", s.zoom.x0, s.zoom.y0, s.zoom.dx)
            println(io, "nblocks ", join(s.nblocks, " "))
        end
    end
    println("exported ", length(files), " snapshots of ", run)
end
