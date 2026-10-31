# test_tables_fast.jl — the allocation-light C/F table builders (build_cf_faces,
# build_ct_tables!) produce exactly the reference builders' tables, on 2-D
# hierarchies with fine patches in the interior and touching domain faces and
# corners, periodic and outflow, hydro and CT.
for bc in (:periodic, :outflow), physics in (:hydro, :mhd)
    h = static2d(; nbase = (64, 32), kids = KIDS_ALL, bc, physics,
                 divb = physics === :mhd ? :ct : :glm)
    lev = h.levels[2]; plev = h.levels[1]
    @test BA.build_cf_faces(lev, plev) == BA._build_cf_faces_ref(lev, plev)
    if physics === :mhd
        BA._build_ct_tables_ref!(h, 1)
        ref = Dict(k => (lev.tabs[k][1], Array(lev.tabs[k][2])) for k in (:ctcorn, :ctface))
        # face restriction: the reference face list vs the device-built per-block path
        fx = rand(Float32, length(lev.FBx)); fy = rand(Float32, length(lev.FBy))
        copyto!(lev.FBx, fx); copyto!(lev.FBy, fy)
        cx0 = Array(plev.FBx); cy0 = Array(plev.FBy)
        BA._ct_restrict_faces!(lev, plev); rx = Array(plev.FBx); ry = Array(plev.FBy)
        copyto!(plev.FBx, cx0); copyto!(plev.FBy, cy0)
        BA.build_ct_tables!(h, 1)
        new = Dict(k => (lev.tabs[k][1], Array(lev.tabs[k][2])) for k in (:ctcorn, :ctface))
        @test new == ref
        BA._ct_restrict_faces!(lev, plev)
        @test Array(plev.FBx) == rx && Array(plev.FBy) == ry
        @test rx != cx0                                       # the restriction did write
    end
end

# fused multi-field ghost fill / restriction ≡ the per-field launch train, bitwise
for BE in BACKENDS, bc in (:periodic, :outflow), physics in (:hydro, :mhd)
    res = Dict{Bool,Any}()
    for fused in (false, true)
        BA._FUSED[] = fused
        h = static2d(; nbase = (64, 32), kids = KIDS_ALL, bc, physics, backend = BE,
                     divb = physics === :mhd ? :ct : :glm)
        setall!(h, x -> (1 + 0.3sin(2π * x[1]) * cos(2π * x[2]), 0.2, -0.1, 0.0, 1 + 0.2x[2]))
        for l in 0:1; fill_ghosts!(h, l; θ = 0.25f0, buf = :R); end
        restrict_level!(h, 1)
        res[fused] = [Array(getfield(h.levels[l + 1], f)) for l in 0:1
                      for f in (:D, :S1, :S2, :S3, :Tau, :Ge, :Bx, :By, :Bz, :Psi) if !isempty(getfield(h.levels[l + 1], f))]
    end
    BA._FUSED[] = true
    @test res[true] == res[false]
end
