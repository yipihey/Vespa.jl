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
        ref = Dict(k => (lev.tabs[k][1], Array(lev.tabs[k][2])) for k in (:ctres, :ctcorn, :ctface))
        BA.build_ct_tables!(h, 1)
        new = Dict(k => (lev.tabs[k][1], Array(lev.tabs[k][2])) for k in (:ctres, :ctcorn, :ctface))
        @test new == ref
    end
end
