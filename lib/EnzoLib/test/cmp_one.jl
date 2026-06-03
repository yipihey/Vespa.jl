# Per-problem subprocess worker for the Enzo test-suite harness. Runs ONE problem
# through Enzo's EvolveHierarchy and the Julia EvolveLevel and prints a parseable
# result line. Run in its own process so that an Enzo C++ abort (uncatchable
# EnzoFatalException → signal 6) on a problematic problem can't take the harness
# down — the harness sees a missing RESULT line and categorizes it.
#
#   julia --project=lib/EnzoLib/test lib/EnzoLib/test/cmp_one.jl <path/to/problem.enzo>

using EnzoLib
include(joinpath(@__DIR__, "enzo_suite_common.jl"))

function main()
    pf0 = ARGS[1]
    nm = basename(dirname(pf0))
    fl = problem_flags(pf0)
    pf = paramfile_for(pf0)        # serial-build gravity patch (UnigridTranspose=0) if needed
    # Reference first, so a failure here is attributed to Enzo (not the Julia driver).
    se = try
        s = EnzoLib.evolve_problem_state(pf)
        println("REFOK|", nm)
        s
    catch e
        println("RESULT|", nm, "|status=enzo_error|", first(sprint(showerror, e), 100))
        return
    end
    try
        sj = EnzoLib.run_amr_state(pf; gravity = fl.gravity, cooling = fl.cooling,
                                   radiation = fl.radiation, star_sources = fl.star_sources,
                                   star_formation = fl.star_formation, cosmology = fl.cosmology)
        r = _max_field_error(sj.fields, se.fields)
        p = _max_particle_error(sj.particles, se.particles)
        println("RESULT|", nm, "|status=ok|err=", r.err, "|nfields=", r.nfields,
                "|perr=", p.err, "|nparticles=", p.nparticles,
                "|hm=", fl.hydromethod, "|grav=", fl.gravity, "|cool=", fl.cooling, "|rad=", fl.radiation)
    catch e
        println("RESULT|", nm, "|status=julia_error|", first(sprint(showerror, e), 100))
    end
end

main()
