# ── the science-grade comparison gate (ADR-0006, flagship-1 strengthened) ─────
#
# One discrete Sedov IC — injected identically through each code's live-field
# bridge — through FOUR engines: Enzo PPM, RAMSES unsplit MUSCL+HLLC, and the
# PPMKernels guest slot on RAMSES's mesh (CPU f64 and Metal f32).  Gates:
# every engine's shock radius tracks the Sedov–Taylor R(t) computed from that
# run's MEASURED injected energy; the engines agree with each other; mass and
# energy are conserved; and the report (with the wall-clock column — the
# scheme-vs-scheme timing on identical meshes) is written.

using Test
using MultiCode
using EnzoLib, RamsesLib

# standalone runs need the hydro build too (runtests.jl sets this for the suite)
haskey(ENV, "RAMSES_LIB") || (ENV["RAMSES_LIB"] =
    normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..",
                      "mini-ramses", "bin64h", "libramses3d.dylib")))

const SEDOV_N = 64            # 64³: the bomb spans ~3 cells; R(t) ≈ 0.35

@testset "Sedov: one IC, every engine" begin
    if !(EnzoLib.grid_available() && RamsesLib.available())
        @test_skip false
    else
        spec = SedovCompareSpec()
        rows = NamedTuple[]

        function gate!(label, r)
            prof = sedov_profile(r.cs)
            Ra = sedov_radius(spec, r.t, r.E_in)
            lg = ledger(r.cs)
            E0 = spec.p0 / (spec.gamma - 1) + r.E_in
            f32 = startswith(label, "guest-metal")
            @test abs(lg.mass - spec.rho0) / spec.rho0 < (f32 ? 1e-6 : 1e-8)   # periodic box
            @test abs(lg.energy - E0) / E0 < (f32 ? 1e-3 : 1e-8)
            @test 0.85 < prof.R_shock / Ra < 1.05                      # the similarity solution
            push!(rows, (label = label, cs = r.cs, t = r.t, E_in = r.E_in,
                         seconds = r.seconds, steps = r.steps, profile = prof))
            @info "sedov engine" label R = prof.R_shock Ra ratio = prof.R_shock / Ra steps = r.steps seconds = round(r.seconds; digits = 2)
            r.free()
            return prof.R_shock
        end

        R_enzo = gate!("enzo-ppm", run_enzo_sedov(spec; n = SEDOV_N))
        R_nat = gate!("ramses-muscl", run_ramses_sedov(spec; level = 6))
        R_gst = gate!("guest-cpu", run_ramses_sedov(spec; level = 6, engine = :guest))
        R_res = gate!("guest-cpu-resident",
                      run_ramses_sedov(spec; level = 6, engine = :guest, resident = true))
        metal_ok = try
            @eval using Metal
            @eval using PPMKernels
            PPMKernels.has_backend(:metal)
        catch
            false
        end
        R_mtl = metal_ok ?
            gate!("guest-metal", run_ramses_sedov(spec; level = 6, engine = :guest, device = :metal)) :
            nothing
        R_mtlr = metal_ok ?
            gate!("guest-metal-resident",
                  run_ramses_sedov(spec; level = 6, engine = :guest, device = :metal, resident = true)) :
            nothing

        # cross-engine agreement on the shock position
        for (a, b) in ((R_enzo, R_nat), (R_nat, R_gst))
            @test abs(a - b) / ((a + b) / 2) < 0.05
        end
        @test abs(R_res - R_gst) / R_gst < 0.01                        # residency = plumbing only
        metal_ok && @test abs(R_mtl - R_gst) / R_gst < 0.02            # f32 ≈ f64
        metal_ok && @test abs(R_mtlr - R_gst) / R_gst < 0.02

        dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
        md = sedov_report(rows, spec; dir = dir)
        @test isfile(md) && isfile(joinpath(dir, "sedov_profiles.svg"))
        @info "Sedov comparison report" path = md engines = [r.label for r in rows]
    end
end
