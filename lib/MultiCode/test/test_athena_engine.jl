# ── Athena++ in the Sod harness (the wrapper-registry on-ramp) ────────────────
#
# The fourth legacy engine: the stock athinput.sod run in-process through
# AthenaLib (the MultiCodeAthenaExt package extension), gated against the SAME
# exact-Riemann oracle as Enzo/RAMSES/Arepo/dfmm.  Skips cleanly when the
# Athena++ capi library is not built.

using Test
using Printf
using MultiCode
using AthenaLib                  # activates MultiCodeAthenaExt

function _fresh_julia_ok(code::AbstractString)
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $code`
    return success(pipeline(cmd; stdout = stdout, stderr = stderr))
end

@testset "the Athena++ engine in the Sod harness" begin
    if !AthenaLib.available()
        @warn "libathena_capi not found — skipping" lib = AthenaLib.libpath()
        @test_skip false
    else
        @test _fresh_julia_ok(raw"""
            using MultiCode, AthenaLib
            n = 8
            nc = n^3
            pos = Matrix{Float64}(undef, nc, 3)
            vol = fill(1.0 / nc, nc)
            rho = ones(nc)
            mom = zeros(nc, 3)
            etot = fill(2.5, nc)
            q = 0
            for k in 1:n, j in 1:n, i in 1:n
                global q += 1
                pos[q, 1] = (i - 0.5) / n
                pos[q, 2] = (j - 0.5) / n
                pos[q, 3] = (k - 0.5) / n
            end
            cs = CellSet(:synthetic, pos, vol, rho, mom, etot,
                         (length = 1.0, time = 1.0, density = 1.0), (;))
            r = athena_stage_cellset(cs; dims = (n, n, n), dt = 0.01, gamma = 1.4)
            @assert MultiCode.ncells(r.cs) == nc
            @assert r.cs.code == :athena_stage
            @assert r.ledger_drift < 1e-10
            @assert maximum(abs.(r.cs.rho .- 1.0)) < 1e-10
            @assert maximum(abs.(r.cs.etot .- 2.5)) < 1e-10
        """)

        spec = SodSpec()                                   # γ=1.4, t̂=0.1, x0=0.5
        # ── Next-13: 3-D Sod → the CANONICAL state (VTK readback) ─────────────
        # RUNS FIRST: Athena++ re-entrancy survives same-or-lower dimensionality
        # (3D→1D fine, re-run fine) but a 1D→3D sequence in one process
        # SEGFAULTS (a static sized at first init) — so 3-D before 1-D.
        # The stock problem extruded to 32³, one MeshBlock → one legacy-VTK
        # file → CellSet.  Ledgers gate at the float32 VTK floor; the problem
        # is 1-D physics in a 3-D box, so the transverse scatter must be ZERO.
        r3 = run_athena_sod3d(spec)
        lg = MultiCode.ledger(r3.cs)
        ref = MultiCode.sod_reference_ledger(spec)
        l13 = MultiCode.sod_l1(r3.profile, spec, r3.t)
        @test abs(lg.mass - ref.mass) / ref.mass < 1e-6       # f32 VTK floor
        @test abs(lg.energy - ref.energy) / ref.energy < 1e-6
        @test r3.profile.scatter == 0.0                       # bit-identical columns
        @test l13.rho < 0.05
        @test MultiCode.ncells(r3.cs) == 32^3
        @info "Athena 3-D → CellSet" mass_err = abs(lg.mass - ref.mass) / ref.mass l1_rho = l13.rho seconds = round(r3.seconds; digits = 2)

        r = run_athena_sod(spec)
        l1 = MultiCode.sod_l1(r.profile, spec, r.t)
        @test r.t_end ≈ spec.t atol = 1e-6                 # reached the epoch
        @test r.mass_drift < 1e-12                         # exact conservation (.hst)
        @test l1.rho < 0.05                                # the PPM-class L1 band
        @test l1.u < 0.05
        @info "Athena++ Sod vs exact" l1_rho = l1.rho l1_u = l1.u mass_drift = r.mass_drift seconds = round(r.seconds; digits = 2)

        dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
        mkpath(dir)
        md = joinpath(dir, "athena_sod.md")
        open(md, "w") do io
            println(io, "# Athena++ in the Sod harness (wrapper-registry on-ramp)\n")
            println(io, "The stock `athinput.sod` (γ = 1.4, interface recentred to x = 0.5) ",
                    "run IN-PROCESS through AthenaLib via the `MultiCodeAthenaExt` package ",
                    "extension, against the same exact-Riemann oracle as the other engines.\n")
            println(io, "| engine | cells | wall-clock [s] | L1(ρ) | L1(u) | mass drift |")
            println(io, "|--------|-------|----------------|-------|-------|------------|")
            @printf(io, "| athena++ (:hydro) | %d | %.2f | %.4f | %.4f | %.1e |\n",
                    r.diag.nx1, r.seconds, l1.rho, l1.u, r.mass_drift)
        end
        @test isfile(md)
        @info "Athena engine report" path = md

    end
end
