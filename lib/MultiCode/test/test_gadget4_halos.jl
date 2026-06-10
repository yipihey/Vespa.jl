# ── GADGET-4 halo finding on harness particles (the last wrapper on-ramp) ────
#
# G4's FOF+SUBFIND as a SERVICE through `run_gadget4_halos` (MultiCode
# conventions in, catalogue out; child process — the D2 transport).  Two
# gates: (a) three planted clumps in a sparse background → EXACTLY three
# groups, each its ~500 particles (the physics gate); (b) a LIVE RAMSES
# Zel'dovich run's particles at a/aᵢ = 4 (pre-caustic, no collapse) → zero
# groups — a real foreign-code particle dump through the service.

haskey(ENV, "RAMSES_LIB") || (ENV["RAMSES_LIB"] =
    normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..",
                      "mini-ramses", "bin64h", "libramses3d.dylib")))

using Test
using Printf
using Random
using MultiCode
using Gadget4Lib                  # activates MultiCodeGadget4Ext
using RamsesLib, CodeBridge
# NOTE: Enzo's grid dylib carries its OWN libhdf5; together with HDF5.jl (via
# Gadget4Lib) the interposed symbols abort the process — so the live-code half
# of this gate uses RAMSES (plain Fortran I/O, no HDF5).  Enzo's particles can
# ride the same service through its worker transport (ADR-0005) when needed.

@testset "GADGET-4 halo service on harness particles" begin
    if !Gadget4Lib.available()
        @warn "GADGET-4 bridge not built — skipping"
        @test_skip false
    else
        # ── (a) planted clumps, MultiCode conventions ([0,1)³ rows) ───────────
        rng = Random.MersenneTwister(42)
        box = 50.0
        centers = ([12.0, 12.0, 12.0], [38.0, 14.0, 30.0], [20.0, 40.0, 40.0])
        npc, nbg = 500, 2000
        xp = Matrix{Float64}(undef, 3npc + nbg, 3)
        for (c, ctr) in enumerate(centers)
            for p in 1:npc
                xp[(c-1)*npc+p, :] .= mod.(ctr .+ 0.5 .* randn(rng, 3), box) ./ box
            end
        end
        for p in 1:nbg
            xp[3npc+p, :] .= rand(rng, 3)
        end
        r = run_gadget4_halos(xp; box_mpch = box, omega_m = 0.308)
        @test r.ngroups == 3                          # exactly the planted clumps
        @test all(l -> 400 <= l <= 600, r.group_lens)
        @test r.nsubhalos >= 3
        @info "G4 halo service: planted clumps" ngroups = r.ngroups lens = r.group_lens

        # ── (b) LIVE RAMSES particles through the service (pre-collapse → 0) ──
        if CodeBridge.available(RamsesLib.BRIDGE, :cosmo)
            rz = run_ramses_zeldovich(ZeldovichSpec())
            try
                rh = run_gadget4_halos(rz.xp; box_mpch = 32.0, omega_m = 1.0,
                                       redshift = 11.5)
                @test rh.ngroups == 0                 # a pre-caustic plane wave
                @info "G4 halo service: live RAMSES particles" ngroups = rh.ngroups np = size(rz.xp, 1)
            finally
                rz.free()
            end
        else
            @test_skip false
        end

        dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
        mkpath(dir)
        md = joinpath(dir, "gadget4_halos.md")
        open(md, "w") do io
            println(io, "# GADGET-4 halo finding as a harness service (ADR-0006 on-ramp)\n")
            println(io, "FOF+SUBFIND on particles in MultiCode's conventions via ",
                    "`MultiCodeGadget4Ext` (child-process transport): three planted clumps ",
                    "→ exactly **$(r.ngroups) groups** ($(r.group_lens) particles each); a ",
                    "live RAMSES Zel'dovich dump at a/aᵢ = 4 → **0 groups** (pre-caustic), ",
                    "demonstrating the foreign-particle service end-to-end.")
        end
        @test isfile(md)
    end
end
