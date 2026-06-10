# ── the comparison report (one spec, N codes, one page) ──────────────────────

const _SVG_COLORS = Dict(:enzo => "#d62728", :ramses => "#1f77b4", :arepo => "#2ca02c")

# Minimal hand-rolled SVG: the density profiles of every code overlaid on the
# exact solution.  No plotting dependency — the report must build headless.
function _sod_svg(results, spec::SodSpec; w = 760, hgt = 460)
    pad = 55
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.1 * spec.rhoL
    sx(x) = pad + (x - xmin) / (xmax - xmin) * (w - 2pad)
    sy(y) = hgt - pad - (y - ymin) / (ymax - ymin) * (hgt - 2pad)
    poly(xs, ys) = join(("$(round(sx(x); digits=2)),$(round(sy(y); digits=2))"
                         for (x, y) in zip(xs, ys)), " ")
    io = IOBuffer()
    print(io, """
    <svg xmlns="http://www.w3.org/2000/svg" width="$w" height="$hgt" viewBox="0 0 $w $hgt">
    <rect width="$w" height="$hgt" fill="white"/>
    <text x="$(w ÷ 2)" y="22" text-anchor="middle" font-family="sans-serif" font-size="15">
      Sod shock tube at t̂ = $(spec.t) — density (one spec, every code, exact overlay)</text>
    <line x1="$pad" y1="$(hgt - pad)" x2="$(w - pad)" y2="$(hgt - pad)" stroke="black"/>
    <line x1="$pad" y1="$pad" x2="$pad" y2="$(hgt - pad)" stroke="black"/>
    <text x="$(w ÷ 2)" y="$(hgt - 14)" text-anchor="middle" font-family="sans-serif" font-size="12">x̂</text>
    <text x="16" y="$(hgt ÷ 2)" font-family="sans-serif" font-size="12" transform="rotate(-90 16 $(hgt ÷ 2))">ρ</text>
    """)
    for v in 0.0:0.25:1.0   # x ticks
        print(io, """<line x1="$(sx(v))" y1="$(hgt - pad)" x2="$(sx(v))" y2="$(hgt - pad + 5)" stroke="black"/>
        <text x="$(sx(v))" y="$(hgt - pad + 18)" text-anchor="middle" font-family="sans-serif" font-size="11">$v</text>
        """)
    end
    for v in 0.0:0.25:1.0   # y ticks
        print(io, """<line x1="$(pad - 5)" y1="$(sy(v))" x2="$pad" y2="$(sy(v))" stroke="black"/>
        <text x="$(pad - 9)" y="$(sy(v) + 4)" text-anchor="end" font-family="sans-serif" font-size="11">$v</text>
        """)
    end
    # exact solution, densely sampled
    xs = range(0.001, 0.999; length = 600)
    ys = [exact_sod(spec, (x - spec.x0) / spec.t).rho for x in xs]
    print(io, """<polyline points="$(poly(xs, ys))" fill="none" stroke="black" stroke-width="2.2" stroke-dasharray="6,3"/>\n""")
    # one polyline per code
    ly = 50
    print(io, """<text x="$(w - pad - 150)" y="$ly" font-family="sans-serif" font-size="12">exact (dashed)</text>\n""")
    for r in results
        c = get(_SVG_COLORS, r.code, "#777777")
        print(io, """<polyline points="$(poly(r.profile.x, r.profile.rho))" fill="none" stroke="$c" stroke-width="1.6"/>\n""")
        ly += 18
        print(io, """<rect x="$(w - pad - 170)" y="$(ly - 10)" width="12" height="3" fill="$c"/>
        <text x="$(w - pad - 150)" y="$ly" font-family="sans-serif" font-size="12">$(r.code) ($(length(r.profile.x)) pts, L1ρ=$(round(r.l1.rho; sigdigits=3)))</text>\n""")
    end
    print(io, "</svg>\n")
    return String(take!(io))
end

"""
    sod_report(results, spec; dir) -> path

Write the cross-code Sod report: a Markdown page (spec, conservation ledgers
vs the analytic reference, L1 errors, round-trip gate results, notes) plus the
SVG profile overlay next to it.  `results` is a vector of NamedTuples with
keys `code, cs, t, profile, l1, roundtrip, notes`.
"""
function sod_report(results, spec::SodSpec; dir::AbstractString)
    mkpath(dir)
    svg = joinpath(dir, "sod_profiles.svg")
    write(svg, _sod_svg(results, spec))
    ref = sod_reference_ledger(spec)
    md = joinpath(dir, "sod_comparison.md")
    open(md, "w") do io
        println(io, "# One Sod shock tube, every code (ADR-0006 Phase 2)\n")
        println(io, "Spec: (ρ,u,p)L = ($(spec.rhoL), $(spec.uL), $(spec.pL)) | ",
                "(ρ,u,p)R = ($(spec.rhoR), $(spec.uR), $(spec.pR)), γ = $(spec.gamma), ",
                "x̂₀ = $(spec.x0), compared at t̂ = $(spec.t).\n")
        println(io, "Each code ran its NATIVE setup path (Enzo `SodShockTube.enzo`, a generated ",
                "RAMSES namelist, Arepo's `shocktube_1d` example); the canonical-state adapters ",
                "(`MultiCode.CellSet`) did all conversion at the bridge boundary.\n")
        println(io, "![density profiles](sod_profiles.svg)\n")
        println(io, "## Conservation ledgers (normalized units; analytic reference: ",
                "mass = $(ref.mass), energy = $(ref.energy))\n")
        println(io, "| code | cells | t̂ | mass | Δmass/mass | energy | Δenergy/energy | L1(ρ) | L1(u) | round-trip |")
        println(io, "|------|-------|----|------|-----------|--------|----------------|-------|-------|------------|")
        for r in results
            lg = ledger(r.cs)
            dm = abs(lg.mass - ref.mass) / ref.mass
            de = abs(lg.energy - ref.energy) / ref.energy
            @printf(io, "| %s | %d | %.6g | %.10g | %.2e | %.10g | %.2e | %.4g | %.4g | %s |\n",
                    r.code, ncells(r.cs), r.t, lg.mass, dm, lg.energy, de,
                    r.l1.rho, r.l1.u, r.roundtrip ? "bit-identical ✓" : "FAILED ✗")
        end
        println(io, "\n## Notes\n")
        for r in results
            isempty(r.notes) || println(io, "- **$(r.code)**: $(r.notes)")
        end
        println(io, "- Energy drift reflects each code's boundary treatment and is bounded by ",
                "the waves staying inside the box at t̂ = $(spec.t).")
        println(io, "- The Arepo round-trip gate covers its settable conserved surface ",
                "(cell energy + momentum); density is derived by Arepo and not directly settable.")
    end
    return md
end
