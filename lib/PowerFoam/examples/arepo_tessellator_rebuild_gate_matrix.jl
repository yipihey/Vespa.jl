using Printf

gate_command(n, repeats; env = false) = string(
    env ? "POWERFOAM_NATIVE_TRACE_REPEAT=$(repeats) " : "",
    "julia --project=lib/PowerFoam -e '",
    "push!(LOAD_PATH, \"/Users/tabel/Projects/Arepo.jl/lib/ArepoLib\"); ",
    "ARGS=[\"$(n)\",\"0.001\",\"hll\",\"$(repeats)\"]; ",
    "include(\"lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl\")'")

const ROWS = [
    (
        "N4 post-drift",
        gate_command(4, 1),
        "lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N4_dt0p001_hll/README.md",
    ),
    (
        "N8 post-drift",
        gate_command(8, 1),
        "lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N8_dt0p001_hll/README.md",
    ),
    (
        "N12 post-drift",
        gate_command(12, 1),
        "lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N12_dt0p001_hll/README.md",
    ),
    (
        "N4 repeated-drift",
        gate_command(4, 3; env = true),
        "lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N4_dt0p001_hll_repeat3/README.md",
    ),
    (
        "N8 repeated-drift",
        gate_command(8, 3; env = true),
        "lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N8_dt0p001_hll_repeat3/README.md",
    ),
    (
        "N12 repeated-drift",
        gate_command(12, 3; env = true),
        "lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N12_dt0p001_hll_repeat3/README.md",
    ),
]

function main()
    println("# AREPO Tessellator Port Rebuild Gate Matrix")
    println()
    println("| Gate | Command | Artifact |")
    println("| --- | --- | --- |")
    for row in ROWS
        @printf("| %s | `%s` | `%s` |\n", row[1], row[2], row[3])
    end
end

main()
