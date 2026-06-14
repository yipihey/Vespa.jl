# dfmm in the Sod harness (ADR-0006 Phase 5)

The dual-frame moment method (variational, symplectic, Lagrangian segments) on the harness Sod spec (γ = 5/3, t̂ = 0.2), via the `MultiCodeDfmmExt` package extension — certified against the same exact-Riemann oracle as the legacy engines.

| engine | cells | steps | wall-clock [s] | L1(ρ) | L1(u) | mass drift | total momentum |
|--------|-------|-------|----------------|-------|-------|------------|----------------|
| dfmm (τ=1e-03) | 100 | 244 | 7.02 | 0.0424 | 0.1029 | 0.0e+00 | 5.8e-17 |

Mass is conserved bit-exactly (the Lagrangian masses are labels, not state) and the total momentum stays at round-off — the variational integrator's exactness claims, reproduced inside the harness.
