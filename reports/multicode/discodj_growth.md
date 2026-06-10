# DISCO-DJ ICs through Enzo + RAMSES (ADR-0006 wrapper on-ramp)

DISCO-DJ's differentiable JAX 1LPT field as ZERO-velocity particles (no velocity-unit convention), evolved aᵢ → 4aᵢ in EdS through both codes; the whole linear field follows b(x) = ⅗x + ⅖x^{−3/2}.

| engine | steps | a/aᵢ | large-scale growth | b(a) exact | ratio | corr vs ICs |
|--------|-------|------|--------------------|------------|-------|-------------|
| enzo | 71 | 4.052 | 2.3880 | 2.4800 | 0.9629 | 0.9678 |
| ramses | 14 | 4.205 | 2.5108 | 2.5692 | 0.9773 | 0.9719 |

Enzo ↔ RAMSES final-field correlation: **0.9981** — differentiable ICs, two legacy codes, one analytic answer.
