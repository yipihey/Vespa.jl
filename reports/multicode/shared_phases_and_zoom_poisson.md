# MUSIC ↔ DISCO-DJ fixed-phase audit

MUSIC is exercised through its direct white-noise file inlet; the mirrored file is the Angulo-Pontzen control. DISCO-DJ is currently seed-driven through its NGenIC-compatible generator, so this report compares MUSIC's explicit white-noise field with DISCO-DJ's 1LPT finite-difference density proxy at the same integer seed.

| res | seed | corr(noise, readback) | corr(noise, mirror) | corr(MUSIC noise, DISCO-DJ proxy) | corr(proxy seed, seed+1) |
|-----|------|-----------------------|---------------------|----------------------------------|--------------------------|
| 16³ | 42 | 1.000000000000000 | -1.000000000000000 | 0.011234 | 0.022733 |

Interpretation: the MUSIC fixed/mirror path should be +1/-1 to round-off. A small same-seed MUSIC↔DISCO-DJ proxy correlation means the two wrappers do not yet share an explicit white-noise realization, even if their seed interfaces are both deterministic.
