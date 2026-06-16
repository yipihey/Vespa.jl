# Predicate Plan for AREPO Tessellator Port

## Scope
- Source basis: `voronoi.h`, `voronoi_3d.c`, `predicates.c`.
- Goal: preserve AREPO predicate semantics on CPU, while making GPU evaluation safe and explicit about ambiguity.

## 1) Predicate path map

| Call site | Fast path | Ambiguous path | Exact path | Notes |
|---|---|---|---|---|
| `Orient3d()` | `Orient3d_Quick`-style FP determinant with error bound | `|x| <= 1e-14 * sizelimit` | `Orient3d_Exact()` | Infinity short-circuits to `0`. |
| `InSphere_Errorbound()` | FP determinant with `errbound = 1e-14 * sizelimit` | `|x| <= errbound` | `InSphere_Exact()` | Infinity short-circuits to `-1`. |
| `InTetra()` | `solve_linear_equations()` + barycentric inside test | outside/near-face/near-edge cases | `Orient3d_Exact()` on 4 subtests | Used for point location and flip setup. |
| `convex_edge_test()` | `solve_linear_equations()` + barycentric classification | edge/triangle boundary cases | `Orient3d_Exact()` on 3 subtests | Decides 2-3, 3-2, 4-4, or no flip. |
| `get_tetra()` / insertion loop | `InSphere_Errorbound()` | `ret == 0` | `InSphere_Exact()` | Legal/illegal facet test before flips. |

## 2) Exact / integer / GMP behavior

- `voronoi.h` maps doubles to integer-like coordinates with `double_to_voronoiint()` / `mask_voronoi_int()` and `USEDBITS=52`.
- Exact predicates operate on these integer-mapped coordinates, not on raw doubles.
- With normal memory mode, exact functions use cached `point.ix/iy/iz`; with `OPTIMIZE_MEMORY_USAGE`, they reconstruct mapped integers on demand via `get_integers_for_point()`.
- `Orient3d_Exact()` and `InSphere_Exact()` build `mpz_t` expressions and return only the sign via `mpz_sgn()`.
- `get_circumcircle_exact()` also uses GMP (`mpz_cdiv_q`, `mpz_tdiv_q_2exp`) for exact circumcenter recovery.
- Trigger rule: exact GMP work is entered only after a quick/error-bounded test returns `0`, or when the caller explicitly needs an exact sign after a borderline linear solve.

## 3) GPU-safe adaptive strategy

1. GPU evaluates the same algebra as the quick/error-bounded predicates, using deterministic double precision and the same operand order as AREPO where practical.
2. GPU accepts the result only when the sign is safely outside the bound:
   - orientation: `|x| > 1e-14 * sizelimit`
   - in-sphere: `|x| > 1e-14 * sizelimit`
3. If the result is zero/gray-zone, or any input is infinite/invalid, emit a fallback ticket to CPU exact evaluation.
4. GPU does not attempt GMP, arbitrary precision, or topology decisions from ambiguous predicates.
5. Preserve periodic/image metadata in the fallback ticket so the CPU exact replay sees the same face ownership and orientation context.

## 4) CPU fallback policy

- CPU exact is the authority for every ambiguous predicate and every topology-changing decision.
- If GPU and CPU disagree on a supposedly safe predicate, treat that as a hard divergence: log it, increment a mismatch counter, and trust CPU exact.
- If a fallback queue grows beyond budget, stop accepting new GPU topology decisions and temporarily run the local rebuild on CPU only.
- Never silently coerce a zero-sign predicate into an arbitrary nonzero sign on GPU.

## 5) Counters required for gates

### Existing AREPO counters to carry through
- `CountInSphereTests`
- `CountInSphereTestsExact`
- `CountConvexEdgeTest`
- `CountConvexEdgeTestExact`
- `Count_InTetra`
- `Count_InTetraExact`
- `CountFlips`
- `Count_1_to_4_Flips`, `Count_2_to_3_Flips`, `Count_3_to_2_Flips`, `Count_4_to_4_Flips`
- `Count_EdgeSplits`
- `Count_FaceSplits`

### New gate counters to add at the port boundary
- `predicate_orient_fast_accept`
- `predicate_orient_ambiguous`
- `predicate_orient_exact_cpu`
- `predicate_insphere_fast_accept`
- `predicate_insphere_ambiguous`
- `predicate_insphere_exact_cpu`
- `predicate_infinity_short_circuit`
- `predicate_gpu_fallback_tickets`
- `predicate_gpu_cpu_mismatches`
- `predicate_cpu_exact_replays`

## 6) Gate reporting shape

- Per run, report total calls, ambiguous fraction, exact fallback count, and mismatch count for each predicate family.
- For drift/rebuild gates, also report flip counts and split counts, because predicate ambiguity should correlate with topology churn.
- For CPU-vs-GPU parity gates, require exact replay completion before comparing compact arrays.
- For scaling gates, compare exact-fallback rate versus problem size; rising ambiguity is a signal, not an automatic bug.

