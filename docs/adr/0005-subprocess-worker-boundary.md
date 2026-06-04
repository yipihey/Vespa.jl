# ADR-0005: Subprocess worker boundary (runtime-agnostic, single-contract RPC)

- **Status:** PROPOSED (design agreed; prototype next). Supersedes the in-process
  embedding approach of ADR-0004 for the multi-rank case.
- **Date:** 2026-06-04
- **Builds on:** ADR-0004 (optional MPI for the Enzo substrate). The C-ABI bridge
  (`EnzoModules/src/enzomodules_problem_bridge.C` ↔ `EnzoNG.jl/lib/EnzoLib/src/session.jl`).

---

## Context

ADR-0004 made Enzo MPI-capable, but driving it **in-process** from Julia is blocked by
a C++ runtime collision: loading the gcc-15/libstdc++ MPI stack (Enzo + MPIwrapper) into
the Julia process double-frees the `std::locale` at iostream static-init (Julia bundles its
own libstdc++; the MPI stack pulls in gcc-15's). The serial in-process path works because
no foreign MPI C++ runtime is loaded. Root cause is the classic "embed a gcc/libstdc++ C++
library into a Julia process" ABI collision — orthogonal to all of our code.

The fix is to stop sharing a process: run the physics worker (Enzo+MPI today; Legion/Regent
or a Rust backend later) in **its own process**, driven by Julia over a typed boundary. The
worker owns the grid and its inter-node communication; Julia orchestrates. No foreign runtime
ever enters the Julia process, so the collision cannot occur — for any worker runtime.

The design goal beyond "make it work": make it **hard to introduce header-drift or
data-corruption bugs** across the boundary. Both failure modes come from maintaining two
descriptions of one interface, so the design keeps exactly **one** contract plus a
**differential oracle**.

## Decisions

1. **One contract, derived from the existing ccalls — no second protocol.** The
   `enzomodules_*` C-ABI bridge already IS the complete interface, and `session.jl` wraps
   each function in exactly one `ccall`. A build-time generator parses those ccall
   declarations into a **function manifest** (name, scalar arg types, typed array buffers
   with shape/dtype) and emits BOTH the C++ worker-side dispatch and the Julia RPC stubs.
   Hand-written C++ stays only the real `enzomodules_*` implementations (already exist).
   This removes the error-prone "edit Grid.h + bridge.C + session.jl in sync" footprint
   (the exact class of bug that recurred during ADR-0004).

2. **Transport swapped behind one seam — zero API churn.** Every call already goes through
   `_gsym` + `ccall`. Abstract that into `_invoke(fn_id, sig, args...)` with two backends:
   **local = ccall** (the serial path, byte-identical to today) and **remote = RPC**. All of
   EnzoNG above `session.jl` is unchanged and agnostic to which it talks to. Chosen by config
   (e.g. `EngineConfig`/env), defaulting to local.

3. **POSIX shared memory for bulk field/flux arrays (zero-copy).** Each node runs its own
   Julia + worker pair (cross-node is worker↔worker over MPI, not this boundary), so the
   Julia↔worker link is always same-host. Field/flux arrays (`problem_get/set_field`,
   subgrid/boundary flux planes) are passed by `mmap`'d shared region + a descriptor
   `(dtype, rank, dims, nbytes)`; only the descriptor crosses the control channel. The
   control channel (unix socket/pipe) carries scalars and descriptors. Precision is explicit
   in the descriptor and routed through EnzoNG's existing convert-at-boundary pattern.

4. **Self-describing, validated frames.** Every buffer carries its descriptor; the receiver
   validates it against the manifest's declared signature. A dtype/shape/size mismatch is a
   hard error — never a reinterpret. Silent corruption is structurally impossible.

5. **Contract-hash handshake.** Both ends embed a hash of the manifest and compare it on
   connect; mismatch → refuse to run. Turns "rebuilt one side, not the other" (an ADR-0004
   failure mode) into a loud startup error instead of a corrupt run.

6. **The serial in-process path is the differential oracle.** Extend the fixture-parity gate:
   run the SAME problem through (a) in-process `ccall` and (b) subprocess-RPC and assert
   **bit-identical** results. Any header/data-passing bug surfaces as a parity divergence, so
   an RPC change that diverges from the proven local path cannot merge. This is what makes
   "easy to not introduce bugs" concrete.

## Runtime-agnostic worker

The worker is "a process that holds the grid, runs a physics backend, does its own inter-node
communication, and speaks the manifest protocol over shm + control channel." Enzo+MPI is the
first worker; Legion/Regent or a Rust backend slot in behind the same protocol and the same
parity oracle, with no change to EnzoNG above `session.jl`.

## Consequences / non-goals

- The `:julia` physics hooks run in the Julia process on grid data exposed via shm — so the
  hot path is mmap descriptor exchange, not data copies.
- Latency of per-call control-channel round-trips is bounded by batching (orchestrate at the
  level/grid granularity already used by `evolve_level!`), and bulk data never round-trips.
- Out of scope: cross-host Julia↔worker (each host is self-contained), and replacing the
  serial in-process path (it stays as both the fast local mode and the oracle).

## Prototype (decisive first slice)

Before the manifest generator: build the `_invoke` seam, one function round-trip
(`session_my_rank`) over the control channel, one shm field round-trip
(`problem_get_field`), and the parity assertion local≡remote on that pair. This proves the
boundary's correctness contract on the smallest surface, then the manifest generator scales
it to the full bridge.
