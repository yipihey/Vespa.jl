"""
    CodeBridge

The shared substrate for wrapping legacy simulation codes (ADR-0006 D1).
EnzoLib, RamsesLib, and ArepoLib each hand-rolled the same five mechanisms;
this package is their single home, extracted from EnzoLib (where every piece
was proven under the ADR-0005 parity oracle):

1. **`LazyLib`** — a lazily-`dlopen`'d shared library: env-var override +
   default path, optional dlopen flags and a pre-open hook (e.g. promoting an
   MPI shim to global scope), with the build hint in the not-found error.
2. **`Bridge`** — one wrapped legacy code: a named bundle of `LazyLib` flavors
   (CPU/Metal, serial/MPI), the active transport backend, the worker
   connection, and the wire contract.
3. **`@xcall`** — ONE call macro, two transports. Local → in-process `ccall`
   (byte-identical to a hand-written binding); remote → RPC to a worker
   process. The C symbol + return type + arg types are written once at the
   call site; there is no second hand-maintained interface to drift.
4. **`manifest`/`contract_hash`** — the bridge surface parsed out of the
   `@xcall` sites themselves; the FNV-1a hash over its canonical serialization
   is exchanged at the worker handshake, so a worker built from different
   bindings is refused rather than silently corrupting data.
5. **`connect_worker!`/`serve`** — the subprocess worker boundary (ADR-0005):
   a line-based control channel + a shared file for bulk arrays, letting a
   foreign-runtime worker (MPI C++, a second legacy code, N instances of a
   global-state singleton) run OUTSIDE the Julia process.

A client wrapper declares `const BRIDGE = CodeBridge.Bridge(...)` and writes
its bindings in `@xcall` style; everything else (loading, transport, contract,
worker) comes from here. The wire protocol and contract serialization are
byte-identical to EnzoLib's originals, so existing built workers (with baked
contract hashes) remain valid.
"""
module CodeBridge

using Libdl
using Base64

export LazyLib, Bridge, @xcall

include("lazylib.jl")
include("bridge.jl")
include("rpc.jl")

end # module
