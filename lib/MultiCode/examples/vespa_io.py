"""
vespa_io — read a CICASS run directory (one source of truth for the dump format).

A run dir is what `MultiCode.run_dir` / `vrun` create on scratch: it holds the
`.dat` P(k) tables, the `.bin` field dumps, and a `fields.toml` side-car written by
`MultiCode.write_grid` that describes every dump (columns, per-column dtype, F-order,
header, ndim, n).  `RunDir` reads that side-car so the ~20 analysis scripts no longer
each re-encode the binary layout.

Usage:
    from vespa_io import open_run
    rd = open_run()                         # uses $VESPA_RUN_DIR, or sys.argv[1]
    g  = rd.grid("enzo_xspec", z=460)       # -> {"rho_b": (N,N,N), "rho_dm": ...}
    for z in rd.redshifts("enzo_cellcmp"):  # available redshifts for a kind
        T = rd.field("enzo_cellcmp", z, "T")
    blocks = rd.pk()                        # {(z, "baryon"): (k, P), ...} from *_pk*.dat

Old archived runs without a `fields.toml` still load via a built-in legacy schema
keyed by the dump-name suffix (xspec/cellcmp/fields/slice/phase) + the Int64-header
heuristic.
"""
import os
import re
import sys
import glob
import numpy as np

# Julia eltype name -> numpy dtype (little-endian, matches the x86-64 writer)
_DT = {"Float64": "<f8", "Float32": "<f4", "Int64": "<i8", "Int32": "<i4",
       "UInt8": "u1", "Int8": "i1"}

# legacy fallback: dump-name suffix -> (column names, ndim, has_header). dtypes all f8.
_LEGACY = {
    "xspec":   (["rho_b", "rho_dm"], 3, True),
    "cellcmp": (["rho_b", "xHII", "fH2", "fHD", "T"], 3, True),
    "slice":   (["rho_b", "rho_dm", "vx", "vy"], 2, True),
    "fields":  (["phi", "rho_dm"], 3, False),
    "phase":   (["rrel", "nH", "T", "fH2", "xHII"], 1, True),
}

_ZRE = re.compile(r"_z(\d+)\.bin$")


def _load_toml(path):
    """Parse fields.toml. Prefer stdlib tomllib (3.11+); else a tiny purpose parser."""
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f).get("field", [])
    except Exception:
        pass
    # minimal parser for our own [[field]] array-of-tables of scalars/string-lists
    fields, cur = [], None

    def _val(s):
        s = s.strip()
        if s.startswith("["):
            inner = s[1:s.rindex("]")]
            return [x.strip().strip('"') for x in inner.split(",") if x.strip()]
        if s in ("true", "false"):
            return s == "true"
        if s.startswith('"'):
            return s.strip('"')
        try:
            return int(s)
        except ValueError:
            return s
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "[[field]]":
                cur = {}
                fields.append(cur)
            elif "=" in line and cur is not None:
                k, v = line.split("=", 1)
                cur[k.strip()] = _val(v)
    return fields


class RunDir:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"run dir not found: {self.path}")
        self._desc = {}  # basename -> descriptor dict
        tp = os.path.join(self.path, "fields.toml")
        if os.path.isfile(tp):
            for d in _load_toml(tp):
                self._desc[d["file"]] = d

    # -- discovery -----------------------------------------------------------
    def _suffix(self, kind):
        return kind.split("_")[-1]

    def files(self, kind):
        return sorted(glob.glob(os.path.join(self.path, f"{kind}*_z*.bin")))

    def redshifts(self, kind):
        zs = []
        for f in self.files(kind):
            m = _ZRE.search(os.path.basename(f))
            if m:
                zs.append(int(m.group(1)))
        return sorted(set(zs), reverse=True)

    def _file_for(self, kind, z):
        for f in self.files(kind):
            m = _ZRE.search(os.path.basename(f))
            if m and int(m.group(1)) == int(z):
                return f
        raise FileNotFoundError(f"no {kind} dump at z={z} in {self.path}")

    # -- the loader ----------------------------------------------------------
    def _descriptor(self, fname, kind):
        d = self._desc.get(fname)
        if d is not None:
            cols = d["columns"]
            dts = d.get("dtypes") or [d.get("dtype", "Float64")] * len(cols)
            return cols, [_DT[t] for t in dts], int(d["ndim"]), bool(d["has_header"])
        # legacy fallback by suffix
        suf = self._suffix(kind)
        if suf not in _LEGACY:
            raise KeyError(f"no fields.toml entry for {fname} and no legacy schema "
                           f"for suffix '{suf}'")
        cols, ndim, has_header = _LEGACY[suf]
        return cols, ["<f8"] * len(cols), ndim, has_header

    def grid(self, kind, z):
        """Return {name: ndarray} for a dump. Grid kinds reshape to (N,)*ndim, F-order;
        flat kinds (ndim==1) return 1-D arrays."""
        fname = os.path.basename(self._file_for(kind, z))
        full = os.path.join(self.path, fname)
        cols, dts, ndim, has_header = self._descriptor(fname, kind)
        with open(full, "rb") as fh:
            header = None
            if has_header:
                header = int(np.fromfile(fh, dtype="<i8", count=1)[0])
            out = {}
            if ndim == 1:
                # column length = header (count), each column its own dtype
                n = header if header is not None else None
                for name, dt in zip(cols, dts):
                    a = np.fromfile(fh, dtype=dt, count=n)
                    out[name] = a
            else:
                N = header if header is not None else self._infer_N(full, dts, ndim, cols)
                m = N ** ndim
                shape = (N,) * ndim
                for name, dt in zip(cols, dts):
                    a = np.fromfile(fh, dtype=dt, count=m).reshape(shape, order="F")
                    out[name] = a
                out["_N"] = N
        return out

    def field(self, kind, z, name):
        return self.grid(kind, z)[name]

    def _infer_N(self, full, dts, ndim, cols):
        # headerless grid (e.g. *_fields*): N from file size and column count/width
        size = os.path.getsize(full)
        width = sum(np.dtype(dt).itemsize for dt in dts)
        m = size // width
        N = round(m ** (1.0 / ndim))
        if N ** ndim != m:
            raise ValueError(f"cannot infer N for {full} (m={m}, ndim={ndim})")
        return N

    # -- P(k) tables ---------------------------------------------------------
    def pk(self, pattern="*_pk*.dat"):
        """Parse `@ z=<z> <component>` blocks from the .dat tables in this run dir.
        Returns {(z_float, component): (k_array, P_array)}."""
        blocks = {}
        for fn in sorted(glob.glob(os.path.join(self.path, pattern))):
            cur, ks, Ps = None, [], []
            with open(fn) as f:
                for line in f:
                    if line.startswith("@"):
                        if cur is not None:
                            blocks[cur] = (np.array(ks), np.array(Ps))
                        p = line.split()
                        cur = (float(p[1].split("=")[1]), p[2])
                        ks, Ps = [], []
                    elif line.strip() and not line.startswith("#"):
                        a, b = line.split()[:2]
                        ks.append(float(a)); Ps.append(float(b))
            if cur is not None:
                blocks[cur] = (np.array(ks), np.array(Ps))
        return blocks


def open_run(path=None):
    """RunDir from `path`, else $VESPA_RUN_DIR, else sys.argv[1]."""
    path = path or os.environ.get("VESPA_RUN_DIR") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not path:
        raise SystemExit("usage: set VESPA_RUN_DIR or pass a run dir as argv[1]")
    return RunDir(path)
