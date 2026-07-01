#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-/Users/tabel/.local/bin/julia}"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
export JULIA_PKG_PRECOMPILE_AUTO=0
export TMPDIR="${TMPDIR:-$ROOT/reports/multicode/tmp}"
mkdir -p "$TMPDIR"

export BACKEND=metal
export CIC_SOLVER=fvgk
export CIC_FVGK_F16=1
export CIC_CHEM=analytic
export CIC_PACKED=1
export CIC_PSORT=0
export CIC_PK=1
export CIC_NGRID=1024
export CIC_NP=1
export CIC_BOX=0.8
export CIC_TAG="${CIC_TAG:-hero_metal1024_fvgk_f16_packed_pk}"
export CIC_SNAP="${CIC_SNAP:-$ROOT/reports/multicode/hero_ics/metal_1024_box0p8/cic_stream_1024_box0p8.cicass}"
export CIC_STREAM_LOAD=1
export CIC_CKPT_PREFIX="${CIC_CKPT_PREFIX:-$ROOT/reports/multicode/perf/$CIC_TAG}"

exec "$JULIA_BIN" --project="$ROOT/lib/MultiCode/test" "$ROOT/lib/MultiCode/examples/patch_cicass.jl"
