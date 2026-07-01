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
export CIC_FVGK_STORE="${CIC_FVGK_STORE:-f16}"
export CIC_FVGK_DEDUP="${CIC_FVGK_DEDUP:-1}"
export CIC_CHEM=analytic
export CIC_PACKED=1
export CIC_PSORT="${CIC_PSORT:-16}"
export CIC_PSORT_BUCKET="${CIC_PSORT_BUCKET:-256}"
export CIC_PIDS="${CIC_PIDS:-0}"
export CIC_VEL16="${CIC_VEL16:-1}"
export CIC_GRAVITY="${CIC_GRAVITY:-gpu}"
export CIC_GRAV_HOST32="${CIC_GRAV_HOST32:-1}"
export CIC_FFT="${CIC_FFT:-ka}"
export CIC_OVERLAP="${CIC_OVERLAP:-0}"
export CIC_GRAV1BUF="${CIC_GRAV1BUF:-1}"
export CIC_PK="${CIC_PK:-1}"
export CIC_CELL_DUMP="${CIC_CELL_DUMP:-0}"
export CIC_NODUMP="${CIC_NODUMP:-1}"
export CIC_NGRID="${CIC_NGRID:-1024}"
export CIC_NP="${CIC_NP:-1}"
export CIC_BOX="${CIC_BOX:-0.8}"
export CICASS_REAL_BYTES="${CICASS_REAL_BYTES:-4}"
export CIC_TAG="${CIC_TAG:-hero_metal1024_fvgk_f16_packed_pk}"
export CIC_SNAP="${CIC_SNAP:-$ROOT/reports/multicode/hero_ics/metal_1024_box0p8/cic_stream_1024_box0p8.cicass}"
export CIC_STREAM_LOAD=1
export CIC_CKPT_PREFIX="${CIC_CKPT_PREFIX:-$ROOT/reports/multicode/perf/$CIC_TAG}"

exec "$JULIA_BIN" --project="$ROOT/lib/MultiCode/test" "$ROOT/lib/MultiCode/examples/patch_cicass.jl"
