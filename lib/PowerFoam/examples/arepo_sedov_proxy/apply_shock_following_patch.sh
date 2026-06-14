#!/bin/bash
set -euo pipefail

AREPO_DIR="${1:-/Users/tabel/Projects/arepo}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$AREPO_DIR"
patch -p1 < "$PATCH_DIR/arepo_shock_following_mesh.patch"
