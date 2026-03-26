#!/usr/bin/env bash
# Materialize xray run outputs/checkpoints into this standalone repo.

set -euo pipefail

THIS_REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT_REPO="$(cd "$(dirname "$0")/../.." && pwd)"

SRC_OUTPUTS="${ROOT_REPO}/outputs"
DST_OUTPUTS="${THIS_REPO}/outputs"

mkdir -p "${DST_OUTPUTS}"

# Copy only likely xray run folders (edit if needed)
echo "[materialize_outputs] Copying xray outputs from ${SRC_OUTPUTS} -> ${DST_OUTPUTS}"

for d in "${SRC_OUTPUTS}"/**/*xray* "${SRC_OUTPUTS}"/full_xray_vlm_* "${SRC_OUTPUTS}"/smoke_xray_vlm_*; do
  if [[ -d "$d" ]]; then
    bn="$(basename "$d")"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --info=progress2 "${d}/" "${DST_OUTPUTS}/${bn}/"
    else
      echo "[materialize_outputs] rsync not found; copying with cp -a: ${bn}"
      mkdir -p "${DST_OUTPUTS}/${bn}"
      cp -a "${d}/." "${DST_OUTPUTS}/${bn}/"
    fi
  fi
done

echo "[materialize_outputs] Done."

