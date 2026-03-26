#!/usr/bin/env bash
# Materialize xray data + HF cache into this standalone repo.
# This copies from the current working repo (GraphAttributeLearning) into this folder.

set -euo pipefail

THIS_REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT_REPO="$(cd "$(dirname "$0")/../.." && pwd)"

SRC_RAW="${ROOT_REPO}/data/raw/raw"
SRC_PROCESSED="${ROOT_REPO}/data/processed/xray"
SRC_HF_CACHE="${ROOT_REPO}/data/hf_cache"

DST_RAW="${THIS_REPO}/data/raw/raw"
DST_PROCESSED="${THIS_REPO}/data/processed/xray"
DST_HF_CACHE="${THIS_REPO}/data/hf_cache"

echo "[materialize_data] THIS_REPO=${THIS_REPO}"
echo "[materialize_data] ROOT_REPO=${ROOT_REPO}"
echo "[materialize_data] Copying raw: ${SRC_RAW} -> ${DST_RAW}"
echo "[materialize_data] Copying processed: ${SRC_PROCESSED} -> ${DST_PROCESSED}"
echo "[materialize_data] Copying hf_cache: ${SRC_HF_CACHE} -> ${DST_HF_CACHE}"

mkdir -p "${THIS_REPO}/data/raw" "${THIS_REPO}/data/processed" "${THIS_REPO}/data"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --info=progress2 "${SRC_RAW}/" "${DST_RAW}/"
  rsync -a --info=progress2 "${SRC_PROCESSED}/" "${DST_PROCESSED}/"
  rsync -a --info=progress2 "${SRC_HF_CACHE}/" "${DST_HF_CACHE}/"
else
  echo "[materialize_data] rsync not found; falling back to cp -a"
  mkdir -p "${DST_RAW}" "${DST_PROCESSED}" "${DST_HF_CACHE}"
  cp -a "${SRC_RAW}/." "${DST_RAW}/"
  cp -a "${SRC_PROCESSED}/." "${DST_PROCESSED}/"
  cp -a "${SRC_HF_CACHE}/." "${DST_HF_CACHE}/"
fi

echo "[materialize_data] Done."

