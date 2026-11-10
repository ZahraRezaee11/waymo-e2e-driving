#!/usr/bin/env bash
# Clone AutoVLA at the pinned commit, apply the patches, copy the scripts of this repo in.
# usage: bash scripts/setup_autovla.sh [target_dir]     (default: ./AutoVLA)
set -euo pipefail

COMMIT=ba34eed74ce6729e7986592d0e66cbaca397b4fa
URL=${AUTOVLA_URL:-https://github.com/ucla-mobility/AutoVLA}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST=${1:-AutoVLA}

[ -d "$DST/.git" ] || git clone "$URL" "$DST"
git -C "$DST" checkout --quiet "$COMMIT"

# 1) patches to upstream files
git -C "$DST" apply "$HERE/patches/0001-requirements-loosen-torch-pins.patch"
git -C "$DST" apply "$HERE/patches/0002-nocot-lazy-nuplan-import.patch"
# the LoRA trainer is the upstream SFT trainer plus a small diff
cp "$DST/tools/run_sft.py" "$DST/tools/run_sft_lora.py"
git -C "$DST" apply "$HERE/patches/0003-run-sft-lora.patch"

# 2) files of this repo, at the places the scripts expect
mkdir -p "$DST/ours_scripts"
cp "$HERE"/rfs/*.py "$HERE"/data/build_lmdb.py "$DST/ours_scripts/"
cp "$HERE"/third_party/waymo_open_dataset/rater_feedback_utils.py "$DST/ours_scripts/"
cp "$HERE"/data/make_eval_samples.py "$HERE"/eval/*.py "$HERE"/selector/*.py "$DST/"
cp "$HERE"/train/run_rft_waymo.py "$DST/tools/"
cp "$HERE"/configs/training/*.yaml "$DST/config/training/"
cp "$HERE"/configs/dataset/*.yaml "$DST/config/dataset/"

echo "AutoVLA is ready in $DST (commit $COMMIT, patches applied)."
