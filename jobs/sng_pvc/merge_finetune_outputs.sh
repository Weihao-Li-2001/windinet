#!/usr/bin/env bash
# Run from the repo root (where finetune_vae_outputs/ lives) on the sng_pvc login node.
#
# Usage:
#   ./jobs/sng_pvc/merge_finetune_outputs.sh            # dry run, just prints what it would do
#   ./jobs/sng_pvc/merge_finetune_outputs.sh --apply    # actually moves files
#
# For each dir under finetune_vae_outputs_sng_pvc/, looks for a same-named dir under:
#   1) finetune_vae_outputs/sng_pvc/archive/done/<name>
#   2) finetune_vae_outputs/sng_pvc/archive/known-bad/<name>
#   3) finetune_vae_outputs/sng_pvc/<name>                (direct)
# and merges into whichever matches (first match wins). Files that already exist at the
# destination are NOT overwritten -- they're left in place under finetune_vae_outputs_sng_pvc/
# and reported, so you can diff/resolve them by hand.
#
# One-shot helper: delete this file (and its jobs/sng_pvc/ entry) once the merge is done.

set -euo pipefail

SRC_ROOT="finetune_vae_outputs_sng_pvc"
DST_ROOT="finetune_vae_outputs/sng_pvc"
DONE_DIR="$DST_ROOT/archive/done"
BAD_DIR="$DST_ROOT/archive/known-bad"

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

if [[ ! -d "$SRC_ROOT" ]]; then
  echo "No $SRC_ROOT here -- run this from the repo root." >&2
  exit 1
fi

no_match=()
had_conflicts=()

for src in "$SRC_ROOT"/*/; do
  src="${src%/}"
  name="$(basename "$src")"

  if [[ -d "$DONE_DIR/$name" ]]; then
    dst="$DONE_DIR/$name"; label="done"
  elif [[ -d "$BAD_DIR/$name" ]]; then
    dst="$BAD_DIR/$name"; label="known-bad"
  elif [[ -d "$DST_ROOT/$name" ]]; then
    dst="$DST_ROOT/$name"; label="direct"
  else
    no_match+=("$name")
    continue
  fi

  echo "== $name -> $label ($dst) =="

  if [[ $APPLY -eq 0 ]]; then
    rsync -ain --remove-source-files "$src"/ "$dst"/ | sed 's/^/  /'
  else
    # -a: preserve attrs, --ignore-existing: never clobber files already at dst,
    # --remove-source-files: delete each file once copied (dirs cleaned up after).
    rsync -a --ignore-existing --remove-source-files "$src"/ "$dst"/
    find "$src" -type d -empty -delete

    # report anything left behind (i.e. name collisions rsync skipped)
    if [[ -d "$src" ]]; then
      had_conflicts+=("$name")
      echo "  !! conflicts left under $src (already existed at $dst, not overwritten):"
      find "$src" -type f | sed 's/^/     /'
    fi
  fi
done

if [[ $APPLY -eq 1 ]]; then
  git add -A -- "$SRC_ROOT" "$DST_ROOT"
fi

echo
if [[ ${#no_match[@]} -gt 0 ]]; then
  echo "No matching destination found for (decide manually -- done/known-bad/direct):"
  printf '  - %s\n' "${no_match[@]}"
fi
if [[ ${#had_conflicts[@]} -gt 0 ]]; then
  echo "Had file conflicts (left in place, review by hand):"
  printf '  - %s\n' "${had_conflicts[@]}"
fi
if [[ $APPLY -eq 0 ]]; then
  echo
  echo "(dry run only -- re-run with --apply to actually move files)"
fi
