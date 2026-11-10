#!/usr/bin/env bash
# Data-boundary scan (the repository's data boundary). Refuses weights and archives, corpora (*.jsonl), private directories,
# files over 1 MiB and credential patterns.
#   --tree REF   scan every file in a commit (default HEAD)
#   --staged     scan the index (pre-commit)
set -euo pipefail
mode=${1:---tree}
ref=${2:-HEAD}
max=$((1024 * 1024))
cd "$(git rev-parse --show-toplevel)"

case "$mode" in
  --tree)   files=$(git ls-tree -r --name-only "$ref"); size_of() { git cat-file -s "$ref:$1"; }; body_of() { git show "$ref:$1"; } ;;
  --staged) files=$(git diff --cached --name-only --diff-filter=ACMR); size_of() { git cat-file -s ":$1"; }; body_of() { git show ":$1"; } ;;
  *) echo "usage: $0 --tree [REF] | --staged" >&2; exit 2 ;;
esac

problems=0
flag() { echo "REFUSED: $1: $2" >&2; problems=$((problems + 1)); }
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  case "$f" in
    *.safetensors|*.bin|*.gguf|*.pt|*.pth|*.whl|*.tar|*.tar.gz|*.tgz|*.zip|*.so|*.npy) flag "$f" "weight/binary/archive extension" ;;
    *.jsonl) flag "$f" "jsonl corpus or generated text" ;;
    generations/*|captures/*|private/*|data/*|logs/*|*/generations/*|*/captures/*|*/private/*) flag "$f" "private directory" ;;
  esac
  size=$(size_of "$f")
  (( size > max )) && flag "$f" "size $size > $max"
  if body_of "$f" | grep -I -q -E '(hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16})'; then
    flag "$f" "credential pattern"
  fi
done <<< "$files"

if (( problems )); then
  echo "data-boundary: $problems problem(s)" >&2
  exit 1
fi
echo "data-boundary: OK ($(wc -l <<< "$files") files, $mode $ref)"
