#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export HF_TOKEN=your_hf_token
#   bash download_llama_models.sh
#
# Optional env vars:
#   PARALLEL=8          # concurrent downloads
#   SKIP_EXISTING=1     # skip files already present

# Auto-detect Hugging Face token if not provided
detect_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "$HF_TOKEN"
    return 0
  fi
  # Common env vars
  if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
    echo "$HUGGINGFACE_TOKEN"; return 0
  fi
  # File: ~/.huggingface/token (first line)
  if [[ -f "$HOME/.huggingface/token" ]]; then
    local t; t=$(head -n1 "$HOME/.huggingface/token" | tr -d '[:space:]')
    if [[ -n "$t" ]]; then echo "$t"; return 0; fi
  fi
  # Cache config (huggingface-cli whoami)
  if command -v huggingface-cli >/dev/null 2>&1; then
    local who; who=$(huggingface-cli whoami 2>/dev/null | grep -i 'Token' | awk -F': ' '{print $2}')
    if [[ -n "$who" ]]; then echo "$who"; return 0; fi
  fi
  return 1
}

HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" ]]; then
  if ! HF_TOKEN=$(detect_token); then
    echo "ERROR: HF token not found. Set HF_TOKEN env or login via huggingface-cli." >&2
    exit 1
  fi
fi

PARALLEL="${PARALLEL:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

DEST_ROOT="/HSC/users/qiaoye/checkpoints"
DEFAULT_MODELS=(
  "meta-llama/Llama-3.2-1B"
  "meta-llama/Llama-3.2-3B"
  "meta-llama/Llama-3.1-70B"
)

# Allow user to override model list via MODEL_LIST env (space separated)
if [[ -n "${MODEL_LIST:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=($MODEL_LIST)
else
  MODELS=(${DEFAULT_MODELS[@]})
fi

# Auto-skip already-downloaded full models when AUTO_SKIP_COMPLETE=1
AUTO_SKIP_COMPLETE=${AUTO_SKIP_COMPLETE:-1}

# Optional include / exclude glob patterns (space-separated) e.g.
#   INCLUDE_PATTERNS="*.safetensors tokenizer.* config.json" EXCLUDE_PATTERNS="original/*"
INCLUDE_PATTERNS=${INCLUDE_PATTERNS:-}
EXCLUDE_PATTERNS=${EXCLUDE_PATTERNS:-}

status_code() {
  local repo="${1:-}" file="${2:-}"
  if [[ -z "$repo" || -z "$file" ]]; then
    echo 000; return 0
  fi
  local url="https://huggingface.co/${repo}/resolve/main/${file}"
  curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" "$url" || echo 000
}

api_list_files () {
  local repo="$1"
  curl -sS --retry 5 --retry-delay 2 -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/models/${repo}" \
    | jq -r '.siblings[].rfilename'
}

download_file () {
  local repo="${1:-}"
  local dest_dir="${2:-}"
  local file="${3:-}"
  if [[ -z "$repo" || -z "$dest_dir" || -z "$file" ]]; then
    echo "WARN download_file missing arguments (repo='$repo' dest='$dest_dir' file='$file')" >&2
    return 0
  fi
  local url="https://huggingface.co/${repo}/resolve/main/${file}"
  local out="${dest_dir}/${file}"

  mkdir -p "$(dirname "$out")"

  if [[ "$SKIP_EXISTING" == "1" && -s "$out" ]]; then
    echo "SKIP $file"
    return 0
  fi

  wget \
    --header="Authorization: Bearer $HF_TOKEN" \
    --quiet \
    --show-progress \
    --retry-connrefused --waitretry=2 --tries=10 --timeout=30 \
    -c "$url" -O "$out" || {
      echo "FAILED $file" >&2
      return 1
    }
}

LOG_DIR="${DEST_ROOT}/download_logs"
mkdir -p "$LOG_DIR"
MANIFEST="${LOG_DIR}/manifest_$(date +%Y%m%d_%H%M%S).txt"
echo "# Download manifest $(date -Is)" > "$MANIFEST"
echo "# Token prefix: ${HF_TOKEN:0:6}" >> "$MANIFEST"

for repo in "${MODELS[@]}"; do
  echo "=== Downloading $repo ==="
  files=$(api_list_files "$repo")
  # Filter include patterns
  if [[ -n "$INCLUDE_PATTERNS" ]]; then
    keep=""
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      for pat in $INCLUDE_PATTERNS; do
        # Convert glob to regex safely
        regex="^${pat//./\\.}$"; regex="${regex//\*/.*}"
        if [[ "$line" =~ $regex ]]; then
          keep+="$line\n"; break
        fi
      done
    done < <(printf '%s\n' "$files")
    files=$(printf '%b' "$keep" | awk 'NF')
  fi
  # Exclude patterns
  if [[ -n "$EXCLUDE_PATTERNS" ]]; then
    for pat in $EXCLUDE_PATTERNS; do
      regex="^${pat//./\\.}$"; regex="${regex//\*/.*}"
      files=$(printf '%s\n' "$files" | grep -Ev "$regex" || true)
    done
  fi
  dest_dir="${DEST_ROOT}/$(basename "$repo")"
  mkdir -p "$dest_dir"

  if [[ "$AUTO_SKIP_COMPLETE" == "1" ]]; then
    # Heuristic completion check: presence of index json + at least one safetensors shard + tokenizer file
    if comp_index=$(ls "$dest_dir"/model.safetensors.index.json 2>/dev/null) \
       && shard=$(ls "$dest_dir"/model-*.safetensors 2>/dev/null | head -n1) \
       && tok=$(ls "$dest_dir"/tokenizer.{model,json} 2>/dev/null | head -n1); then
        echo "COMPLETE: $repo (found index + shard + tokenizer) -> skipping"
        echo "# SKIPPED COMPLETE $repo" >> "$MANIFEST"
        continue
    fi
  fi

  # Print rough size estimate (sum of sizes via API) if desired
  echo "$files" | wc -l | awk '{print "Files to fetch:", $1}'

  # Parallel download
  export -f download_file
  export HF_TOKEN SKIP_EXISTING
  # shellcheck disable=SC2086
  export -f status_code
  echo "$files" | while read -r f; do
    [[ -z "$f" ]] && continue
    code=$(status_code "$repo" "$f")
    case "$code" in
      403) echo "WARN 403 (Forbidden) $f - license or auth issue" >> "$LOG_DIR/preflight_${repo##*/}.log" ;;
      401) echo "ERROR 401 (Unauthorized) $f - invalid token" >> "$LOG_DIR/preflight_${repo##*/}.log" ;;
      404) echo "WARN 404 (Not Found) $f" >> "$LOG_DIR/preflight_${repo##*/}.log" ;;
    esac
  done

  parallel --jobs "$PARALLEL" download_file "$repo" "$dest_dir" ::: $files | tee -a "$LOG_DIR/parallel_${repo##*/}.log" || true
  echo "# ${repo} files:" >> "$MANIFEST"
  printf '%s\n' $files >> "$MANIFEST"
done

echo "All downloads complete."
echo "Manifest: $MANIFEST"

echo "If many files FAILED for a Llama repo:"
echo "  1) Accept the model license at:  https://huggingface.co/meta-llama"
echo "  2) Validate your token:          'huggingface-cli whoami' (after 'huggingface-cli login')"
echo "     Non-interactive check:       huggingface-cli whoami --token $HF_TOKEN"
echo "     Raw API check:               curl -s -H 'Authorization: Bearer $HF_TOKEN' https://huggingface.co/api/whoami-v2 | jq ."
echo "  3) If you still see 403, re-login: huggingface-cli logout && huggingface-cli login"
echo "  4) Re-run with selective patterns, e.g.:"
echo "     INCLUDE_PATTERNS='*.safetensors tokenizer.* config.json generation_config.json' EXCLUDE_PATTERNS='original/*' bash llama_download.sh"
echo "NOTE: The older suggestion 'huggingface-cli whoami -t' is invalid; use '--token' if you need to pass a token explicitly."