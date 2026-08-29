#!/usr/bin/env bash
# Local-model path (PaperCoder convention, section 15.2).
#
# Runs entirely against a local OpenAI-compatible endpoint - Ollama or vLLM.
# No key, no quota, no per-minute ceiling, which is what makes an ablation
# sweep or a repeated debugging run practical. Extraction quality is lower
# than the hosted free tiers, so expect more rejected claims.
set -euo pipefail

export PAPERSYNTH_PROVIDER_CHAIN="vllm"
export PAPERSYNTH_VLLM_URL="${PAPERSYNTH_VLLM_URL:-http://localhost:11434/v1}"
export PAPERSYNTH_VLLM_MODEL="${PAPERSYNTH_VLLM_MODEL:-qwen2.5:14b}"

curl -sf "${PAPERSYNTH_VLLM_URL}/models" >/dev/null || {
  echo "No local model server at ${PAPERSYNTH_VLLM_URL}." >&2
  echo "Start one with:  ollama serve" >&2
  echo "Then pull a model:  ollama pull ${PAPERSYNTH_VLLM_MODEL}" >&2
  exit 1
}

curl -sf "${PAPERSYNTH_VLLM_URL}/models" | grep -q "${PAPERSYNTH_VLLM_MODEL%%:*}" || {
  echo "${PAPERSYNTH_VLLM_MODEL} is not pulled. Run:" >&2
  echo "  ollama pull ${PAPERSYNTH_VLLM_MODEL}" >&2
  exit 1
}

exec "$(dirname "$0")/run.sh" "$@"
