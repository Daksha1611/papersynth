#!/usr/bin/env bash
# Hosted-API path (PaperCoder convention, section 15.2).
#
# Ingests a paper set, runs the pipeline, and emits the spec only when nothing
# blocks. Exits 2 when a human has to decide something first, which is a
# correct outcome rather than a failure - the whole design gates on it.
set -euo pipefail

PAPERS="${PAPERS:-1810.04805,1907.11692,1909.11942}"
OBJECTIVE="${OBJECTIVE:-Implement BERT-style masked language model pretraining}"
OUT="${OUT:-runs/example}"
EXTRACTORS="${EXTRACTORS:-hyperparameter,method}"

papersynth doctor || {
  echo "Preflight failed. Fix the above before running." >&2
  exit 1
}

papersynth run \
  --papers "$PAPERS" \
  --objective "$OBJECTIVE" \
  --extractors "$EXTRACTORS" \
  --out "$OUT" || RUN_STATUS=$?

# Exit 2 from `run` means blocking conflicts, not breakage.
if [ "${RUN_STATUS:-0}" -eq 2 ]; then
  echo
  echo "Blocking conflicts remain. Review them, then re-emit:"
  echo "  papersynth conflicts $OUT --status open --severity BLOCKING"
  echo "  papersynth resolve   $OUT <ctr_id> --select <clm_id> --note '...'"
  echo "  papersynth approve   $OUT --reviewer \"\$USER\""
  exit 2
elif [ "${RUN_STATUS:-0}" -ne 0 ]; then
  exit "${RUN_STATUS}"
fi

papersynth gaps "$OUT"
echo
echo "Spec: $OUT/implementation_spec.yaml"
echo "Review: $OUT/SPEC_REVIEW.md"
echo "Approve with: papersynth approve $OUT --reviewer \"\$USER\""
