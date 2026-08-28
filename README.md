# PaperSynth

**Multi-paper implementation spec synthesizer.** Feed it a set of related papers; get back a single reconciled, citation-traced, contradiction-annotated implementation specification that a human approves before any code is written.

> PaperCoder converts *one* paper into *code*.
> PaperSynth converts *many* papers into *one verified spec* that a human approves before any code exists.

---

## Why

Implementing a method from the literature fails in four recurring ways:

| Failure mode | Consequence |
|---|---|
| Equations and algorithms degrade during PDF extraction | Silent numerical errors |
| Papers omit implementation details | The coding agent quietly guesses |
| Related papers prescribe **conflicting** methods | The implementer picks arbitrarily, or blends incompatibly |
| No traceability from code back to a paper section | Reproduction failures are impossible to audit |

PaperSynth addresses all four by making provenance mandatory, contradictions first-class, and the human review gate non-optional.

**It deliberately stops before code generation.** A wrong hyperparameter caught in a 200-line YAML costs minutes; the same error caught after a failed 12-hour training run costs a day and gets misattributed to a bug elsewhere.

## Pipeline

```
ingest → extract → verify → align → contradict → reconcile → gapcheck → synthesize
   │        │         │        │          │           │           │          │
 per-paper ─┴─────────┘        └───────── corpus-wide ┴───────────┴──────────┘
```

Stages 0–2 run per paper and are parallelizable. Stages 3–6 reason across the corpus. Stage 7 emits `implementation_spec.yaml` — the deliverable.

Every stage writes a human-readable artifact to `runs/{run_id}/`, so any decision can be inspected after the fact.

## Design commitments

These are the non-negotiable ones. The rest is implementation detail.

- **Every claim carries provenance.** A claim whose payload cannot be located in its cited span is *rejected*, not downgraded. A spec field with no traceable source blocks emission.
- **Extractors never see another paper.** If extraction were corpus-aware, an LLM holding Paper A's learning rate in context while extracting Paper B's would drift toward agreement — genuine contradictions would vanish before the detector ever ran.
- **The reconciliation fallback is always ESCALATED.** No policy rule fires unambiguously? A human decides. There is no "best guess" path. A tool that resolves 100% of conflicts is indistinguishable from one that resolves 60% correctly and 40% arbitrarily, unless it tells you which is which.
- **Values absent from the paper are never invented.** Absent means a `Gap` gets raised, not a plausible default filled in.
- **IDs are content-derived.** Re-ingesting the same PDF produces the same span IDs, so a spec emitted today still diffs cleanly against one from last month.

## Status

Pre-alpha, under active construction. Building the minimal viable architecture first, in dependency order:

- [x] Schemas, contracts, and validator
- [x] Span addressing with lossless round-trip
- [x] Ingest (LaTeX + GROBID/PDF + arXiv)
- [x] LLM layer with free-tier fallback chain
- [x] Hyperparameter extractor
- [x] Citation trace + range check
- [x] Align + VALUE_CONFLICT detection
- [x] Policy engine with ESCALATED fallback
- [x] Spec builder + provenance closure gate
- [x] CLI wiring for the full pipeline
- [x] Gap check (`missing_but_critical`) — static checklist pass
- [x] Remaining extractors: equation, algorithm
- [x] SplitterAgent (opt-in; see below)
- [x] Symbol closure check

**The MVA acceptance criterion now passes.** Given three papers with one hand-planted conflicting hyperparameter, the pipeline emits a spec that (a) contains that conflict in `open_conflicts` with both positions and full provenance, and (b) contains no fabricated conflicts — the two values stated under *different* conditions are correctly left alone.

See `tests/e2e/test_mva_acceptance.py`. The end-to-end run is deterministic and offline: extraction responses are scripted, so the suite costs nothing and cannot drift.

## On the SplitterAgent

The split gate (`--split-gate`) reviews clusters that span papers and rejects merges
of things that are only superficially similar. It works, and it is **off by default**
on the measurement rather than on principle.

On BERT/RoBERTa/ALBERT it correctly separated `num_steps` from `warmup_steps` and
`hidden_dim` into three model variants — but it also split `batch_size` into three
concepts, losing the genuine BERT-256 vs RoBERTa-8000 disagreement, which is that
corpus's headline finding. The splits it got right removed no false contradiction,
because condition-and-unit grouping had already kept those apart. One real finding
lost, none gained.

Enable it where its designed job actually arises — alongside `--embedding-merges`,
where it rejected all five bad merges that feature proposes:

```bash
papersynth run ... --embedding-merges   # implies --split-gate
```

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/Daksha1611/papersynth.git
cd papersynth

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
papersynth --version
```

The PDF path additionally needs GROBID:

```bash
docker run --rm -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.1
curl -s http://localhost:8070/api/isalive   # expect: true
```

## Development

```bash
pytest                        # unit + e2e + regressions, no network
ruff check . && ruff format --check .
mypy papersynth
```

### The review flow

```bash
papersynth run --papers a.tex,b.tex,c.tex --objective "..." --out runs/demo
papersynth conflicts runs/demo --status open      # exits 2 while any remain
papersynth resolve   runs/demo ctr_ce42684b --select clm_11f315 --note "why"
papersynth gaps      runs/demo
papersynth spec      runs/demo --format yaml
papersynth approve   runs/demo --reviewer you
```

`resolve` re-emits the spec from stage artifacts without re-extracting anything, so a
human decision costs no model calls.

Tests never hit a live model. LLM interactions are recorded as cassettes and replayed, so CI is deterministic and free. A nightly job re-records against the real model and diffs the resulting specs — a diff signals prompt or model drift.

## Schemas

`papersynth/schemas/*.json` are the public contract, versioned semantically and readable without touching Python. `spec.schema.json` in particular is treated as this system's API: downstream agents pin against it, so it is additive-only within a minor version.

## License

MIT. See [LICENSE](LICENSE).
