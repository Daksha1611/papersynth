# Design amendments

Revisions to the v0.1 design document, each forced by something a run
actually did. The master document is not in this repository, so these are
written as replacement prose for the sections they revise.

Every entry names what the design assumed, what happened instead, and what the
code now does. Where an amendment removes a claim rather than adding one, that
is the point: a design that promises a property the implementation does not
have is worse than one that admits the gap, because the promise is what a
reader relies on.

---

## §13.3 — Ablation harness

**Assumed:** a paper with a complete methods section exists, so deleting one
required field gives unambiguous ground truth in both directions - the deleted
field should be reported, and nothing else should be.

**Observed:** the first half holds. The second does not.

Gap recall is measured exactly as designed and works. Deleting a field and
checking that it is reported is self-labelling, needs no annotation, and cannot
drift. Pass A scores 1.00 on nine fields, deterministically.

Precision cannot be measured this way. The method requires a reference
specification complete enough that any gap raised against it is invented. Three
rounds against a reference corrected twice suggest no such specification
exists:

| round | reference size | gaps raised | inspected verdict |
|---|---|---|---|
| 1 | 16 fields | 11 | all genuine omissions |
| 2 | 32 fields | 3 | all genuine omissions |
| 3 | 35 fields | 4 | three genuine, one borderline |

Eighteen gaps examined across three rounds, none an invention, and the count
stopped decreasing. The reason is structural rather than a property of this
fixture: "what would you have to guess to implement this today" always has
correct answers, because a specification is complete only relative to a
question, and an implementer's questions are not bounded by the fields anyone
thought to write down.

**Amendment.** The harness reports two numbers and calls them what they are:
gap recall, and gaps raised against the reference. The second is not labelled a
false-positive rate, and convergence is flagged separately. Reporting zero
false positives would require treating "found real gaps" as "found no false
ones", which are different claims - and the honest reading of three rounds is
that precision is unmeasured, not that it is perfect.

Ablation of a `one_of` requirement removes the whole group. Deleting
`num_steps` while `num_epochs` remains proves nothing, because the requirement
is genuinely still satisfied; scoring that as a miss marked correct behaviour
as failure, and would have pushed the checklist the wrong way to "fix" it.

---

## §13.1 — Quality dimensions

**Was:** gap recall >= 0.80, with precision implied to be measurable alongside
it.

**Amendment.** Two rows rather than one.

| Dimension | Metric | Target |
|---|---|---|
| Gap recall | recall on ablated fields | >= 0.80 (Pass A: 1.00) |
| Gap precision | not measurable by ablation (§13.3) | guards justified by named failure modes; rate unquantified |

Pass B's precision rests on deterministic guards, each added after observing
the specific failure it prevents, rather than on a measured rate:

- a gap naming something the corpus already supplies is dropped
- a gap duplicating one Pass A found is merged, keeping Pass A's criticality
  and Pass B's phrasing
- a gap naming a *contested* value is dropped, because the papers answer it and
  disagree, and the spec surfaces that under `open_conflicts` already
- a gap proposing an answer has the suggestion stripped, and one that is
  *only* a suggestion is discarded entirely (ER-02)

Each of those exists because the pass produced that exact failure and it was
observed. That is weaker evidence than a rate and it is what there is.

---

## §8.6 — Gap check

**Amendment, Pass B.** Convergence of the gap list is not expected and is not a
success criterion. A specification is always incomplete relative to an
implementer's questions, so an adversarial audit will keep finding real
omissions however many are filled in. Reviewers should read the list as
prioritised by criticality rather than as a set that shrinks to empty, and a
non-empty list after several rounds is the normal state rather than a signal
that the audit is malfunctioning.

Consequently Pass B is opt-in (`--adversarial-gaps`) rather than default. Pass
A is deterministic, free, and cannot hallucinate; Pass B costs a call and
returns a list whose length has no natural stopping point.

---

## §8.1 — Math layer

**Was:** on detecting unreliable math, rasterize the region and re-recognize it
with nougat or pix2tex, marking the result `ocr_recovered`.

**Observed:** recovery needs a vision model and roughly two gigabytes of
dependencies. Gating the whole mitigation behind that left it inert on any
machine without them - which was every machine, so the confidence penalty, the
`symbol_check` corruption heuristic and the document fidelity flag were all
dead branches keyed to a value nothing produced. R-01 is the highest
likelihood-and-impact risk in the register and its mitigation had never fired.

**Amendment.** Detection and recovery are separate concerns with separate
requirements.

Detection needs nothing beyond the string and always runs: unbalanced
delimiters, glyphs the text layer could not map, and a density of characters
unusable in LaTeX. Mathematical symbols are explicitly not counted toward that
density, since real display math is symbol-dense and flagging it would make the
penalty meaningless by applying it everywhere.

Recovery is optional, behind the `math` extra.

This adds a fourth `source_fidelity` value, `text_layer_suspect`: the text layer
was detected as damaged and could not be re-recognized. It carries the same
confidence penalty as `ocr_recovered` but records honestly that no OCR ran,
which matters because the two need different follow-up - one wants the OCR
output checked, the other wants the extra installed or LaTeX source found.

---

## §6.4.2 — Provider chain

**Status.** Two legs verified in production, the third by construction.

`groq -> gemini` fell through on a real `RateLimitError` during extraction, in
chain order. The `openrouter` leg has never run live, because an unset key
narrows the chain rather than failing it, so its behaviour is covered by tests
that fake exhaustion rather than by observation. Its transport is the same
OpenAI-compatible client verified live against Groq; what remains untested is
OpenRouter's own responses.

**Amendment.** Two behaviours the section described but did not require, both
now implemented after their absence was noticed only when quotas actually ran
out:

A provider skipped because it is already known exhausted is recorded in the
ledger. §6.4.3 promises the ledger explains which provider served each call,
and a silent skip made a call on the second leg look unexplained.

`AllProvidersExhausted` names `--resume`, which now exists. The message
previously instructed users to run a flag that had never been implemented, and
an instruction that fails when followed is worse than a missing feature.
Resumption is per paper rather than per stage: an interrupted run has some
papers fully extracted and others untouched, and extraction is where the calls
go.


---

## §4.2 — Per-paper parallelism

**Was:** stages 0-2 run per paper and are parallelizable, with
`max_parallel_papers` defaulting to 3 to meet the fifteen-minute target in
NFR-03.

**Observed:** the design assumed latency was the binding constraint. On a
hosted free tier it is tokens per minute. Groq allows 8,000 and one extraction
prompt is roughly 3,000, so three concurrent papers put 9,000 in flight and
guarantee the rate limit that concurrency was meant to avoid waiting for.
Concurrency does not raise the ceiling; it reaches it faster.

**Amendment.** Parallelism is implemented and defaults to 1. It pays on the
local vLLM path, which has no per-minute cap, and the setting says so.

Results are merged by input position rather than completion order. Merging as
papers finish would make the claim set - and therefore cluster IDs,
contradiction IDs and the emitted spec - depend on which paper happened to
return first, and identical inputs would stop producing identical specs
(NFR-02). A test asserts a three-way parallel run and a sequential run emit
byte-identical specs.

---

## §8.3.4 — Self-consistency

**Amendment, on how the re-extractions differ.** The design says "different
prompt orderings" without saying why, and the reason turns out to be load
bearing. Extraction runs at temperature 0 and responses are cached by prompt
hash, so re-issuing an identical prompt returns the identical answer - free,
instant, and worth nothing as a second opinion. The variation has to come from
the prompt itself, so passes rotate the section order.

Agreement is scored on what counts as the same fact, not on `claim_id`. The
identifier folds in the span, and a model quoting a different sentence for the
same value on a second pass still reported the same fact; scoring that as
disagreement would penalise exactly the claims that were found twice.

A claim that disagrees is not dropped. It is reported with its agreement
recorded and its confidence scaled, and the confidence threshold then decides
whether it may be promoted to `verified`. Below the threshold it stays
`extracted`: not rejected, because the checks passed and it may well be right,
but excluded from alignment and from auto-resolution, because a claim its own
re-extractions disagreed on should not settle a conflict.


---

## §10.3 — Detector registration

**Was:** detectors self-register and are run over every multi-paper cluster.

**Observed:** nothing constrained a detector to the claim type it understands.
ValueConflictDetector was written when hyperparameters were the only type
carrying a `value`, so adding `result` silently activated it on benchmark
scores. It groups by condition and unit, neither of which a result payload has,
so every score landed in one group regardless of dataset, split or model
variant - and a dev score was reported as contradicting a test score. That is
exactly the ER-06 violation RESULT_CONFLICT exists to prevent, arriving through
a different detector.

**Amendment.** A detector declares the claim type it scans and the scan loop
enforces it. The failure was not that the value detector was wrong; it was that
nothing said what it was for, so a later claim type could quietly enter its
scope. Adding the declaration makes the next claim type safe by default rather
than by whoever remembers.

---

## §7.6 — RESULT_CONFLICT severity

**Amendment.** Result conflicts are always MATERIAL, never BLOCKING.

The severity ladder defines BLOCKING as "cannot write correct code without
resolving". A disagreement about a reported score never meets that: an
implementer can write the code either way. What they cannot do is tell whether
the finished code is right, because the reproduction target is in dispute.
That is a validation problem rather than an implementation decision, so it
rides along in `open_conflicts` where the implementer sees both numbers and
picks a target.

One refinement the design does not mention but the payload makes possible: two
results whose reported variances overlap are not treated as conflicting.
Agreement within stated uncertainty is agreement, and reporting it would ask a
reviewer to adjudicate noise. Where no paper states a variance, a difference is
reported, because there is then no basis for calling it noise.

---

## M8 corrections (items 1-4 and 6)

The first out-of-domain corpus produced four correctness defects and one
noise defect. Details in `docs/m8-findings.md`; the fixes:

### §10.1 internal_consistency, and scope on every claim

Implemented. Every claim now carries `provenance.scope_id` - the section it
came from. `SpecBuilder` splits verified hyperparameters into recognized
training settings (which collapse to `cmp_global` as before) and unrecognized
quantities, and the latter, when two or more share one section of one paper,
are emitted under a component named for that section with each value scoped to
it. The `internal_consistency` hook detects the pattern in the verify stage
and records it.

This is the M8 failure fixed at the source. Kunzel's field experiment reports
68,378 voters, 1,295 households, 913 treated, 501 final - a funnel. Flattened
into independent config those numbers contradict, and the coding agent said
so. Scoped to their section they read as what they are.

### §7.4 components carry design decisions

`_components` filtered `claim.type != "hyperparameter"`, written before the
method and result types. 27 of 39 verified M8 claims never reached the
deliverable. Components now carry a `design_decisions[]` array populated from
method claims, grouped by `applies_to`; result claims already fed
`expected_results`.

### NFR-09 at batch granularity

`registry.run_all` caught per extractor. One malformed response on batch 1
discarded 19 valid batches for equation and algorithm on a paper with 78
equations. The batch loop now catches per batch: a failed batch is a warning
and zero claims for its sections, and the rest run. `run_all` records
per-extractor coverage - sections read against total, batches ok against total
- carried into `coverage.yaml`, the CLI summary and SPEC_REVIEW.md. A thorough
run and a 1%-read run no longer look identical.

### §8.1 semantic section triage

Chosen: per-extractor, gated by a regex pre-filter. The static regex list was
BERT-shaped and matched almost nothing on the M8 papers, and the fallback only
fires when nothing matches, so a partial match left them 96-99% unread in
silence. `applicable_sections` now runs the regex as a pre-filter; when it
covers under 60% of the paper and a provider is available, one classification
call per extractor per paper decides relevance from the actual section titles.
Per-extractor rather than shared because "relevant to hyperparameter
extraction" and "relevant to method extraction" do not carve a paper the same
way; merge to a shared map only if the cost becomes real at scale. Skipped
entirely for papers under six sections. Cost: one call per extractor per paper
when triggered, ~35 in the M8 corpus against ~46 for the whole run.

### §13.2 checklist gate

Pass A's ML-training section gated on `any_hyperparameter`, which a field
experiment's sample counts satisfy. 9 of 13 M8 gaps were "no learning rate /
optimizer / dropout" on an agent-architecture corpus. The gate now requires a
recognized training hyperparameter, not any hyperparameter.

## §8.4 — Cross-paper alignment (M8 item 5)

The last M8 finding, and the one that mattered most: zero of 37 clusters
spanned more than one paper. Cross-paper contradiction detection is the
product, and a run where nothing aligns reports "0 contradictions" for the
same reason an empty run does. Nothing in the artifacts distinguishes the two.

**Cause.** The blocking key is a string. For hyperparameters that is a strong
and near-exact signal - `learning_rate` is `learning_rate` everywhere. For
method claims, which align on `sub_problem`, it requires two papers to
independently invent the same snake_case name for one question. CaMeL produced
`security_mechanism`, `data_flow_security`, `capability_tagging`; NeMo produced
`rail_specification_language`, `canonical_form_definition`. Same question, no
shared key. 27 of 39 verified M8 claims were method claims, so this ate the
corpus.

**Embeddings were the designed fallback and cannot do it.** On M8 the best
cross-paper pair scored 0.401 against a 0.82 threshold and the next best 0.107.
On BERT/RoBERTa/ALBERT surface similarity proposed five merges and all five
were wrong - `num_steps` with `warmup_steps`,
`next_sentence_positive_ratio` with `next_sentence_negative_ratio` - which
fabricated two of the three contradictions that run reported. Hyperparameter
names are composed from shared words, so surface similarity tracks naming
convention rather than meaning: too blunt to find the real merges and sharp
enough to invent false ones. No threshold separates those two behaviours, so
the embedding merge path is removed rather than defaulted off, along with
`align_threshold` and `embedding_model`, which no longer decide anything.

**Chosen: semantic merge proposal, split-gated.** Same shape as the §8.1
triage fix - cheap exact key as the primary, one model call only where it
fails, existing adversary on the output.

- Exact-name blocking is unchanged and still primary.
- `align/semantic.py` then takes the keys blocking left inside a single paper -
  a key already spanning papers has found its match - and asks, in one call per
  claim type, which of them name the same question. Keys already matched are
  never offered, so the call scales with the failure rather than the corpus.
- The answer is validated as a proposal about indices we supplied, not trusted:
  a group survives only if it names two or more distinct keys across two or
  more papers, no key may be claimed twice, and the existing unit and
  value-type guards still veto. Single-paper groups are rejected outright -
  collapsing quantities one paper stated separately is the M8 funnel failure
  from item 1.
- Every cluster a merge creates goes to the SplitterAgent regardless of the
  `--split-gate` flag. This is the gate's designed job and what it was waiting
  for; until now it had nothing to review.

**The split gate's scope narrowed to match.** It previously reviewed every
multi-paper cluster or none. It now always reviews semantically merged
clusters and reviews exact-name clusters only under `--split-gate`, because
reviewing those measured badly: on BERT/RoBERTa/ALBERT it split `batch_size`
into three concepts and lost the genuine 256-against-8000 disagreement, the
corpus's headline finding, while the splits it got right removed no false
contradiction that condition grouping had not already prevented.

**A latent bug this uncovered.** `splitter._merge_identical` re-merges groups
holding the same value in the same unit, to undo the splitter separating three
identical `max_sequence_length` values of 512. Method claims carry no value, so
every method group keyed on the same empty value and the re-merge silently
undid every split - on exactly the clusters semantic alignment creates, where
the gate is the only review there is. It now applies only to claims that carry
a value.

**Manifest.** `align_threshold` left the reproducibility fingerprint with the
code it configured. The per-run switches - `semantic_merges`,
`split_all_clusters`, `entailment`, `adversarial_gaps` - live on the Pipeline
rather than in Settings, so the fingerprint alone reported defaults the run
may not have used. The manifest now merges both, which is what R-06 asks of it.

R019 pins the finding, the fix, the reversibility of a merge, and that a model
proposing nothing leaves exact-name blocking alone.
