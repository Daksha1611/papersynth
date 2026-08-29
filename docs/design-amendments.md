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
