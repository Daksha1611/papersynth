# M8: first handoff to a coding agent

Five papers, four on agent safety and one on causal inference, against the
objective "a bounded agent architecture separating LLM proposal from
deterministic policy enforcement, with an adversarial evaluation harness and
per-cause treatment-effect measurement."

The corpus was chosen to be unlike the BERT-family fixtures the pipeline was
tuned on. It found what that was meant to find.

Run: `runs/m8`. 46 calls, $0.00, 64 claims, 39 verified, 0 contradictions,
13 gaps, spec emitted and approved.

## What worked

Ingestion. All five resolved to `latex_native` with no OCR fallback needed;
span round-trip and provenance closure held at 100%.

Pass B of the gap check. Its four gaps - deterministic policy rules, treatment
effect estimator, adversarial harness protocol, LLM proposal prompt template -
are precisely the four things the objective needs and no paper supplied. The
coding agent independently blocked on all four.

Rejection. 39% of claims failed citation_trace, which is high but correct:
these papers are prose-dense and the extractor quoted loosely.

## Root finding: the pipeline read 1-21% of each paper

`applicable_sections` narrows by section-title regex, with a fallback to the
whole paper when nothing matches. On these papers 1-8 titles match, so the
fallback does not fire and the rest is skipped silently.

| paper | chars | hyperparam | method | result |
|---|---|---|---|---|
| Kunzel | 97,568 | 7% | 5% | 21% |
| ToolEmu | 252,125 | 9% | 7% | 5% |
| NeMo | 46,202 | 4% | 8% | 2% |
| AgentDojo | 44,027 | 1% | 7% | 11% |
| CaMeL | 321,684 | 1% | 4% | 2% |

Partial matching is worse than no matching, and nothing warns. A run that read
1% of a paper is indistinguishable from a thorough one in every artifact the
system produces.

## Cross-paper alignment did not happen at all

Zero of 37 clusters spanned more than one paper, so no detector could fire.

Method claims align on `sub_problem`, which requires two papers to
independently choose the same snake_case name for the same question. CaMeL
produced `security_mechanism`, `data_flow_security`, `capability_tagging`;
NeMo produced `rail_specification_language`, `canonical_form_definition`. Same
question, no shared key.

Embedding merges would not have rescued it: the best A-B pair scores 0.401
against a 0.82 threshold, and the best C-D pair 0.107. The relationship is
semantic and no surface metric reaches it. This is not a threshold to tune.

## The pipeline manufactured a contradiction, then hid it

Seven of ten verified hyperparameters come from one section of Kunzel:
"Reducing transphobia: A field experiment", the paper's empirical application.
`registered_voters = 68378`, `randomization_households = 1295`,
`treatment_group_size = 913`, `final_sample_size = 501` are stages of one
study design, coherent in context.

Extraction stripped that context, the builder emitted them as flat
configuration, and the coding agent correctly reported them as mutually
inconsistent - 913 treated out of a 501 final sample. The inconsistency is an
artifact of the pipeline, not of the paper.

Section 10.1 specifies an `internal_consistency` hook for exactly this. It is
not implemented.

## Method claims never reach the spec

27 of 39 verified claims were `method` type. `SpecBuilder._components` filters
`claim.type != "hyperparameter"`; it predates the method and result types. The
richest extraction output is invisible in the deliverable, which is why the
handed-off spec contained nine "Configuration for X" wrappers around single
numbers and no design decisions at all.

## Two extractors lost 38 batches to one bad response each

`equation` and `algorithm` produced zero claims despite Kunzel having 78
equations and 9 algorithms. Each failed on its first batch with malformed JSON
and `registry.run_all` catches per extractor rather than per batch, so 19
remaining batches were abandoned in each case.

## Pass A fired nine irrelevant gaps

The implementability checklist is ML-training-specific and gated on
`any_hyperparameter`. Kunzel's numbers tripped that gate, so the spec asked an
agent-architecture corpus for a learning rate, optimizer, batch size, dropout,
warmup and weight initialization. Nine of thirteen gaps were noise.

This inverts the BERT result, where Pass A was clean and Pass B needed guards.
