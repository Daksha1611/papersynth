You are the engineer who has to implement this specification TODAY. You have no
access to the papers it was built from, and no way to ask their authors.

Below is everything the specification tells you.

## The specification

{spec}

## Your task

List every question you would be forced to GUESS an answer to before this could
run. Not questions you would like answered - questions that stop you.

Return JSON:

```
{"gaps": [{"field": "<snake_case_name>", "question": "<what you must know>",
           "criticality": "BLOCKING" | "MATERIAL" | "COSMETIC",
           "blocks": "<what you cannot write without it>"}]}
```

`criticality` means:

- `BLOCKING`  there is no defensible default; any guess is a coin flip
- `MATERIAL`  a conventional default exists but changes the results
- `COSMETIC`  affects convenience or presentation only

## Rules

DO check the specification above before asking. If a value is already there,
it is not a gap, and listing it wastes the reviewer's attention on a question
already answered.

DO name `field` in snake_case for the thing that is missing, e.g.
`weight_initialization`, `gradient_accumulation_steps`. This is matched against
gaps already found, so a consistent name prevents duplicates.

DO ask about what is needed to run, not what is needed to match the paper's
reported numbers exactly. Both matter, but the first blocks and the second does
not.

DO think about what a specification usually omits: initialization, padding and
masking conventions, tokenizer vocabulary, gradient clipping, mixed precision,
data ordering and shuffling, checkpoint selection, evaluation protocol.

DON'T list a question you could answer from the specification by reading
carefully.

DON'T list anything the specification already shows as UNRESOLVED. Those values
are contested, not missing, and the reviewer is already deciding them. Saying
they are absent is simply wrong.

DON'T list framework, harness or infrastructure choices. Dataloader worker
counts, checkpoint intervals, logging frequency and which library to use are
your decisions, not gaps in the paper. Ask only about what the paper should
have told you.

DON'T pad the list. A short list of genuine blockers is far more useful than a
long list where the real ones are buried - a reviewer who stops reading this
list gets nothing from it at all.

If the specification is genuinely sufficient to implement, return
`{"gaps": []}`.
