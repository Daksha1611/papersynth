Two or more papers appear to describe the SAME configurable quantity. Your job
is to decide whether they really do.

You are adversarial. Look for reasons these are DIFFERENT quantities that merely
share a name. Agreeing is not your job; finding the distinction is.

## The candidate group

Name assigned by alignment: `{canonical_name}`

{claims}

## Output

Return JSON:

```
{"assignments": [{"claim_id": "clm_xxxxxx", "concept": "<label>"}, ...],
 "reason": "<one sentence>"}
```

Give every claim a `concept` label. Claims sharing a label are the same
quantity. Use one label for all of them only if they genuinely are the same.

## What makes them DIFFERENT

DIFFERENT MODEL VARIANT. A hidden size of 4096 and one of 768 are not a
disagreement if one describes a large variant and the other a base variant.
Label them separately, e.g. `hidden_dim_large` and `hidden_dim_base`.

DIFFERENT QUANTITY, SIMILAR NAME. Total training steps and warmup steps are
unrelated numbers that both end in "steps". Peak learning rate and final
learning rate are different quantities.

DIFFERENT UNIT OR BASIS. A batch measured in sequences and one measured in
tokens are the same setting expressed two ways, not two settings - but they are
not comparable as numbers, so label them separately.

DIFFERENT STAGE. A value used during pretraining and one used during
fine-tuning are different settings even under one name.

DIFFERENT SYMBOL MEANING. The same Greek letter often denotes unrelated things
in different papers.

## What makes them the SAME

They configure the same thing, at the same stage, for a comparable model, in
the same unit - even when the values disagree. A genuine disagreement about one
quantity is exactly what should stay merged, because that disagreement is the
finding.

DON'T split just because the values differ. Differing values are the point. A
batch size of 256 and one of 8,000 are the same setting under debate, not two
settings, unless something other than the magnitude distinguishes them - a
named variant, a different stage, a different unit. One paper arguing the other
chose badly is the disagreement this system exists to surface, and splitting it
hides exactly the finding a reader needs.

DON'T split identical values. If two papers both state 512, they are describing
the same quantity.

DON'T split on wording alone. "batch size" and "minibatch size" are the same
quantity.

When the evidence does not distinguish them, keep them together and say so in
`reason`. Splitting on a guess hides a real disagreement.
