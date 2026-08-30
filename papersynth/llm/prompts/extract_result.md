You are extracting REPORTED RESULTS from ONE research paper, so that an
engineer reimplementing it knows what numbers to expect.

A result is a measurement the paper reports: a benchmark score, an accuracy, a
perplexity. It is not a configuration value.

## Output

Return a JSON array. One object per reported number:

- `metric`            e.g. `BLEU`, `accuracy`, `F1`, `perplexity`
- `value`             the number
- `dataset`           the benchmark it was measured on, or null
- `split`             e.g. `newstest2014`, `dev`, `test`, or null
- `model_variant`     e.g. `base`, `large`, `xxlarge`, or null
- `conditions`        protocol details the paper states, as an object, e.g.
                      `{"beam_size": 4, "length_penalty": 0.6}`
- `reported_variance` standard deviation or plus-or-minus interval, or null
- `stated_explicitly` false if you read it from a figure rather than a table
                      or the text
- `quote`             VERBATIM text from the paper containing this number

## Rules

DO fill in `dataset`, `split` and `model_variant` whenever the paper says
them. These are not decoration: a score is only comparable to another score
measured the same way, and a missing split turns a dev number and a test number
into a contradiction that does not exist.

DO record `reported_variance` when the paper gives one. Two results whose
intervals overlap are not disagreeing.

DO copy `quote` character-for-character. It is checked against the document,
and a claim whose quote cannot be found is discarded.

DON'T record a number the paper attributes to earlier work as this paper's
result. A baseline row in a comparison table belongs to whoever produced it.

DON'T invent a split or a variant the paper does not state. Null is correct
and safe; a guessed split makes two incomparable numbers look comparable.

DON'T record configuration values. A learning rate is a setting, not a result.

DON'T record a range as a value. If the paper gives 27.3-27.8, record the
value the paper itself headlines and put the spread in `reported_variance`.

If the paper reports no results, return an empty array `[]`.

## Paper

{sections}
