You are extracting DESIGN DECISIONS from ONE research paper, so that an
engineer knows which approach to implement and where this paper disagrees with
others.

A design decision is a choice between approaches to a sub-problem: which
pretraining objective, which positional encoding, which normalization
placement, which tokenizer, which masking strategy. It is not a number.

## Output

Return a JSON array. One object per decision:

- `sub_problem`   snake_case name of what is being decided, e.g.
                  `sentence_level_objective`, `positional_encoding`,
                  `normalization_placement`
- `approach`      the approach as the paper names it, e.g. "next sentence
                  prediction (NSP)"
- `adopted`       true if this paper uses it, false if it explicitly rejects or
                  removes it
- `alternatives_rejected`  approaches the paper considered and did not take
- `rationale`     the reason the paper gives, or null
- `applies_to`    the component it concerns, or `"global"`
- `condition`     the scope under which it holds, or null
- `stated_explicitly`  false if you inferred it rather than read it
- `quote`         VERBATIM text from the paper stating this decision

## Rules

DO record a REMOVAL as a decision with `adopted: false`. A paper saying "we
remove the next sentence prediction objective" is making a claim about NSP, and
recording only what a paper adds would make it look silently identical to one
that never mentioned it.

DO name `sub_problem` for the QUESTION being answered, not the answer. Both
"next sentence prediction" and "sentence order prediction" answer
`sentence_level_objective`. If you name the sub-problem after the answer, two
papers that genuinely disagree will never be compared.

DO copy `quote` character-for-character. It is checked against the document,
and a claim whose quote cannot be found is discarded.

DON'T record a rationale the paper does not give. Use null.

DON'T record hyperparameter values here. A learning rate is a number, not a
choice between approaches.

DON'T record a decision this paper merely attributes to earlier work unless
this paper adopts it too.

DON'T invent a decision from silence. A paper not mentioning an approach has
not rejected it.

If the paper states no design decisions, return an empty array `[]`.

## Paper

{sections}
