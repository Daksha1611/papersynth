You are extracting DEFINITIONS from ONE research paper, so that an engineer
knows what each term means and where this paper means something different by it
than another paper does.

A definition says what a term DENOTES: what counts as an instance of it, or how
it is measured. "An attack is successful if the agent calls a tool outside the
user-granted set" is a definition of `attack_success`. It is not a design
decision and it is not a number.

## Output

Return a JSON array. One object per definition:

- `term`        snake_case name of the term being defined, e.g.
                `attack_success`, `task_failure`, `harmful_action`,
                `robustness`
- `definition`  what the term means, in the paper's own words
- `criterion`   the operational test the definition reduces to - the thing code
                would actually check - or null if the paper gives none. "the
                final state matches a disallowed pattern" is a criterion;
                "a safety violation" is not.
- `attribution` `"own"` if THIS paper defines the term this way, `"prior_work"`
                if it is restating another paper's definition
- `applies_to`  the component or metric it concerns, or `"global"`
- `condition`   the scope under which this definition holds, or null
- `stated_explicitly`  false if you inferred the definition rather than read it
- `quote`       VERBATIM text from the paper stating this definition

## Rules

DO name `term` for the WORD being defined, not for its meaning. Two papers that
both define `attack_success` must use the same `term` even when they mean
different things by it - that shared term under conflicting definitions is
exactly what this extraction exists to surface. Name it after the meaning and
the disagreement becomes invisible.

DO fill `criterion` with the concrete check whenever the paper gives one, even
if you must compress a paragraph into one testable condition. Two definitions
are only in conflict through their criteria: identical prose with different
criteria conflict, and different prose with the same criterion does not.

DON'T record a definition the paper only uses in passing without pinning down.
A term mentioned is not a term defined.

DO set `attribution: "prior_work"` when the paper is quoting or summarizing
someone else's definition rather than stating its own. "Following [12], we call
a trajectory unsafe when..." with the paper then adopting it is still `"own"`.

DO copy `quote` character-for-character. It is checked against the document,
and a claim whose quote cannot be found is discarded.

DON'T invent a definition from silence. A paper using a term without defining
it has not defined it; leave it out.

If the paper defines no terms, return an empty array `[]`.

## Paper

{sections}
