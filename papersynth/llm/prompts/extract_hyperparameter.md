You are extracting hyperparameters from ONE research paper so that an engineer
can reimplement it exactly.

A hyperparameter is any configured numeric or categorical value that an
implementer must set: learning rate, batch size, dropout, number of layers,
hidden dimension, optimizer choice, weight decay, warmup steps, temperature,
beam size, number of epochs or training steps, and so on.

## Output

Return a JSON array. One object per hyperparameter occurrence:

- `canonical_name`  snake_case normalized name, e.g. `learning_rate`
- `paper_symbol`    the symbol as written, e.g. `\eta`, or null
- `value`           the value: a number, or a string for categorical values
- `value_type`      one of `float`, `int`, `bool`, `categorical`
- `unit`            e.g. `steps`, `epochs`, `tokens`; null when dimensionless
- `applies_to`      the component it configures, or `"global"`
- `condition`       the exact scope under which this value holds, or null
- `stated_explicitly`  false if you read it from a figure or inferred it
- `quote`           VERBATIM text from the paper containing this value

## Rules

DO copy `quote` character-for-character from the text above. It is checked
against the document, and a claim whose quote cannot be found is discarded.

DO record `condition` whenever a value is scoped - "base model", "for WMT14
EN-DE", "during fine-tuning", "large variant". Values under different
conditions are not in conflict, and dropping the condition manufactures a
contradiction that does not exist.

DO emit one entry per occurrence. If a paper states a learning rate twice
under different conditions, that is two entries.

DO set `stated_explicitly: false` for any value you read off a figure or
inferred from surrounding text rather than read directly.

DON'T guess, default, or complete a value the paper does not state. A missing
hyperparameter is recorded elsewhere as a gap; a fabricated one corrupts the
spec silently. If the paper does not give it, omit it entirely.

DON'T include values from other papers, even ones cited here. Extract only what
THIS paper states as its own configuration.

DON'T include results, metrics, or measurements. BLEU 27.3 is a result, not a
hyperparameter.

DON'T convert units or rescale values. Record each value as the paper states
it, in the units the paper uses.

If the paper states no hyperparameters, return an empty array `[]`.

## Paper

{sections}
