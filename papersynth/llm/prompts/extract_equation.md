You are annotating equations from ONE research paper so that an engineer can
reimplement them exactly.

The equations below were extracted from the paper's source. They are given to
you verbatim and are NOT to be rewritten.

## Output

Return a JSON array, one object per equation, in the same order given:

- `index`     the equation's index from the list below, as an integer
- `symbols`   every symbol appearing in the equation:
  - `sym`         the symbol as written, e.g. `d_k`, `\theta`, `Q`
  - `role`        what it denotes, in a few words, e.g. "key dimension"
  - `shape`       tensor shape if the paper states one, e.g. `[n, d_k]`, else null
  - `defined_by`  VERBATIM text from the paper where this symbol is defined or
                  described, or null if the paper never defines it

## Rules

DO copy `defined_by` character-for-character from the paper text. It is checked
against the document; a symbol whose text cannot be found is treated as
undefined.

DO set `defined_by: null` for any symbol the paper does not define. An equation
with undefined symbols is not implementable, and recording that honestly is the
point of this task.

DO include every free symbol, including subscripts and superscripts that carry
meaning, such as `d_k` rather than just `d`.

DON'T rewrite, simplify, correct, or reformat the LaTeX. It is taken from the
paper's own source and must survive unchanged.

DON'T invent a role for a symbol the paper never explains. Say it is undefined
instead.

DON'T include standard operators as symbols: softmax, exp, log, sum, max, and
the like are notation, not quantities to implement.

## Paper text

{sections}

## Equations

{equations}
