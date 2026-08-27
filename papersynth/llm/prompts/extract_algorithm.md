You are transcribing algorithms from ONE research paper so that an engineer can
implement them without reading the paper.

The algorithm blocks below were extracted from the paper's source, in their
original pseudocode form.

## Output

Return a JSON array, one object per algorithm, in the same order given:

- `index`         the algorithm's index from the list below, as an integer
- `name`          a short descriptive name, e.g. "Scaled dot-product attention"
- `inputs`        list of `{name, type, description}`; type and description may be null
- `outputs`       list of `{name, type, description}`
- `steps`         list of `{index, text, refs_equations, refs_symbols}`, one per
                  numbered step, in order, starting at 1
- `complexity`    `{time, space}` using the paper's own notation, or null for
                  either if the paper does not state it
- `preconditions` list of stated assumptions, or an empty list

## Rules

DO transcribe every step. An algorithm missing a step is worse than no
algorithm, because it looks complete.

DO keep each step's text faithful to the pseudocode. Reword only to expand
notation into a readable sentence, never to change what the step does.

DO record `refs_equations` when a step invokes a numbered equation, and
`refs_symbols` for the symbols it reads or writes.

DON'T state a complexity the paper does not give. Use null. A plausible-looking
complexity that the authors never claimed is a fabrication.

DON'T fill in a step the pseudocode leaves implicit, however obvious it seems.
If initialization is unstated, it is unstated - that becomes a recorded gap.

DON'T merge or reorder steps.

## Paper text

{sections}

## Algorithm blocks

{algorithms}
