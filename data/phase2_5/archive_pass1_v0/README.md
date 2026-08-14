# Archived — the first pass/re-pass attempt (v0 guidelines)

Moved 15 Aug 2026, **not deleted**: the measurement it produced is real and is
cited in `chatcontext1.md` §3a and `CLAUDE.md`.

**What it measured:** two annotators, 20 shared items per decoder, under
guidelines v0 — **D1 kappa 0.262, D2 kappa 0.179**. Both in the slight-to-fair
band, i.e. the guidelines were not producing reproducible labels. That finding
is why guidelines v1 exist and why the exercise restarts here.

**Why the labels are not carried forward:** they were produced under v0, and v1
changes the answer to whole classes of item (rule 1 alone flips 10 of Sabbir's
40 D1 labels). Mixing v0 and v1 labels in one training set would put two
different labelling standards behind one gold column.

Contents: both annotators' D1/D2 labels, the adjudication CSV and its
assistant-drafted proposal, the derived gold sets, and the item batches they
were drawn against.
