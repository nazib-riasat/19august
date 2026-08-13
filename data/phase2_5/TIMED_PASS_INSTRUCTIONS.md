# Phase 2.5 — the human timed pass, step by step

**What this produces:** Gate-0 item 8 — how many annotations per hour *you* can
produce, and how self-consistent your labels are (Cohen's κ). This is the one
number in Phase 2.5 no machine can produce, because it describes the person who
will annotate Phase 6's training data.

**Give this file to a Claude Code session and it can walk you through it** —
but the annotation itself must be typed by **you, in a real terminal you open
yourself**. The tool times each item from display to keypress; Claude's shell
is non-interactive and cannot host it, and a machine-assisted pass does not
answer Gate-0 item 8 (`PHASE2_5_DECISIONS.md` §4).

---

## Before you start (5 minutes)

1. Read `data/phase2_5/GUIDELINES_D1_v0.md` and `GUIDELINES_D2_v0.md` once.
2. Pick your annotator name — your own name or handle, **not** anything starting
   with "claude" (that prefix marks machine-assisted rows).
3. Open a terminal at the repo root (`e:\STUDY\CSE498R`).

## Pass 1 — the timed pass (~1.5–2 h total, 84 items)

Do each decoder in one sitting if you can; the tool times per item either way.
Replace `YOURNAME` throughout.

```
.venv\Scripts\python.exe scripts\phase2_5\annotate.py d1 --annotator YOURNAME
.venv\Scripts\python.exe scripts\phase2_5\annotate.py d2 --annotator YOURNAME
```

- **D1 keys** (34 items): `L<n>` link to candidate n · `C` create new entity ·
  `N` non-entity · `D` defer.
- **D2 keys** (50 items): `I` independent · `U` duplicate · `C` conflict ·
  `S` supersedes (B supersedes A).
- Append `?your note` to any answer to flag an unclear item.
- **Interrupted?** Re-run the same command — finished items are skipped.
- Don't rush and don't deliberate abnormally: the point is *your natural pace*.

## Pass 2 — the re-annotation (~30–45 min, 20 items per decoder)

**Wait at least 2 calendar days after pass 1** (the tool checks the timestamps
and the κ report says whether the gap was honoured — see the note below if you
cannot wait).

```
.venv\Scripts\python.exe scripts\phase2_5\annotate.py d1 --annotator YOURNAME --pass-2 --subset 20
.venv\Scripts\python.exe scripts\phase2_5\annotate.py d2 --annotator YOURNAME --pass-2 --subset 20
```

Same keys. Do **not** look at your pass-1 answers first — that is the whole
measurement.

## The numbers

```
.venv\Scripts\python.exe scripts\phase2_5\annotate.py kappa d1 --annotator YOURNAME
.venv\Scripts\python.exe scripts\phase2_5\annotate.py kappa d2 --annotator YOURNAME
```

Then hand the rest to Claude — paste this:

> Read `data/phase2_5/labels/` (my annotator name is YOURNAME),
> `data/phase2_5/power.json` and `PHASE2_5_DECISIONS.md` §3/§5. Compute my
> items/hour per decoder from the recorded `seconds`, run the go/no-go
> arithmetic (items/hour × my hours/week × weeks available vs `n_test +
> n_train` per decoder — ask me for my hours/week and deadline), record the
> κ values and the honoured gap, and fill `PHASE2_5_DECISIONS.md` §2 and §5.
> If the answer is negative, present the three scope reductions from
> `GRAFT_PHASE2_5_BUILD.md` §5 and make me choose one — deferring is the one
> outcome the plan forbids.

## If you cannot wait 2 days

The gap exists so you *forget your answers* and re-derive them from the
guidelines; re-annotating from memory measures recall, not label stability, and
**inflates κ** — which biases the go/no-go toward "go", the risky direction.
If the deadline genuinely forces it: wait as long as you can (overnight
minimum), run pass 2, and let the κ report record the true gap. The number is
then labelled "measured at a shorter gap than the declared protocol; κ is an
upper bound", and `PHASE2_5_DECISIONS.md` §5 must say so. A weakened
measurement honestly labelled beats a protocol violated silently.
