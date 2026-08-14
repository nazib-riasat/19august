# D2 annotation guidelines — v1

**Supersedes v0.** Revised 15 August 2026 from the first inter-annotator
measurement: 20 shared items, **60% raw agreement, Cohen's κ = 0.179**. That is
lower than D1's, and D2 is the decoder Contribution 1 rests on — so this
revision matters more.

Every rule is followed by the disagreement that produced it.

---

## The decision

Two claims from one conversation. Exactly one of four.

### `I` — INDEPENDENT

Different facts. **This is the default and it is correct most of the time.**

### `U` — DUPLICATE

**The same speaker restating the same fact.** Wording may differ.

> "I'm planning to visit the Japanese Alps Visitor Center" / "I am looking
> forward to visiting the Japanese Alps Visitor Center" → `U`.

### `C` — CONFLICT

**Two claims asserting incompatible things about the same fact**, with no update
story. Both stay live; neither wins.

### `S` — SUPERSEDES

**The same fact, updated.** The later claim replaces the earlier one.

> "I weigh 70kg" → "I weigh 68kg" is `S`. The user changed; the old value is
> retired.

---

## The three rules the measurement forced

### Rule 1 — an assistant echoing the user is `I`

A user *decision* plus an assistant *endorsement* are two different assertions by
two different speakers. Not a duplicate: the user never said the assistant's
sentence.

> user "I'll go with the Think Tank 40L" / assistant "The Think Tank 40L is a
> fantastic choice" → **`I`**.
>
> This produced **seven** first-pass errors before it was stated. An assistant
> echo also cannot `S`: agreeing with someone does not retire what they said.

### Rule 2 — `S` needs the *same fact* updated

Two different objects are `I`, however similar the surrounding advice sounds.

> "A yoga strap is a useful tool" / "A thicker mat provides more cushioning" →
> **`I`**. Strap and mat are different objects.
>
> "issues with my old 18-55mm kit lens" / "I recently got a new 50mm prime" →
> **`I`**. Both are true at once; owning a new lens does not retire the old
> lens's problems.

### Rule 3 — `C` needs incompatible claims, not a shared topic

> "I'm happy to help you stay motivated on lazy Sundays" / "I've been using my
> Fitbit Charge 3 for 9 months" → **`I`**. They share a wellness topic and assert
> nothing incompatible.

**A note on the class that did not appear.** The first pass produced **zero `C`
labels in 49 items**, 34 of which were drawn from knowledge-update questions
*because* that is where conflicts live. Rule 3 tightens `C`, so do not read it as
licence to use `C` less — read it as: if you find a real one, it matters, and it
should carry a note.

### Rule 4 — sharing a subject is not sharing a fact

**Added after the clean re-measurement (15 Aug 2026).** All three residual
disagreements were this, and in all three the second annotator's note said
"might be an I also" — so it was doubt resolving toward the rarer class, not a
competing rule.

`U`, `C` and `S` all require the two claims to be about the **same
proposition** — not merely the same object, person or topic.

> "The 50mm f/1.8 is great for portraits" / "the image quality of a prime like
> the 50mm is superior to a kit lens" → **`I`**. One lens, two different
> properties, two facts.
>
> "Can you provide more information on social identity theory?" / "influencers
> can use social identity to drive sales" → **`I`**. A question and an answer
> are not one fact.

**When in doubt, `I` is the correct default.** The three non-independent classes
each make a strong claim — that one fact was restated, contradicted, or
replaced — and if you cannot name the single shared proposition, none of them
applies.

### Rule 5 — added detail restates; a changed value replaces

**Added 15 Aug 2026** from the one disagreement that survived at κ = 0.813.

> "it's been losing some leaves, but I've read that's normal"
> "My peace lily has been losing leaves since I brought it home."
>
> One annotator read the second as `U` (the same claim, with a timing detail
> added); the other as `S` (a temporal refinement replacing the vaguer version).

**The rule: `S` requires the *value* of the fact to change.** More detail about
the same unchanged fact is `U`.

* "I've used my Fitbit for 6 months" → "for 9 months" is **`S`** — the value moved.
* "it's losing leaves" → "it's been losing leaves since I brought it home" is
  **`U`** — still losing leaves; only the description is fuller.

Ask: **if I stored the first claim, would the second make it wrong?** If yes,
`S`. If it merely makes it less complete, `U`.

---

## Keys

| Key | Label |
|---|---|
| `I` | INDEPENDENT |
| `U` | DUPLICATE |
| `C` | CONFLICT |
| `S` | SUPERSEDES |

`S ?might just be a correction` appends a note. Use them — D2's first pass left
11 notes across 49 items and they were more useful than the labels.

---

## Quick reference

| Situation | Answer |
|---|---|
| Assistant agreeing with the user | `I` |
| Different objects, similar advice | `I` |
| Same topic, nothing incompatible | `I` |
| Same speaker, same fact, different words | `U` |
| Same fact, later value replaces earlier | `S` |
| Same fact, incompatible, no update | `C` |

---

## Provenance of this revision

Derived from `data/phase2_5/adjudication_v1.csv` — 8 D2 disagreements
between **Sabbir** (pass 1, 49 items) and **Nazib** (independent re-pass, 20).
The adjudication was **drafted by the assistant and accepted by delegation**
rather than resolved item-by-item by both annotators; that is weaker than the
two-annotator adjudication `GATE0_CONTRACT.md` item 7 specifies, and it is
recorded rather than glossed. **Two calls in it are genuinely contestable and are
flagged in that file**: `d2p_0023` (two assistant statements about Matsumoto —
`U` or `C`) and `d2p_0005`, where the proposal sided against pass 1.
