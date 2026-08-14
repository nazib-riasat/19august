# D1 annotation guidelines — v1

**Supersedes v0.** Revised 15 August 2026 from the first inter-annotator
measurement: two annotators labelled the same 20 items and agreed on **half of
them** (Cohen's κ = 0.262). Five of the ten disagreements had the *same* cause,
and it was a rule v0 never stated. This version states it first.

Every rule below is followed by the disagreement that produced it, so a later
reader can see the rule was measured rather than invented.

---

## The decision, in order

Ask these two questions, in this order. Most items are settled by the first.

### 1. Is the mention's own entity already in the candidate list?

**If yes → `L<n>`, link to it.** Do not think about whether *you* have seen it
before.

> **Why this is rule one.** The items you annotate are a *sample*. The candidate
> list is built from the whole corpus, so an entity can be there because of a
> mention you were never shown. Five of ten disagreements were exactly this:
> one annotator answered "I haven't seen this yet, so `CREATE`" on
> `Japanese Alps Visitor Center`, `peace lily`, `Lowepro ProTactic 450 AW`,
> `Manfrotto MT057CXPRO4` and `Vans Old Skool sneakers` — and in all five the
> entity was sitting in the candidate list.
>
> **The candidate list is the graph's memory, not yours.**

Two things this rule does *not* say:

* **In the list ≠ link.** Check it is the *same thing*. `Meiji Shrine` had
  `kawai shrine` in its list; they are two different shrines, so the answer was
  `C`. Linking them would be a wrong merge — the single most damaging error this
  task can produce, because every later fact about one attaches to the other.
* **A near-miss is not a match.** `brown leather boots` next to
  `white Adidas Superstars` is a different pair of shoes.

### 2. Is it a specific thing in the world at all?

**If no → `N` (non-entity).** Three kinds recur:

| Kind | Examples from this corpus | Answer |
|---|---|---|
| Generic plurals / mass nouns | `beaches`, `cliffs`, `books` | `N` |
| **Generic product categories** | `remote flash triggers`, `shoe rack` | `N` |
| **Abstract phrases** — procedures, qualities, topics | `data preparation steps`, `consumer behavior and social media`, `authenticity in establishing credibility` | `N` |

> **Why rows two and three are new.** v0 said "generic nouns, abstract
> qualities" and left the boundary to intuition — and the annotators split in
> *both directions* on the three abstract phrases, which is the signature of a
> rule that does not exist rather than a rule being applied badly.
>
> The line: **a category is not an entity; a specific instance is.** "remote
> flash triggers" (a kind of product) is `N`; "my Manfrotto MT057CXPRO4" (one
> object) is an entity. A *possession* is an entity even when unnamed — "my car",
> "my parents" — because there is a specific thing being referred to.

**The category test, sharpened (added after the v1 re-measurement).** The two
remaining disagreements at κ = 0.829 were both on this line — `yoga apps` and
`customer data` — so it needs one more sentence:

> **Ask: could you point at exactly one of them?** "yoga apps" is a class of
> software with no particular app meant → `N`. "Down Dog" is one app → entity.
> "customer data" is a kind of data, not one dataset the user owns → `N`; "my
> Q3 sales spreadsheet" would be an entity.
>
> A plural almost always signals a category. A possessive or a proper name
> almost always signals an instance.

**Otherwise → `C` (create).** A real, specific thing that is not in the list.
Including personal entities: "my car", "my brother", "my peace lily" all create.

### 3. `D` (defer) — rare, and narrower than it looks

**Only when there is a specific referent and a later turn will identify it.**

* `Alex's famous ones` → `D` (famous *what*? the next turn says)
* `data preparation steps` → **`N`, not `D`** — there is no specific referent to
  resolve, so deferring never terminates.

> One disagreement was exactly this confusion. `D` is *"I know something specific
> is meant and I cannot yet tell which"*. It is not *"I am unsure"*.

---

## Keys

| Key | Action |
|---|---|
| `L0`, `L1`, … | link to candidate 0, 1, … — the **number**, not the literal `<n>` |
| `C` | create new entity |
| `N` | non-entity |
| `D` | defer |

Append `?` and a note to flag anything: `C ?not sure if brand or model`.

**Notes are not optional decoration.** Both systematic findings in this project
came from notes, not labels. If you hesitate more than a few seconds, leave ten
words. The first pass recorded **zero** notes across 40 items, including one that
took 73 seconds and turned out wrong.

---

## Worked examples

| Turn | Mention | Candidates | Answer | Why |
|---|---|---|---|---|
| "how often should I mist my peace lily?" | `peace lily` | includes `peace lily` | **`L<n>`** | rule 1 — in the list |
| "visit the Meiji Shrine… The Kawai Shrine, on the other hand…" | `Meiji Shrine` | `kawai shrine` only | **`C`** | in the list ≠ the same thing |
| "took shots of the cliffs and beaches" | `beaches` | `cliffs` | **`N`** | generic plural |
| "thinking about getting a remote shutter release" | `remote flash triggers` | — | **`N`** | product *category* |
| "the role of authenticity in establishing credibility" | `authenticity in…` | — | **`N`** | abstract quality |
| "I'll take those data preparation steps into consideration" | `data preparation steps` | — | **`N`** | abstract procedure, not a deferred referent |
| "I just got a new 70-200mm zoom lens" | `70-200mm zoom lens` | not in list | **`C`** | specific possession |

---

## Provenance of this revision

Derived from `data/phase2_5/adjudication_v1.csv` — 18 disagreements
between annotators **Sabbir** (pass 1, 40 items) and **Nazib** (independent
re-pass, 20 items). The adjudication was **drafted by the assistant and accepted
by delegation** rather than resolved item-by-item by both annotators; that is
weaker than the two-annotator adjudication `GATE0_CONTRACT.md` item 7 specifies,
and it is recorded here rather than glossed. The five rules above are what the
disagreements *mechanically* implied; the two genuinely contestable calls
(`d2p_0023`, `d2p_0005`) are flagged in that file.
