# Phase 2.5 — what the spike decided and measured

Date: 13 August 2026
Parent: `GRAFT_PHASE2_5_BUILD.md` · `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (Phase 2.5) · `GRAFT_RESEARCH_PLAN_v1.md` (v1.2 §7 Gate 0 item 8, risk #15)
Status: **tooling built and run end to end; bootstrap labels exist; the Gate-0
item-8 number awaits the human timed pass** — see §5, which is not a footnote.

Same convention as the other DECISIONS files: **[EVIDENCE]** = named paper,
venue stated · **[HYPOTHESIS]** = this project tests it · **[ANALYSIS]** =
judgment made here.

---

## 1. The §6 decisions — signed by the run, with two amendments

`GRAFT_PHASE2_5_BUILD.md` §6 was unsigned when the spike started. The build
adopted decisions 1–7 as written with **two amendments**, both ruled by the
project owner on 13 Aug 2026 before anything ran:

| # | Decision | As adopted |
|---|---|---|
| 1 | Corpus | **LongMemEval-S** (HF `xiaowu0162/longmemeval`, file `longmemeval_s`, licence **MIT** per the HF card), SHA-256 `08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894`, pinned in `scripts/phase2_5/common.py` and verified at every load. Knowledge-updates over-sampled per G3 |
| 2 | Sample | seed `20260813`: 6 knowledge-update + 2 multi-session + 1 temporal-reasoning + 1 single-session-user questions → their evidence sessions, **windowed** to the `has_answer` turns ±1 (sessions run ~25 turns; ten questions' full sessions are ~250, five times the plan's "~50") → **10 questions, 21 sessions, 58 turns**, inside the declared 45–70 band |
| 3 | Item definitions | G2's, verbatim — D1 carries the candidate list and the entity id is part of the answer; D2 shows both session dates |
| 4 | Agreement | Cohen's κ, self-agreement, ≥ 2-day gap — **amendment A1**: the re-annotation subset is **20 items per decoder**, not 20 total. κ on a 4-way decision over ~7 items (a 20-total split) is noise; the plan's singular "20-item subset" against exit criterion 4's "per decoder" was a genuine ambiguity, resolved on the side that makes the statistic mean something. `annotate.py --pass-2 --subset 20` implements it per decoder |
| 5 | Span-grounding floor | **amendment A2**: the floor is **0.80** on the 20 audited assertions. The plan deferred to "the Gate-0 threshold", which does not exist yet (the Gate-0 contract is unwritten); a measurement gate cannot wait on it. **[ANALYSIS]** at 0.80, at most one audited item in five is noise — below that the timing measures judging garbage, which is G7's failure |
| 6 | Storage | JSONL + provenance, no `graft/schemas.py` change. Labels: `data/phase2_5/labels/{d1,d2}_labels_<annotator>[_pass2].jsonl`; items and sample under `data/phase2_5/`; the raw corpus is gitignored and re-fetched per machine against the SHA pin |
| 7 | Required volume | derived in `scripts/phase2_5/power.py` → `data/phase2_5/power.json`; see §3 |

### The extractor amendment — a declared deviation from the architecture

**The spike's extractor is Qwen2.5-3B-Instruct, bf16 — not the architecture's
Qwen2.5-7B-Instruct 4-bit.** Ruled by the project owner on 13 Aug 2026, sized
to the actual machine (RTX 5050 Laptop, 8 GB). Three consequences, recorded
rather than glossed:

1. **Gap G5's original question — does the 4-bit 7B fit in 8 GB — remains
   unmeasured.** What is measured instead is the 3B bf16 fit (§2), which
   doubles as a rehearsal of the **Phase-10 reader load**: same model family
   and precision the frozen reader uses.
2. **Extraction quality is below what Phase 5 assumes.** If the G7 floor had
   failed, the first suspect would have been the prompt and the second the
   smaller model. (It passed — §2.)
3. The architecture is **not** edited: its extractor spec stands for Phase 5;
   this row records the spike-local deviation, per the protected-docs rule.

---

## 2. What the spike measured

All numbers from `data/phase2_5/extraction_stats.json`, `provenance.json` and
the G7 audit, on the RTX 5050 Laptop GPU (8,151 MiB), 13 Aug 2026.

**The G5-class hardware measurement (for the 3B, not the architecture's 7B):**

| Quantity | Measured |
|---|---|
| model load (bf16, from local cache) | 17.3 s |
| peak VRAM, torch-allocated | **6,478 MB** (system total during generation ~7.4 GB of 8.1) |
| generation throughput | **18.4 tokens/s**, greedy |
| 58 turns end to end | ~9.5 min |

The reader-family model **fits with ~0.7–1.6 GB of headroom** — which doubles
as a rehearsal of the Phase-10 reader load. Whether the architecture's 4-bit
7B extractor fits stays unmeasured (§1's amendment).

**Extraction quality (the numbers the timing measurement's validity rests on):**

| Quantity | Measured | Reading |
|---|---|---|
| turns processed | 58 / 58 | — |
| turns with JSON parse failure | **9 / 58 (15.5%)** | a genuine 3B-extractor cost; those turns yielded nothing. A Phase-5 finding delivered early: schema-constrained decoding (the architecture's plan) or the 7B is needed for production extraction |
| mentions / assertions extracted | 34 / 78 | **0.59 mentions per turn** — the plan's "~50 turns → ~100 D1 items" assumed ~2/turn; with this extractor and prompt the D1 yield is a third of that. Recorded, not tuned toward: reaching ~100 D1 items needs ~3× the turns or a mention-richer prompt |
| grounding failures (dropped) | 11 (~8.9% of extracted objects) | exact-then-fuzzy recovery per the SpanGrounder pattern; one recovered span is mis-bounded (`d1_0022`, leading "s ") — fuzzy windows can land off by a word |
| **G7 span-support audit, first 20 assertions** | **17 / 20 = 0.85 ≥ the 0.80 floor → the timing measurement is valid** | fails: an assertion paired to an unrelated cleaning-tips quote, a book-status claim paired to a question, an app description paired to a pricing parenthetical; two further borderline fragments counted as supported |

**Items derived:** D1 **34** (target 100 — see the yield row), D2 **50 / 50**,
of which 42 from knowledge-update questions and 23 cross-session — the G3
over-sampling did its job: the bootstrap labels contain CONFLICT and
SUPERSEDES cases (4 + 2), which a uniform sample would essentially never
produce.

**Bootstrap label distributions** (machine-assisted, §4): D1 — 21 CREATE, 8
LINK, 4 NON_ENTITY, 1 DEFER; D2 — 41 INDEPENDENT, 3 DUPLICATE, 4 CONFLICT, 2
SUPERSEDES. The D2 rare-class share (~12%) is consistent with `power.json`'s
over-sampled 10% assumption. 21 items carry flags/notes feeding the
guideline v1 revision — the two systematic ones: assistant list-replies flood
the pair pool with INDEPENDENT advice pairs (a pair-proposer issue, not an
annotation issue), and extraction can decontextualise conditional estimates
into apparent CONFLICTs (`d2_0023/24`).

---

## 3. The go/no-go arithmetic (G1)

`power.json` (assumptions printed beside every number):

- **D1 `n_test`**: McNemar paired-proportion, α = 0.05 two-sided, power 0.8,
  over δ ∈ {0.05, 0.08} and disagreement ψ ∈ {0.1, 0.2, 0.3} — a **range**,
  as the plan requires, roughly **123–941** items depending on the assumption
  pair; the headline planning figure is **δ = 0.05, ψ = 0.2 → n ≈ 627**.
- **D1 `n_train`**: assumption, 1,500 (ConEL-2-scale head-tuning; stated as
  an assumption with basis, per G1).
- **D2 `n_test`**: rare-class floor of 30 per class → **300** pairs at a 10%
  over-sampled rare share, **600** at 5%.
- **D2 `n_train`**: assumption, 1,200 (DialogRE per-relation order).

**The go/no-go compares:** `items/hour × hours/week × weeks available`
against `n_test + n_train` per decoder. The items/hour term is the human
annotator's — see §5.

---

## 4. What the bootstrap pass is and is not

The labels under `data/phase2_5/labels/*_claude-fable-5-bootstrap*` were
produced by the AI assistant reading each item (**machine-assisted flag set in
every row**). They exist so that:

- the guidelines were exercised on real items before a human uses them;
- Phase 6 has non-empty seed labels to develop loaders against;
- the item-derivation pipeline's output was sanity-checked end to end.

**They do not answer Gate-0 item 8.** The feasibility number is
minutes-per-item *for the person who will produce the real volume*, and no
machine pass measures that. Machine timing is recorded as absent, not faked.

---

## 5. Open — the human half of the measurement

| Step | Command | Constraint |
|---|---|---|
| Timed pass 1 | `python scripts/phase2_5/annotate.py d1 --annotator <you>` then `d2` | do it in one sitting per decoder if possible; the tool times each item |
| Timed pass 2 | same with `--pass-2 --subset 20` | **≥ 2 calendar days after pass 1** (G4); the κ report checks the gap from timestamps |
| κ + go/no-go | `annotate.py kappa d1 --annotator <you>` (and `d2`), then compare items/hour against §3 | exit criterion 9: if negative, the scope reduction is chosen **here** (plan §5's three options), not deferred |

---

## 6. Handoff

* `data/phase2_5/sample.json` — the pinned, seeded sample with full provenance.
* `data/phase2_5/extraction.jsonl` — grounded mentions and assertions.
* `data/phase2_5/{d1,d2}_items.jsonl` — the annotation items.
* `data/phase2_5/GUIDELINES_{D1,D2}_v0.md` — guidelines with worked examples.
* `data/phase2_5/labels/` — bootstrap labels (machine-assisted, flagged).
* `data/phase2_5/power.json` — the denominator, with assumptions.
* `data/phase2_5/provenance.json` — corpus SHA + licence, model, seeds.
* `requirements-spike.txt` — the spike's ML deps, pinned, separate from
  `requirements.txt`/`requirements-ml.txt` so Phases 0–4 stay lean.
