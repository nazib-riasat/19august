# Session handoff — GRAFT, Phases 0–2

**Written 11 Aug 2026, at the end of the session that built Phases 0 and 1 and
planned Phase 2.** Read this plus `CLAUDE.md` and you can continue without
re-deriving anything.

---

## 1. Where the project actually is

| | State |
|---|---|
| **Phase 0** — scaffold, schemas, event log, ledger, config | **Built, green.** All 13 exit criteria. |
| **Phase 1** — `H`, `U`, `R`, masks, obligations, `d(s)` | **Built, green.** All 16 exit criteria. |
| **Phase 2** — ProofLattice + exact evaluator | **Planned, signed off, not started.** |
| Phases 3–11 | Not started. |

**367 tests, ~8 s, all passing.** Head is `b7c2155` on `master`.

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
pytest -q
python scripts/check_plan_consistency.py
python scripts/verify_handoff.py
```

Python **3.11** (`py -3.11`); the system default here is 3.14, which PyTorch has
no wheels for. `.venv` is never copied between machines — see
`PHASE0_DECISIONS.md` §3.1.

---

## 2. Read these, in this order

1. `CLAUDE.md` — why decisions are what they are, what was cut, cost to reverse.
2. `GRAFT_RESEARCH_PLAN_v1.md` (v1.2) — the science. **Wins any conflict.**
3. `GRAFT_EXECUTION_ARCHITECTURE_v1.md` (v1.1) — 12 phases, fixes F1–F13.
4. `PHASE0_DECISIONS.md`, `PHASE1_DECISIONS.md` — what the builds decided.
5. `GRAFT_PHASE2_BUILD.md` — **the next thing to build.** §6 is normative.

---

## 3. What to do next

**Build Phase 2**, following `GRAFT_PHASE2_BUILD.md` §4's eight steps. §6 is
signed off (including the `Δd ≤ 0.6` band, 9 Aug 2026); there are no open
questions.

The plan's own build order puts the tiny instance, the target and the flow oracle
**before** the generator (steps 3–5). That ordering was earned: the flow
recurrence, the FCS estimator and the mode definitions are exactly where review
found errors, and they are the parts a hand-computed 6-atom table settles in an
afternoon.

---

## 4. The one thing to understand before continuing

**The plan documents went through ten review rounds and roughly seventy defects.
The code went through one.** That gap is not about care; it is structural, and it
matters for how the next phase is run.

Measured: each decision is restated in four or five places — the gap section, the
module spec, the exit criterion, the normative table, the next phase's handoff.
A correction is therefore a four-or-five-place edit, and landing three of them
leaves a contradiction that *reads* as authoritative. In the fifth round, seven of
eleven defects were introduced by the fourth round's own fixes; in the eighth, ten
of fourteen were introduced by the seventh's.

Two mitigations are in place:

* **§6 of the Phase-2 plan is normative.** Where a restatement disagrees with the
  table, the table wins and the restatement is a bug. Restatements of high-drift
  decisions now *point at a decision number* instead of repeating its value.
* **`scripts/check_plan_consistency.py` runs in the test suite** — 36 retired
  wordings, 3 incompatible pairs, and a sequential-numbering check. It catches
  *recurrence*, not new errors. It has caught its own over-broad rules twice.

**The practical conclusion: stop reviewing the prose and start writing the code.**
Further review rounds on this document have been producing defects at roughly the
rate they remove them. The build will surface real problems — the ones that matter
— and tests will hold them fixed.

---

## 5. Defects found by review that you should not re-introduce

These were real, and each is now a registered retirement. They are listed because
the reasoning matters more than the string.

| Defect | Correct position |
|---|---|
| `exp(β·0) = 1` — an invalid set scoring 1 | The validity indicator is **multiplicative** |
| The empty proof set is formally valid and scores `R = 7.39` | `1 ≤ \|X\| ≤ max_atoms`; `stop_allowed(root)` is `False` |
| `p*(FAIL)` called a TV floor | `FAIL` is in **both** distributions, so TV = 0 is reachable |
| "A dead end proves no proof exists" | It licenses only *no proof found under this pool, policy, attempt count, budget* |
| `coverage = 1 − mean(d₁..d₄)` | Makes `temporal_correctness` identically `1 − d_time`; coverage is **binary per slot** |
| Equivalent-action collisions treated as measurable | Impossible on labelled sets — a theorem, and SA-GFN's correction does not apply |
| SA-GFN "L₁ ≈ 0.12 vs 0.01" | **Not in the paper.** Use 5,220 vs 1,042 cyclohexane fragments per 5,000 samples |
| "FCS is needed because Phase 9 relies on it" | **Invented.** No document specifies a Phase-9 distribution metric |
| Per-terminal DP for `p_θ` | One forward pass over the policy-independent state graph — 35 h → 0.3 h |
| `uint32` for `pool_cap ≤ 64` | 32 bits cannot address 64 atoms; `uint64` |

---

## 6. Open items, honestly

* **Gate-0 data contract** (v1.2 §7 items 1–10) is still unwritten. It blocks
  Phases 5–9, not Phases 2–4. `CLAUDE.md` §7 has the detail.
* **Three late-found baselines** — HyperMem, Chain-of-Memory, *How Memory
  Management Impacts LLM Agents* — are in the library but not in plan §5.3.
* **No reader-size experiment** exists, though the minimality claim's scope
  condition rests on a provisional single-author preprint.
* **β = 4.0** is a placeholder until the Phase-3 sweep. The `r_fail_margin` check
  in `graft/config/schema.py` refuses a β that makes `FAIL` competitive.

---

## 7. Working agreements that produced the good parts

* Label every claim `[EVIDENCE]` / `[HYPOTHESIS]` / `[ANALYSIS]`, and never blur
  them.
* **Verify a citation before relying on it.** One number in this project's
  documents survived six rounds and turned out not to be in the paper.
* Record decisions with their *cost to reverse*, not just their value.
* When a review finds a defect, fix the class, not the instance.
* Say what is not known. "No evidence found" is a finding.
