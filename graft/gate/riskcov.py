"""P8.5 — selective prediction: the risk–coverage curve and AURC (decision 6).

**[EVIDENCE]** The methodology is Geifman & El-Yaniv, *Selective Classification
for Deep Neural Networks* (NeurIPS 2017): a model paired with a **confidence
threshold**, evaluated not by one accuracy but by the *curve* traded between
coverage (what fraction it answers) and risk (how wrong it is on what it
answered), with the operating point chosen on dev.

**AURC is the primary** (decision 6, and Gate-0 item 10's one-primary rule).
**The named safety secondary is the false-abstention rate on answerable
questions**, and it is not decoration: AURC rewards *ranking*, and a gate that
abstains on everything has zero risk at zero coverage. Without the secondary, a
uselessly cautious gate scores well.

**Dual prevalence, because the training balance is not the deployment one** (G4).
Contrast pairs are 1:1 **by construction**; LongMemEval's natural unanswerable
rate is 30/500 = 0.06. A threshold picked at balanced prevalence and applied at
0.06 inflates false abstention. So the same dev predictions are **reweighted** to
both prevalences — no extra data, nothing discarded — the threshold is picked on
the natural curve, and **both curves ship** in the artefact.

**Nothing here imports torch.** The instrument is pure numpy over
``(p_answerable, label)`` pairs, which is what lets Phase 10/11 reuse it
unchanged on end-to-end outputs (§8) and lets its arithmetic be checked against a
hand-computed toy on a bare interpreter.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from graft.gate.pins import PREVALENCES, TARGET_RISK

__all__ = [
    "risk_coverage",
    "aurc",
    "selective_metrics",
    "reweight",
    "choose_threshold",
    "contrast_pair_accuracy",
    "brier",
    "expected_calibration_error",
    "bootstrap_interval",
    "evaluate",
]


def _arrays(scores: Sequence[float], labels: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.shape != y.shape:
        raise ValueError(f"{s.shape[0]} scores against {y.shape[0]} labels")
    return s, y


def risk_coverage(
    scores: Sequence[float],
    labels: Sequence[float],
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """The curve: answer the most-confident first, and track risk as coverage grows.

    ``scores`` is ``p_answerable``; ``labels`` is 1.0 for answerable. A question is
    *answered* when its score clears the operating threshold, and the **risk** at a
    coverage level is the fraction of answered questions that were in fact
    unanswerable — the errors a gate is supposed to prevent.

    **Ties are resolved by expectation, never by the label** (corrected 15 Aug
    2026). The first version sorted on ``(-score, label)``, which is wrong twice
    over: it lets the *gold label* order the curve — the metric peeking at the
    answer it is scoring — and ``np.lexsort`` in any case put label ``0`` first,
    the opposite of what its docstring claimed. It also invented coverage points
    no threshold can realise: two questions tied at 0.5 produced a 50% coverage
    point, though ``score >= t`` must accept both or neither.

    The correction: questions are grouped by score, and within a tied group each
    prefix position carries the group's **mean** error rate — the expected risk
    under a random ordering, which is the only label-independent answer. With no
    ties this is identical to the ordinary cumulative curve, so nothing changes
    for a model that separates its inputs. ``realizable`` marks the prefixes that
    a threshold can actually produce (the last index of each tie group).
    """
    s, y = _arrays(scores, labels)
    if s.size == 0:
        return {"coverage": [], "risk": [], "thresholds": [], "realizable": [], "n": 0}
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)

    order = np.argsort(-s, kind="stable")
    s, y, w = s[order], y[order], w[order]

    covered = np.cumsum(w)
    total = float(w.sum())
    errors = np.empty_like(covered)
    realizable = np.zeros(s.size, dtype=bool)

    # Walk tie groups: everything inside one group shares a score, so no
    # threshold can separate its members.
    before_w = 0.0
    before_e = 0.0
    start = 0
    while start < s.size:
        stop = start + 1
        while stop < s.size and s[stop] == s[start]:
            stop += 1
        group_w = float(w[start:stop].sum())
        group_e = float((w[start:stop] * (1.0 - y[start:stop])).sum())
        rate = (group_e / group_w) if group_w else 0.0
        for i in range(start, stop):
            errors[i] = before_e + rate * (float(covered[i]) - before_w)
        realizable[stop - 1] = True
        before_w += group_w
        before_e += group_e
        start = stop

    return {
        "coverage": (covered / total).tolist(),
        "risk": (errors / np.maximum(covered, 1e-12)).tolist(),
        "thresholds": s.tolist(),
        "realizable": realizable.tolist(),
        "n": int(s.size),
    }


def aurc(curve: Mapping[str, Any]) -> float:
    """Area under the risk–coverage curve — **lower is better**.

    **The arithmetic mean of the selective risks over all n prefixes**, which is
    the standard empirical AuRC: **[EVIDENCE]** Franc, Průša & Vörös,
    *Optimal Strategies for Reject Option Classifiers* (JMLR 2023), Eq. 27.

    Corrected 15 Aug 2026. The first version integrated trapezoidally over
    coverage and then renormalised by ``coverage[-1] - coverage[0]`` — i.e. over
    ``[1/n, 1]`` rather than ``[0, 1]`` — which is neither the published
    definition nor internally consistent: on the four-point toy it gives 0.19444
    against the standard 0.20833. Using a non-standard estimator would make every
    AURC in this project incomparable with every published one, which is exactly
    the comparison the metric exists to support.

    A perfect ranker still has AURC > 0 whenever any unanswerable exists — the
    curve must eventually cover them — so AURC is compared *between* models on one
    evaluation set and never read as an absolute score.
    """
    risk = list(curve.get("risk", []))
    if not risk:
        return 0.0
    return float(np.mean(risk))


def reweight(labels: Sequence[float], target_prevalence: float) -> np.ndarray:
    """Per-row weights that move the *observed* class balance to ``target``.

    G4's mechanism, and the reason it is reweighting rather than resampling: the
    same dev predictions are reused, so the two curves are computed over
    identical model outputs and differ *only* by the prevalence assumption. A
    resampled curve would also differ by which rows happened to be drawn.

    ``target_prevalence`` is the share of **unanswerable** questions, matching
    ``pins.PREVALENCES``.
    """
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    n_pos = float((y >= 0.5).sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.ones_like(y)
    target = float(target_prevalence)
    w_neg = target / (n_neg / y.size)
    w_pos = (1.0 - target) / (n_pos / y.size)
    return np.where(y >= 0.5, w_pos, w_neg)


def selective_metrics(
    scores: Sequence[float], labels: Sequence[float], threshold: float
) -> dict[str, Any]:
    """The metric group at one operating point (plan §6.4, in full).

    ``false_abstention_rate`` is the safety secondary: of the questions that were
    genuinely answerable, how many did the gate refuse. It is reported at every
    operating point precisely so that a good AURC cannot hide a gate that
    abstains its way to a low risk.
    """
    s, y = _arrays(scores, labels)
    answered = s >= threshold
    answerable = y >= 0.5
    n_answered = int(answered.sum())
    n_answerable = int(answerable.sum())
    n_unanswerable = int((~answerable).sum())
    wrong = int((answered & ~answerable).sum())
    return {
        "threshold": float(threshold),
        "coverage": (n_answered / s.size) if s.size else 0.0,
        "selective_risk": (wrong / n_answered) if n_answered else 0.0,
        "selective_accuracy": ((n_answered - wrong) / n_answered) if n_answered else 0.0,
        "abstention_recall": (
            int((~answered & ~answerable).sum()) / n_unanswerable if n_unanswerable else None
        ),
        "false_abstention_rate": (
            int((~answered & answerable).sum()) / n_answerable if n_answerable else None
        ),
        "n": int(s.size),
        "n_answered": n_answered,
    }


def choose_threshold(
    scores: Sequence[float],
    labels: Sequence[float],
    *,
    target_risk: float | None = None,
    prevalence: float | None = None,
) -> dict[str, Any]:
    """Decision 5: the dev threshold, at **natural** prevalence (G4).

    The rule: the **lowest** threshold whose reweighted selective risk stays at or
    under ``target_risk`` — i.e. answer as much as possible subject to a risk
    budget, which is the selective-prediction framing (**[EVIDENCE]** Geifman &
    El-Yaniv, NeurIPS 2017) rather than an accuracy one. If no threshold meets the
    budget the most conservative candidate is returned and ``met`` is ``False``,
    because silently returning a threshold that misses the budget is how a risk
    target stops being one.

    **The target comes from ``pins.TARGET_RISK`` and is no longer a bare default**
    (corrected 15 Aug 2026). An unpinned 0.10 sat *above* the natural base rate of
    0.06, so answering everything already satisfied it: on the first full run
    every arm and seed selected a threshold with weighted coverage 1.0 and
    abstained on nothing. A target that is not fingerprinted is also not part of
    the run's identity, so two runs at different targets were indistinguishable.

    **Vacuity is detected and reported, not left for a reader to notice.** A
    target at or above the base rate is met by the trivial gate; the result then
    carries ``vacuous: True`` and says so, because a threshold that abstains on
    nothing is not a selective classifier and no number computed at it means what
    it appears to.
    """
    s, y = _arrays(scores, labels)
    if s.size == 0:
        return {"threshold": 1.0, "met": False, "vacuous": True, "reason": "no dev rows"}
    p = PREVALENCES["natural"] if prevalence is None else float(prevalence)
    target = float(TARGET_RISK if target_risk is None else target_risk)
    w = reweight(y, p)

    best: dict[str, Any] | None = None
    for candidate in sorted({float(v) for v in s}):
        answered = s >= candidate
        covered = float(w[answered].sum())
        if covered <= 0:
            continue
        wrong = float(w[answered & (y < 0.5)].sum())
        risk = wrong / covered
        if risk <= target:
            best = {
                "threshold": candidate,
                "met": True,
                "weighted_risk": risk,
                "weighted_coverage": covered / float(w.sum()),
            }
            break
    if best is None:
        best = {
            "threshold": float(max(s)),
            "met": False,
            "reason": f"no threshold reaches risk <= {target} at prevalence {p}",
        }
    best["prevalence"] = p
    best["target_risk"] = target
    best["rule"] = "lowest threshold meeting the risk budget at natural prevalence"
    # **The vacuity guard.**  At prevalence `p` the trivial gate — answer
    # everything — has weighted risk exactly `p`.  A target at or above `p` is
    # therefore satisfied without abstaining on anything, and the "chosen"
    # threshold is an artefact of the budget rather than a property of the model.
    best["vacuous"] = target >= p
    if best["vacuous"]:
        best["vacuity_reading"] = (
            f"target_risk {target} >= natural prevalence {p}: answering everything "
            "already meets the budget, so this threshold abstains on nothing and "
            "no metric computed at it measures the gate. Lower the target below "
            "the base rate (pins.TARGET_RISK) before reading any operating point."
        )
    return best


def contrast_pair_accuracy(
    scores: Sequence[float],
    labels: Sequence[float],
    group_ids: Sequence[str],
) -> dict[str, Any]:
    """Within-pair accuracy: does the gate score the answerable twin higher?

    **This is the metric MuSiQue's contrast-pair design exists for, and it is the
    leakage-immune one.** AURC is a *global* ranking over the whole evaluation
    set, so an arm that sees the question embedding can improve it by ranking
    *across* questions — calibrating for question difficulty — without ever
    separating a twin. That is not leakage, but it does mean **an AURC gap
    between the arms is not by itself evidence of leakage** (a claim this
    project's own Phase-8 runner asserted before measuring it, and which the
    measurement corrected).

    Within a pair the question string is byte-identical, so every question-side
    feature is identical across the label by construction. A model can therefore
    only separate the twins using **pool-side** features — exactly the property
    decision 1 chose this corpus for. Ties count as failures, since a gate that
    cannot order the pair has not made the distinction.

    Groups that are not exactly one answerable and one unanswerable are skipped
    and counted rather than scored, for the same reason ``musique_pairs``
    reports malformed groups: a silently dropped pair is a silently smaller
    denominator.
    """
    s, y = _arrays(scores, labels)
    if s.size != len(group_ids):
        raise ValueError(f"{s.size} scores against {len(group_ids)} group ids")
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(group_ids):
        groups.setdefault(str(key), []).append(i)

    correct = ties = skipped = 0
    for key in sorted(groups):
        idx = groups[key]
        if len(idx) != 2 or {float(y[i] >= 0.5) for i in idx} != {0.0, 1.0}:
            skipped += 1
            continue
        pos = next(i for i in idx if y[i] >= 0.5)
        neg = next(i for i in idx if y[i] < 0.5)
        if s[pos] > s[neg]:
            correct += 1
        elif s[pos] == s[neg]:
            ties += 1
    scored = len(groups) - skipped
    return {
        "pairs_scored": scored,
        "pairs_skipped": skipped,
        "correct": correct,
        "ties": ties,
        "accuracy": (correct / scored) if scored else None,
        "chance": 0.5,
        "reading": (
            "within a pair the question is byte-identical, so this can only be "
            "won on pool-side features. Chance is 0.5; ties count as failures."
        ),
    }


def brier(scores: Sequence[float], labels: Sequence[float]) -> float:
    s, y = _arrays(scores, labels)
    return float(np.mean((s - y) ** 2)) if s.size else 0.0


def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[float], bins: int = 10
) -> float:
    """ECE with equal-width bins.

    Decision 9 reports Brier and ECE and adds **no calibrator** in Tier 1 — a knob
    without an instrument is what the decision refuses. These two numbers are the
    instrument that would justify one later.
    """
    s, y = _arrays(scores, labels)
    if s.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (s > lo) & (s <= hi) if lo > 0 else (s >= lo) & (s <= hi)
        if not in_bin.any():
            continue
        total += (in_bin.sum() / s.size) * abs(float(y[in_bin].mean() - s[in_bin].mean()))
    return float(total)


def bootstrap_interval(
    scores: Sequence[float],
    labels: Sequence[float],
    statistic: Any = None,
    *,
    group_ids: Sequence[str] | None = None,
    resamples: int = 1000,
    seed: int = 13,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap interval — **[ANALYSIS]**, this project's own protocol.

    Dror et al. (ACL 2018) is the authority for choosing a test **according to the
    experimental structure**, not for this resampling count; `CLAUDE.md` §3's rule
    is that the predeclaration and the seed count are labelled as the project's
    discipline rather than attributed to a paper.

    **Two corrections, 15 Aug 2026, both of which invalidated the intervals this
    function previously produced.**

    1. *It bootstrapped the wrong statistic.* The default was unweighted AURC
       while ``evaluate`` declares the **natural-prevalence** AURC as primary, so
       the reported interval did not bracket — or even describe — the reported
       number (measured: primary 0.036 against an interval of [0.362, 0.399]).
       The default is now the natural-prevalence AURC, the same quantity
       ``aurc_primary`` reports.
    2. *It resampled rows independently.* The evaluation set is made of **contrast
       pairs**: an answerable question and its twin, which are not independent
       observations. Resampling rows splits twins across draws and understates the
       variance. ``group_ids`` resamples **whole groups**, which is the structure
       Dror et al. require a test to respect. Without ``group_ids`` the old
       row-wise behaviour is kept, and the result says which was used.

    Required by exit criterion 7 for LongMemEval's **30** abstention questions,
    which must carry an interval and never a point estimate: at n = 30 one flipped
    decision moves a rate by 3.3 points (`DATASET_DECISION.md` §1.2).
    """
    s, y = _arrays(scores, labels)
    stat = statistic or (lambda a, b: aurc(risk_coverage(a, b, reweight(b, PREVALENCES["natural"]))))
    if s.size == 0:
        return {"point": None, "lo": None, "hi": None, "resamples": 0}
    rng = np.random.default_rng(int(seed))

    if group_ids is not None:
        if len(group_ids) != s.size:
            raise ValueError(f"{s.size} scores against {len(group_ids)} group ids")
        buckets: dict[str, list[int]] = {}
        for i, key in enumerate(group_ids):
            buckets.setdefault(str(key), []).append(i)
        keys = sorted(buckets)
        members = [np.asarray(buckets[k], dtype=int) for k in keys]

        def draw() -> np.ndarray:
            picked = rng.integers(0, len(members), len(members))
            return np.concatenate([members[j] for j in picked])
    else:
        def draw() -> np.ndarray:
            return rng.integers(0, s.size, s.size)

    draws: list[float] = []
    for _ in range(int(resamples)):
        idx = draw()
        value = stat(s[idx], y[idx])
        if value is not None and math.isfinite(float(value)):
            draws.append(float(value))
    point = float(stat(s, y))
    if not draws:
        return {"point": point, "lo": None, "hi": None, "resamples": 0}
    return {
        "point": point,
        "lo": float(np.percentile(draws, 100 * alpha / 2)),
        "hi": float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "resamples": len(draws),
        "seed": int(seed),
        "unit": "contrast pair" if group_ids is not None else "row",
        "statistic": "aurc at natural prevalence (the reported primary)",
        "method": "percentile bootstrap [ANALYSIS] — this project's protocol, not Dror et al.'s",
    }


def evaluate(
    scores: Sequence[float],
    labels: Sequence[float],
    *,
    threshold: float | None = None,
    interval: bool = True,
    seed: int = 13,
    group_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The full §6.4 group at both prevalences — what the artefact carries.

    Both curves ship (G4, exit criterion 6). The threshold, when not supplied, is
    chosen on the **natural**-prevalence curve per decision 5.
    """
    s, y = _arrays(scores, labels)
    curves = {
        name: risk_coverage(s, y, reweight(y, p)) for name, p in sorted(PREVALENCES.items())
    }
    chosen = (
        {"threshold": float(threshold), "met": None, "rule": "supplied by caller"}
        if threshold is not None
        else choose_threshold(s, y)
    )
    out: dict[str, Any] = {
        "n": int(s.size),
        "prevalences": dict(PREVALENCES),
        "curves": {k: {"coverage": v["coverage"], "risk": v["risk"]} for k, v in curves.items()},
        "aurc": {k: aurc(v) for k, v in curves.items()},
        "aurc_primary": aurc(curves["natural"]),
        "threshold": chosen,
        "at_threshold": selective_metrics(s, y, float(chosen["threshold"])),
        "brier": brier(s, y),
        "ece": expected_calibration_error(s, y),
        "primary_metric": "aurc (natural prevalence)",
        "safety_secondary": "false_abstention_rate_on_answerable",
    }
    if interval:
        out["aurc_interval"] = bootstrap_interval(s, y, seed=seed, group_ids=group_ids)
    return out
