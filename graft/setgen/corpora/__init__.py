"""P9.2 — corpus adapters: the **only** package that may name a corpus.

Everything above :mod:`graft.setgen.proofs` consumes ``ProofExample`` and knows
nothing of 2Wiki, MuSiQue or HoVer.  ``test_setgen_real.py`` asserts that with an
import-graph test, which is what the architecture asks for by name and what keeps
the conversational track a drop-in if the Wikipedia→conversation transfer claim
fails (`CLAUDE.md` §7 — a declared, untested claim).

Adapters register on **import**, not by scanning: an adapter that is never
imported is never registered, so the boundary holds at runtime and not merely in
the source.  Importing this package registers the two Tier-1 adapters; HoVer is
gated (decision 1) and lands only after the pair runs end to end.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

__all__ = ["wiki2", "musique_ans", "scoring", "stratified_sample"]


def stratified_sample(
    rows: Sequence[Mapping[str, Any]],
    stratum_of: Callable[[Mapping[str, Any]], str],
    limit: int | None,
    seed: int,
) -> list[Mapping[str, Any]]:
    """``limit`` rows drawn proportionally across strata, seeded.

    **A head slice is not a sample, and on one of the two Tier-1 corpora it is
    catastrophically not one** (found by adversarial audit, 16 Aug 2026):
    ``musique_ans_v1.0_train.jsonl`` is **sorted by hop count**, so its first
    2,000 rows are *all* 2-hop while the file as a whole is ~72% 2-hop, 22%
    3-hop, 5% 4-hop. Decision 2 pins the subset as "stratified by hop /
    decomposition depth" precisely against this, and the first version of the
    adapters ignored it and took ``rows[:limit]``.

    The cost of getting it wrong is not cosmetic. `DATASET_DECISION.md` §2 adopts
    HoVer because "2Wiki/MuSiQue/HotpotQA are all thin at >= 3 hops" and
    "**minimality and the closure rule only bite when proof sets are genuinely
    large**". An all-2-hop MuSiQue subset removes exactly the questions Stage D's
    minimality claim needs, and it would have done so silently.

    Proportional rather than balanced: the aim is a subset whose stratum mix
    matches the corpus, not one that over-weights rare strata into significance
    they do not have. Remainders go to the largest strata first, and every
    stratum is shuffled under ``seed`` before slicing so the draw is reproducible
    and is not itself a head slice within the stratum.
    """
    import random

    if limit is None or limit >= len(rows):
        return list(rows)

    buckets: dict[str, list] = {}
    for row in rows:
        buckets.setdefault(str(stratum_of(row)), []).append(row)

    order = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
    total = len(rows)
    quota = {k: int(limit * len(buckets[k]) / total) for k in order}
    while sum(quota.values()) < limit:
        for k in order:
            if quota[k] < len(buckets[k]):
                quota[k] += 1
                if sum(quota.values()) == limit:
                    break

    out: list = []
    for k in order:
        pool = list(buckets[k])
        random.Random(f"{seed}:{k}").shuffle(pool)
        out.extend(pool[: quota[k]])
    # Restored to corpus order so downstream reports and caches are stable and
    # do not depend on the bucket iteration above.
    index = {id(r): i for i, r in enumerate(rows)}
    return sorted(out, key=lambda r: index[id(r)])


from graft.setgen.corpora import musique_ans, scoring, wiki2  # noqa: E402,F401  (registration)
