"""P6.8 — the LLM-prompted-linking baseline (G11, decision 12).

**[EVIDENCE]** This is what Mem0 and Zep actually do — Mem0's update pipeline is
LLM-prompted (ECAI 2025) — which is the entire point: the baseline is faithful to
the deployed pattern rather than a strawman built to lose.  If prompting a
mid-tier API model matches the learned constructor, Contribution 1 is in trouble
and the project should learn that at Gate 1, cheaply.

**It spends real money, so four rules apply and none is optional.**

*Ledgered.*  Every call goes through ``Ledger`` (``llm_calls``, ``llm_tokens_in``,
``llm_tokens_out``), because the Gate-1 table reports cost per arm and a baseline
whose cost is unrecorded cannot be compared on cost.

*Capped, and enforced rather than observed.*  ``would_exceed`` is consulted
**before** each call — Phase 4's lesson, where a ledger that only counted let
methods drift over budget and the comparison became silently unfair.  With no
budget signed off (``pins.LLM_BASELINE["max_usd"] is None``) it refuses to spend
anything at all: the same fail-closed shape as an unfrozen extractor.

*Cached to disk, keyed by prompt hash, and the cache is the run's record.*  After
the budget is spent the numbers stay reproducible — re-runs replay the cache
byte-for-byte, exactly like ``ReplayExtractor``.  This is what makes the baseline
auditable by someone who never had an API key.

*No local stand-in.*  Substituting a local model would change what the comparison
means, so the smoke path replays a recorded cache fixture instead and **no test
spends money**.

**The prompt lives in the Stage-B registry** (``graphbuild.prompts``), whose SHA
is a component of ``stage_b_fingerprint`` — so a prompt edit moves a recorded
hash.  It deliberately does *not* join the Phase-5 registry: that registry's SHA
is baked into frozen Phase-5 artefacts, and "one SHA covers every prompt" holds
per stage (see ``graphbuild.prompts`` for the full reasoning).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from graft.canonical import digest_of
from graft.graphbuild.pins import LLM_BASELINE
from graft.graphbuild.prompts import LINK_SYSTEM, LINK_USER
from graft.ledger import Ledger

__all__ = ["LINK_SYSTEM", "LINK_USER", "PromptCache", "LlmLinker", "BudgetRefused"]


class BudgetRefused(RuntimeError):
    """Raised instead of spending money that was never authorised."""


class PromptCache:
    """Prompt hash → reply, on disk, committed as the run's record.

    One JSON file rather than a directory of blobs: at Gate-1 volumes (hundreds
    to a few thousand calls) it is small, and a single file is what makes "the
    cache is the record" a thing you can actually commit and diff.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, Any] = {}
        if self.path.is_file():
            self._entries = json.loads(self.path.read_text(encoding="utf-8"))

    def key(self, system: str, user: str, model: str) -> str:
        return digest_of({"system": system, "user": user, "model": model}, 24)

    def get(self, key: str) -> Any | None:
        return self._entries.get(key)

    def put(self, key: str, value: Any) -> None:
        self._entries[key] = value

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, indent=1, sort_keys=True), encoding="utf-8", newline="\n"
        )

    def __len__(self) -> int:
        return len(self._entries)


class LlmLinker:
    """The baseline, with the cache in front of the API and the cap in front of both.

    ``replay_only=True`` is the test and re-run mode: a cache miss then **raises**
    rather than falling through to the API, so a test that accidentally exercises
    an unrecorded prompt fails loudly instead of quietly costing money.
    """

    name = "llm_prompted_linking"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        cache_path: str | Path | None = None,
        ledger: Ledger | None = None,
        replay_only: bool = True,
    ) -> None:
        self.config = dict(config or LLM_BASELINE)
        self.cache = PromptCache(
            cache_path or Path(self.config["cache_dir"]) / "cache.json"
        )
        self.ledger = ledger
        self.replay_only = bool(replay_only)
        self.calls = 0
        self.cache_hits = 0
        self.spent_usd = 0.0

    # -- the budget gate ---------------------------------------------------

    def _authorise(self) -> None:
        """Refuse before spending, never after (Phase-4's enforcement lesson)."""
        budget = self.config.get("max_usd")
        if budget is None:
            raise BudgetRefused(
                "no LLM budget is signed off (pins.LLM_BASELINE['max_usd'] is None). "
                "Decision 12 requires a dollar cap declared at sign-off; until then "
                "the baseline runs from its cache only."
            )
        # Refuse when the NEXT call could overshoot, not merely when the cap is
        # already crossed *(corrected 14 Aug 2026: checking spent-so-far alone
        # authorises the final call that lands past the cap)*.
        est = float(self.config.get("est_max_usd_per_call", 0.0))
        if self.spent_usd + est > float(budget):
            raise BudgetRefused(
                f"LLM budget would be exceeded: ${self.spent_usd:.4f} spent + "
                f"${est:.4f} worst-case next call > ${float(budget):.2f} cap"
            )
        if self.calls >= int(self.config.get("max_calls", 0)):
            raise BudgetRefused(
                f"call cap reached: {self.calls} of {self.config.get('max_calls')}"
            )

    # -- the interface -----------------------------------------------------

    def link(
        self,
        mention: str,
        turn_text: str,
        candidates: Sequence[Mapping[str, Any]],
        context: str = "",
    ) -> dict[str, Any]:
        """One mention → ``{action, entity_id, cached}``.

        The candidate list is rendered with ids so the reply can name one — the
        same information the learned D1 gets, which is what makes the comparison
        a comparison rather than a handicap.
        """
        rendered = (
            "\n".join(f"- {c['entity_id']}: {c.get('name', '')}" for c in candidates)
            or "(none)"
        )
        user = LINK_USER.format(
            context=context or "(none)", turn=turn_text, mention=mention, candidates=rendered
        )
        key = self.cache.key(LINK_SYSTEM, user, self.config["model"])

        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return {**self._parse(cached["reply"]), "cached": True}

        if self.replay_only:
            raise BudgetRefused(
                f"cache miss for mention {mention!r} in replay-only mode. Tests and "
                "re-runs replay a recorded cache; spending requires "
                "replay_only=False and a signed-off budget."
            )

        self._authorise()
        reply, tokens_in, tokens_out, cost = self._call(LINK_SYSTEM, user)
        self.calls += 1
        self.spent_usd += cost
        if self.ledger is not None:
            self.ledger.count("llm_calls", 1)
            self.ledger.count("llm_tokens_in", tokens_in)
            self.ledger.count("llm_tokens_out", tokens_out)
        self.cache.put(
            key,
            {
                "reply": reply,
                "model": self.config["model"],
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "usd": cost,
            },
        )
        return {**self._parse(reply), "cached": False}

    def _call(self, system: str, user: str) -> tuple[str, int, int, float]:
        """The one place money is spent.  Isolated so it is trivially auditable."""
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - not installed in CI
            raise RuntimeError(
                "the LLM baseline needs the `openai` client. It is deliberately not "
                "in any requirements file: the baseline is run once, by a person "
                "with a key and a signed-off budget, and every test replays its "
                "cache instead."
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise BudgetRefused("OPENAI_API_KEY is not set; refusing to attempt a call")

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        usage = response.usage
        tokens_in = int(getattr(usage, "prompt_tokens", 0))
        tokens_out = int(getattr(usage, "completion_tokens", 0))
        # gpt-4o-mini list price at time of writing; recorded in the artefact so a
        # later reader can re-derive the spend rather than trust this constant.
        cost = tokens_in * 0.15e-6 + tokens_out * 0.60e-6
        return response.choices[0].message.content or "", tokens_in, tokens_out, cost

    @staticmethod
    def _parse(reply: str) -> dict[str, Any]:
        """A reply → an action, failing to ``DEFER`` rather than to an exception.

        An unparseable reply is a *baseline behaviour*, not a harness error: it is
        exactly the failure mode that makes prompting fragile, and scoring it as
        `DEFER` (do nothing) is the most charitable reading available — it neither
        credits nor punishes the baseline for the parse failure itself.  The rate
        is reported.
        """
        import re

        match = re.search(r"\{.*\}", reply or "", re.DOTALL)
        if not match:
            return {"action": "DEFER", "entity_id": None, "parse_ok": False}
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"action": "DEFER", "entity_id": None, "parse_ok": False}
        action = str(obj.get("action", "DEFER"))
        if action not in ("LINK_EXISTING", "CREATE_NEW_ENTITY", "NON_ENTITY", "DEFER"):
            action = "DEFER"
        entity = obj.get("entity_id")
        return {
            "action": action,
            "entity_id": str(entity) if entity else None,
            "parse_ok": True,
        }

    def report(self) -> dict[str, Any]:
        return {
            "model": self.config["model"],
            "provider": self.config["provider"],
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cached_prompts": len(self.cache),
            "spent_usd": round(self.spent_usd, 4),
            "budget_usd": self.config.get("max_usd"),
            "replay_only": self.replay_only,
            "reading": (
                "the cache is the run's record: after the budget is spent the "
                "numbers stay reproducible by replay, and no test spends money "
                "(G11)"
            ),
        }
