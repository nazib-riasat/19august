"""Every prompt Phase 6 sends to a model, and one hash over all of them.

**A Stage-B registry, deliberately separate from the Phase-5 one.**  The
Phase-5 registry's SHA is a component of `ingestion_fingerprint`, and that
fingerprint is baked into recorded Phase-5 artefacts (`phase5_pilot.json`,
`phase5_bakeoff.json`) — adding a Stage-B prompt to it would silently invalidate
frozen experimental records for a change that never touched ingestion.  So each
stage carries its own registry, each folded into its own fingerprint
(``pins.stage_b_fingerprint`` here), and "one SHA covers every prompt" holds
*per stage*, which is the form of the claim that can actually stay true.
*(Corrected 14 Aug 2026: `llmlink` claimed its prompt "joins the Phase-5
registry" while living in neither registry — a fingerprint hole where a prompt
edit would have moved no recorded hash at all.)*

The prompts themselves follow the Phase-5 registry's rules: they ask for
decisions, never for truth, and a word changed in any of them changes what the
baseline does — which is why the SHA is in the fingerprint.
"""

from __future__ import annotations

from graft.canonical import digest_of

__all__ = ["LINK_SYSTEM", "LINK_USER", "REGISTRY", "REGISTRY_SHA"]


#: The LLM-prompted-linking baseline's prompt (G11, decision 12).  The candidate
#: list is rendered with ids so the reply can name one — the same information
#: the learned D1 gets, which is what makes the comparison a comparison rather
#: than a handicap.
LINK_SYSTEM = """You resolve an entity mention in a conversation against a list of known entities.
Output ONLY a JSON object:
{"action": "LINK_EXISTING|CREATE_NEW_ENTITY|NON_ENTITY|DEFER", "entity_id": "<id or null>"}
Rules:
- LINK_EXISTING only when the mention refers to one of the listed candidates; then entity_id is that candidate's id.
- CREATE_NEW_ENTITY when the mention names a real entity that is not in the list (including personal entities like "my car").
- NON_ENTITY when the phrase names no entity at all.
- DEFER when the turn does not carry enough information to decide.
- No commentary. JSON only."""

LINK_USER = """CONTEXT (recent turns):
{context}

CURRENT TURN:
{turn}

MENTION: "{mention}"

KNOWN ENTITIES:
{candidates}"""


REGISTRY: dict[str, object] = {
    "link_system": LINK_SYSTEM,
    "link_user": LINK_USER,
}

#: One digest over every Stage-B prompt, folded into ``stage_b_fingerprint``.
REGISTRY_SHA = digest_of(REGISTRY)
