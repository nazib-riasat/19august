"""Every prompt Phase 5 sends to a model, and one hash over all of them.

**One registry, one SHA** (P5.3).  Three prompts run in this phase — extraction,
the rolling summary, and the obligation parser — and they are the same kind of
frozen instrument as `beta` or the grounding ladder: a word changed in any of
them changes what comes out of the extractor, which changes assertion ids, which
changes the log.  Keeping them in one module with one digest means the manifest
carries a single value that moves when any prompt moves, rather than three that
somebody has to remember to check.

**The prompts ask for quotes, never for offsets.**  LLM character offsets are
unreliable; the spike established the working pattern and Phase 5 keeps it, with
one extension (G9): a quote names *which* context turn it came from, so a claim
assembled across turns can record every supporting span, as plan §3.1 requires.

**No prompt asks a model whether something is true.**  Extraction asks what was
said, the summary asks what has been established in the conversation, and the
parser asks what the question requires.  Truth is not a field in this schema and
must not enter through a prompt (`test_no_schema_field_claims_truth` guards the
schema; this docstring is the same rule for the prompts).
"""

from __future__ import annotations

from graft.canonical import digest_of

__all__ = [
    "EXTRACT_SYSTEM",
    "EXTRACT_USER",
    "REPAIR_USER",
    "SUMMARY_SYSTEM",
    "SUMMARY_USER",
    "OBLIGATION_SYSTEM",
    "OBLIGATION_USER",
    "EXTRACTION_JSON_SCHEMA",
    "REGISTRY",
    "REGISTRY_SHA",
]


# --------------------------------------------------------------------------
# extraction (G9's schema)
# --------------------------------------------------------------------------

#: The frozen output shape.  ``kind`` is closed over ``ASSERTION_KINDS`` — the
#: spike validated the vocabulary and G9 froze it — and ``turn_offset`` is the
#: cross-turn provenance G9 adds: ``0`` is the current turn, ``-1`` the one
#: before it, down to ``-m``.  Also the grammar candidate B constrains against,
#: which is why it is a data structure rather than only prose in the prompt.
EXTRACTION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        "assertions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"enum": ["claim", "value", "event", "time"]},
                    "text_norm": {"type": "string"},
                    "quotes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "turn_offset": {"type": "integer"},
                                "text": {"type": "string"},
                            },
                            "required": ["turn_offset", "text"],
                        },
                    },
                },
                "required": ["kind", "text_norm", "quotes"],
            },
        },
    },
    "required": ["mentions", "assertions"],
}

EXTRACT_SYSTEM = """You extract structured memory from one conversation turn.
Given the conversation context and the CURRENT TURN, output ONLY a JSON object:
{
 "mentions": [{"text": "<exact substring of the CURRENT TURN naming an entity>"}],
 "assertions": [{"kind": "claim|value|event|time",
                 "text_norm": "<the fact, one sentence, self-contained: resolve pronouns and relative dates from context>",
                 "quotes": [{"turn_offset": 0, "text": "<exact contiguous substring of that turn>"}]}]
}
Rules:
- Every "text" MUST be copied verbatim from the turn its "turn_offset" names.
- turn_offset 0 is the CURRENT TURN; -1 is the turn immediately before it, and so on.
- Use more than one quote only when the fact genuinely needs support from an earlier turn.
- Extract facts about the user and their world (preferences, events, values, dates).
- kind: "value" for numbers/quantities, "time" for dates/durations, "event" for things that happened, "claim" otherwise.
- No commentary. JSON only. Empty lists are fine for turns with no extractable content."""

EXTRACT_USER = """{summary_block}CONTEXT (earlier turns, {session_date}):
{context}

CURRENT TURN [{speaker}] (turn_offset 0):
{text}"""

#: The repair reprompt (candidates A and C).  It carries the *parse error* back
#: rather than only saying "try again": the failure the spike measured was
#: malformed JSON, and an error message is the one piece of information the
#: model does not already have.
REPAIR_USER = """Your previous reply could not be parsed as JSON.

Parser error: {error}

Reply again with ONLY the JSON object described in the system message. No prose,
no code fences, no trailing text."""


# --------------------------------------------------------------------------
# the rolling summary (G3)
# --------------------------------------------------------------------------

SUMMARY_SYSTEM = """You maintain a running summary of one conversation, used only as
context for fact extraction.
Write at most 8 short bullet points recording what has been established about the
user and their world: stable preferences, ongoing situations, entities they refer
to by name, and anything a later turn might refer back to with a pronoun or a
relative date.
Do not speculate, do not add advice, and do not repeat the assistant's
suggestions unless the user accepted them. Output the bullets and nothing else."""

SUMMARY_USER = """{previous_block}CONVERSATION SO FAR (most recent {n_turns} turns):
{turns}"""


# --------------------------------------------------------------------------
# the obligation parser (G7, fix F2)
# --------------------------------------------------------------------------

#: ``scope`` is **absent from this prompt on purpose** (see
#: ``graft.ingest.oblparse``): it is a tuple of ``conv_id``s that `H`'s
#: sub-check 5 rejects evidence against, and a hallucinated id would silently
#: reject correct evidence.  It is metadata, supplied by the caller.
#:
#: ``time_expression`` is asked for as a **phrase**, not as a resolved interval.
#: The widening to a half-open interval at natural granularity is arithmetic and
#: is done in code (``graft.ingest.timeexpr``), because Phase 0 §2.5 makes it a
#: requirement of Stage A and an LLM computing date arithmetic is an error source
#: with no upside.
OBLIGATION_SYSTEM = """You read one question asked of a conversational memory system
and fill typed slots. Output ONLY a JSON object:
{
 "entity_anchor": "<the entity the question is about, as a short noun phrase, or null>",
 "value_type": "<the kind of value the answer must supply, or null>",
 "time_expression": "<the time expression the question constrains the answer to, copied or lightly normalised, or null>",
 "needs_source": true|false,
 "aggregate": true|false
}
Rules:
- entity_anchor names WHO or WHAT the question is about, not the answer.
- value_type is a short type name such as "date", "price", "duration", "location",
  "name", "count", "preference"; null when the question does not request a typed value.
- time_expression is the phrase itself ("last May", "in 2023", "yesterday"), never a
  computed date range, and null when the question imposes no time constraint.
- needs_source is true only when the question asks who said something or where it
  came from.
- aggregate is true when the answer requires combining several facts (a count, a
  total, a list, a comparison).
- No commentary. JSON only."""

OBLIGATION_USER = """QUESTION_DATE: {question_date}
QUESTION: {question}"""


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

REGISTRY: dict[str, object] = {
    "extract_system": EXTRACT_SYSTEM,
    "extract_user": EXTRACT_USER,
    "repair_user": REPAIR_USER,
    "summary_system": SUMMARY_SYSTEM,
    "summary_user": SUMMARY_USER,
    "obligation_system": OBLIGATION_SYSTEM,
    "obligation_user": OBLIGATION_USER,
    "extraction_json_schema": EXTRACTION_JSON_SCHEMA,
}

#: One digest over every prompt in this phase.  Stamped into the run manifest
#: and into :func:`graft.ingest.pins.ingestion_fingerprint`.
REGISTRY_SHA = digest_of(REGISTRY)
