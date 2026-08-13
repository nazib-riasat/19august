# D1 annotation guidelines v0 — mention resolution

**The item.** One highlighted mention inside one conversation turn, plus the
list of entities already registered in this conversation. Decide what the
mention is, and — when linking — *which* entity it is. The entity id is part
of the answer, not a detail: a right action with a wrong id scores wrong
(research plan v1.2 §6.4).

**The four actions, in the order to consider them:**

1. **LINK_EXISTING(entity_id)** — the mention refers to an entity already in
   the candidate list. Choose the id. Surface form may differ ("my sister" vs
   "Ana"); what matters is the referent.
2. **CREATE_NEW_ENTITY** — a genuinely new entity: a person, place,
   organisation, product, pet, project, or a *personal* entity ("my car",
   "my old apartment") mentioned for the first time in this conversation.
   Personal entities count (ConEL-2's precedent): "my car" is an entity even
   though it has no name.
3. **NON_ENTITY** — not an entity mention at all: generic nouns ("a book" in
   "I like reading a book before bed" when no specific book is meant),
   abstract qualities, verbs mis-extracted as mentions.
4. **DEFER** — you genuinely cannot tell *yet* from this turn plus its
   context, but a later turn plausibly resolves it. Use sparingly; it is not
   "hard", it is "under-determined here".

**Tie-breakers.**
- New specific referent vs generic: would a memory system need a node for it?
  If a later question could ask about *it specifically*, it is an entity.
- Same name, different referent → CREATE_NEW_ENTITY, not LINK.
- The speaker ("I", "me", "my") is the user entity; if the registry has no
  user entity yet, CREATE on first personal mention is correct.

**Worked examples.**

| Turn fragment | Candidates | Answer | Why |
|---|---|---|---|
| "I finished *The Hitchhiker's Guide to the Galaxy* yesterday" | (empty) | CREATE_NEW_ENTITY | a specific book, first mention |
| "the book was hilarious" (next turn) | `e_000: the hitchhiker's guide…` | LINK_EXISTING(e_000) | anaphora to the registered book |
| "I've been reading more *books* lately" | `e_000: …` | NON_ENTITY | generic plural, no specific referent |
| "my brother recommended it" | (no brother yet) | CREATE_NEW_ENTITY | a specific person, nameless but referable |
| "we might go to *the place* John mentioned" | `e_001: john` | DEFER | referent exists but is unresolvable until a later turn names it |

**Flagging.** Append `?your note` to any answer to flag an unclear item — the
note lands in the label file and feeds the guideline revision.
