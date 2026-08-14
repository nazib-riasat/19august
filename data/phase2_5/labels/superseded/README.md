# Superseded labels — kept as a record, not as training data

`d1_labels_Sabbir_spikebatch.jsonl` — 34 D1 labels against `d1_items.jsonl`
(the Phase-2.5 spike batch), 14 Aug 2026.

**Why it is not usable.** The spike batch's candidate lists carry *synthetic*
entity ids (`e_08e075c7_000`), invented by the spike's own item builder. The
real graph keys entities by content hash (`node_id_of("Entity", …)`), and
**0 of the batch's 19 candidate ids exist as real nodes**. So the 9
`LINK_EXISTING` labels in this file point at a dead namespace: they cannot be
resolved to an entity, and a D1 head trained on them would learn that linking
never resolves — gap G4's failure, arriving from the label side.

The other 25 labels (CREATE_NEW_ENTITY / NON_ENTITY / DEFER) are sound but
cannot be used alone: a batch that teaches only the non-link actions trains a
head that never links.

**Superseded by** `d1_labels_Sabbir_pilot.jsonl`, annotated against
`d1_items_pilot.jsonl` — 40 items derived from the live-pilot log with real,
incrementally-accumulated candidate lists (every item has one; up to 10 each).

Kept rather than deleted because the timing data is real and the episode is
worth being able to point at.
