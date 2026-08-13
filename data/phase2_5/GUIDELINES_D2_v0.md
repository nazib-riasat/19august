# D2 annotation guidelines v0 — claim-pair relation

**The item.** Two extracted claims, A and B, each shown with its **session
date**. B is always the one from the *later* (or equal) session. Decide the
one relation that holds — the four are mutually exclusive by design (research
plan v1.2 §3.2).

**The four relations, with the decision procedure:**

Ask, in order:

1. *Do A and B talk about the same subject-slot of the same entity?*
   No → **INDEPENDENT**. (Different facts about one entity are still
   INDEPENDENT: "Ana lives in Lisbon" / "Ana has a dog".)
2. *Do they assert the same value?* Yes, same fact in different words →
   **DUPLICATE**. Paraphrase, rounding ("25:50" vs "about 26 minutes"), and
   granularity differences that a reader would call "the same answer" are
   DUPLICATE.
3. *Do they assert incompatible values with a temporal update reading?*
   The later claim **replaces** the earlier one as the current state (moved
   cities, new personal best, changed job) → **SUPERSEDES** (B supersedes A).
   The dates matter: an update reading needs B's session to be later.
4. *Do they assert incompatible values with no coherent update reading?*
   Same time, both claimed true, cannot both hold → **CONFLICT**. Also
   CONFLICT when the incompatibility is at the *same* date, or when the order
   is wrong for an update (the later session asserts the stale value).

**Boundary cases.**
- Additive change ("I ran 5k" then "I ran 10k" as *events*) is INDEPENDENT —
  two events, no replacement. Supersession applies to *states*, not events.
- A correction inside one session ("actually, it was Tuesday") is SUPERSEDES.
- If either claim is too garbled to interpret, flag it (`?note`) and pick the
  best reading rather than skipping — the flag rate is itself a measurement.

**Worked examples.**

| A (earlier) | B (later) | Answer | Why |
|---|---|---|---|
| "My personal best 5K is 27:10" (May 3) | "New PB today — 25:50!" (Jun 9) | SUPERSEDES | same slot (user's 5K PB), replaced state |
| "I live in Austin" (Jan 2) | "I live in Austin, Texas" (Mar 8) | DUPLICATE | same value, more precise wording |
| "My sister is a nurse" (Feb 1) | "My sister works nights at the hospital" (Feb 20) | INDEPENDENT | related subject, different fact slots |
| "The meeting is on Thursday" (Apr 4) | "The meeting is on Friday" (Apr 4, same session) | CONFLICT | incompatible, no later-state reading |

**Flagging.** Append `?your note` to any answer to flag an unclear item.
