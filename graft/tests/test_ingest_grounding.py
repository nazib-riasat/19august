"""The grounding ladder (P5.4, gap G5) — exit criteria 1, 7.

The ladder is the part of Phase 5 that Phase 6 inherits most directly: it trains
on these offsets.  So the tests are about *where* a span lands, not merely that
one was found, and the mis-bound case the spike actually measured is a test
rather than a note.
"""

from __future__ import annotations

from graft.ingest.grounding import (
    RUNG_EXACT,
    RUNG_FUZZY,
    RUNG_NORMALISED,
    ground,
    ground_assertion_quotes,
    normalise,
    rung_counts,
)
from graft.ingest.records import Quote


# -- rung 1 -----------------------------------------------------------------


def test_exact_match_is_rung_one_and_selects_the_quote():
    text = "I moved to Yokohama in March and I love the harbour."
    hit = ground("moved to Yokohama", text)
    assert hit is not None
    assert hit.rung == RUNG_EXACT
    assert text[hit.start : hit.end] == "moved to Yokohama"


def test_the_first_occurrence_wins():
    """Deterministic, and stated: a later occurrence would give the same text a
    different span id, so two runs must not be free to choose differently."""
    text = "cats. dogs. cats."
    hit = ground("cats", text)
    assert hit is not None and hit.start == 0


# -- rung 2 -----------------------------------------------------------------


def test_case_and_whitespace_differences_are_rung_two_with_original_offsets():
    text = "She said:  I  visited   Sankeien Garden last spring."
    hit = ground("i visited sankeien garden", text)
    assert hit is not None
    assert hit.rung == RUNG_NORMALISED
    # The offsets index the RAW turn, not the normalised string — this is what
    # exit criterion 1 (spans resolve on replay) actually depends on.
    assert text[hit.start : hit.end] == "I  visited   Sankeien Garden"


def test_normalise_maps_every_offset_back():
    text = "  A  b\tc  "
    norm, index = normalise(text)
    assert norm == "a b c"
    assert len(index) == len(norm) + 1
    for i, ch in enumerate(norm):
        if ch != " ":
            assert text[index[i]].lower() == ch


def test_a_normalised_match_does_not_keep_trailing_whitespace():
    text = "he left Tokyo   yesterday"
    hit = ground("LEFT TOKYO", text)
    assert hit is not None
    assert text[hit.start : hit.end] == "left Tokyo"


# -- rung 3, and the measured defect ---------------------------------------


def test_a_small_typo_recovers_at_rung_three():
    text = "I bought a bicycle for 240 euros last month."
    hit = ground("bought a bycicle for 240 euros", text)
    assert hit is not None
    assert hit.rung == RUNG_FUZZY
    assert "bicycle for 240 euros" in text[hit.start : hit.end]


def test_the_spike_mis_bound_case_no_longer_glues_a_partial_word():
    """The measured G5 defect: `d1_0022` came back as ``"s Sankeien Garden"``.

    The window landed one character early and swallowed the tail of the previous
    word.  Boundary snapping generates the candidate windows and keeps whichever
    best fits the quote, which drops the orphaned ``"s "`` — and, because the
    choice is made by score rather than by rule, cannot damage a quote that
    genuinely starts mid-word (the next test).
    """
    text = "We wandered the gardens Sankeien Garden is famous for."
    hit = ground("Sankeien Gardens", text)
    assert hit is not None
    found = text[hit.start : hit.end]
    assert not found.startswith("s "), found
    assert found.startswith("Sankeien"), found


def test_a_genuinely_mid_word_quote_is_not_snapped_outward():
    """Snapping is chosen by fit, so a legitimate partial word survives it."""
    text = "The word Sankeien appears here."
    hit = ground("ankeie", text)
    assert hit is not None
    assert text[hit.start : hit.end] == "ankeie"


def test_a_quote_that_is_not_in_the_turn_fails_rather_than_landing_somewhere():
    text = "I had lunch with Priya on Tuesday."
    assert ground("the quarterly revenue forecast was revised upward", text) is None


def test_an_empty_quote_never_grounds():
    assert ground("", "anything at all") is None


# -- multi-span, cross-turn (G9) -------------------------------------------


def test_every_quote_must_ground_or_the_assertion_is_dropped():
    """All-or-nothing (G9): a multi-span claim with one span lost is a claim with
    unrecorded provenance, and plan §3.1 requires *every* supporting span."""
    offsets = {0: ("t1", "It costs 40 euros."), -1: ("t0", "I asked about the pass.")}
    ok = ground_assertion_quotes(
        [Quote(0, "costs 40 euros"), Quote(-1, "the pass")], "t1", offsets
    )
    assert ok is not None and len(ok) == 2

    bad = ground_assertion_quotes(
        [Quote(0, "costs 40 euros"), Quote(-1, "a completely unrelated sentence")],
        "t1",
        offsets,
    )
    assert bad is None


def test_a_quote_naming_a_turn_outside_the_window_fails_closed():
    """Silently re-homing it to the current turn would attach provenance to a
    turn that does not contain it."""
    offsets = {0: ("t1", "It costs 40 euros.")}
    assert ground_assertion_quotes([Quote(-4, "costs 40 euros")], "t1", offsets) is None


def test_spans_come_back_in_turn_order():
    offsets = {0: ("t1", "and it was cheap"), -1: ("t0", "I bought the pass")}
    got = ground_assertion_quotes(
        [Quote(0, "it was cheap"), Quote(-1, "bought the pass")], "t1", offsets
    )
    assert got is not None
    assert [q.turn_id for q in got] == ["t0", "t1"]


def test_the_stored_text_is_the_turns_not_the_models():
    """On rungs 2 and 3 they differ, and storing the model's string beside the
    turn's offsets would make provenance a fiction."""
    offsets = {0: ("t1", "I  visited   Sankeien Garden.")}
    got = ground_assertion_quotes([Quote(0, "i visited sankeien garden")], "t1", offsets)
    assert got is not None
    assert got[0].text == "I  visited   Sankeien Garden"


# -- reporting --------------------------------------------------------------


def test_rung_counts_reports_every_rung_including_zeros():
    """'No fuzzy spans' and 'the fuzzy count was never computed' must not look
    the same in a report."""
    counts = rung_counts([])
    assert set(counts) == {"exact", "normalised", "fuzzy", "failed"}
    assert all(v == 0 for v in counts.values())


# -- the 13-14 Aug 2026 audit's regressions ----------------------------------


def test_normalise_survives_lowercase_expanding_unicode():
    """'İ' (U+0130) lowers to two characters.  The old map appended the pair as
    one element against one index entry, shifting every offset after it — 13
    turns in the pinned corpus contain such characters, and a quote reaching the
    end of the turn crashed with IndexError."""
    text = "Trip to İstanbul  Was GREAT fun overall"
    norm, index = normalise(text)
    assert len(index) == len(norm) + 1  # the documented invariant, restored

    hit = ground("was great fun", text)
    assert hit is not None
    assert text[hit.start : hit.end] == "Was GREAT fun"

    # A normalised match running to the very end of the turn must not raise.
    tail = ground("great fun overall", text)
    assert tail is not None
    assert text[tail.start : tail.end] == "GREAT fun overall"


def test_a_whitespace_only_quote_never_grounds():
    """It would 'match' any blank run at rung 1 with score 1.0 and store a blank
    span as provenance — the all-or-nothing rule satisfied vacuously."""
    assert ground("   ", "some text with   spaces inside") is None
    assert ground("\n\t", "anything at all") is None


def test_the_fuzzy_refinement_is_not_locked_to_a_single_coarse_winner():
    """The measured decoy case: two near-identical sentences differing in a
    value.  Refining only around the single coarse best bound a one-typo quote
    of the *forty*-euro sentence to the *ninety*-euro sentence at ratio 0.95 —
    provenance saying the evidence states a different amount."""
    decoy = "I paid ninety euros for the day pass at the gate yesterday morning after we arrived"
    true = "I paid forty euros for the day pass at the gate yesterday morning after we arrived"
    text = decoy + " " + true
    quote = "I paid fourty euros for the day pass at the gate yesterday morning after we arrived"

    hit = ground(quote, text)
    assert hit is not None
    assert "forty" in text[hit.start : hit.end]
    assert "ninety" not in text[hit.start : hit.end]


def test_a_fuzzy_window_never_keeps_a_blank_edge():
    """Two windows tying on ratio can differ only by an edge blank; the stored
    span is evidence text, not padding (the rung-2 trailing trim, both edges)."""
    text = "We wandered the gardens Sankeien Garden is famous for."
    hit = ground("Sankeien Gardens", text)
    assert hit is not None
    got = text[hit.start : hit.end]
    assert got == got.strip()


def test_a_rung_2_match_ending_inside_a_case_expansion_keeps_the_whole_character():
    """The offset map is non-decreasing, not strictly increasing: one original
    character can produce several normalised ones.  Taking the end from
    ``index[j + n]`` lands on that character's *start*, storing a span one
    character short — or of zero width — at score 1.0.  (Introduced by the fix
    for the İ crash and caught by the second audit pass.)"""
    text = "Ankara İ. Yilmaz spoke"
    hit = ground("ankara i", text)
    assert hit is not None
    assert hit.rung == "normalised"
    assert text[hit.start : hit.end] == "Ankara İ"

    # The degenerate case: the match is exactly the expanding character.
    single = ground("i", "İstanbul")
    assert single is None or single.end > single.start
