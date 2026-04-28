"""Unit tests for extended-wer-normalizer.

Number normalization tests ported from aicpy/tests/test_wer_normalization.py.
Additional tests cover email, URL, filler, abbreviation, currency, ordinal,
percentage, and symbol normalizations.
"""

import pytest
import jiwer

from extended_wer_normalizer import normalize_for_wer, english_wer_pipeline
from extended_wer_normalizer.transforms import (
    CollapseRepetitions,
    DigitWordsToChars,
    ExpandAbbreviations,
    ExpandDigitRuns,
    FinalDigitWordCleanup,
    NormalizeCurrency,
    NormalizeEmails,
    NormalizeOrdinals,
    NormalizePercentages,
    NormalizeSymbols,
    NormalizeURLs,
    RemoveFillerWords,
    WhisperEnglishNormalize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wer(ref: str, hyp: str) -> float:
    return jiwer.wer(
        normalize_for_wer(ref),
        normalize_for_wer(hyp),
    )


# ===========================================================================
# NUMBER NORMALIZATION (ported from aicpy)
# ===========================================================================


@pytest.mark.parametrize(
    "a, b",
    [
        ("0176", "0 1 7 6"),
        ("0176", "zero one seven six"),
        ("0 1 7 6", "zero one seven six"),
        ("One Two Three Four Five", "1 2 3 4 5"),
        ("One. One. Two. Three.", "one one two three"),
        ("No. Yes. No. No. Yes.", "no yes no no yes"),
        ("0178 One. Two. Three.", "zero one seven eight one two three"),
        ("0178 One. Two. Three.", "0 1 7 8 1 2 3"),
        ("Please press 1 2 3 and confirm.", "please press one two three and confirm"),
        ("0 One 2 Three 4", "zero one two three four"),
    ],
)
def test_number_normalization_produces_equal_strings(a, b):
    assert normalize_for_wer(a) == normalize_for_wer(b), (
        f"\n  normalize_for_wer({a!r}) → {normalize_for_wer(a)!r}"
        f"\n  normalize_for_wer({b!r}) → {normalize_for_wer(b)!r}"
    )


@pytest.mark.parametrize(
    "ref, hyp",
    [
        ("0176. 4151. 5589", "zero one seven six four one five one five five eight nine"),
        ("0176. 4151. 5589", "0 1 7 6 4 1 5 1 5 5 8 9"),
        ("01715589", "zero one seven one five five eight nine"),
        ("1 2 3 4 5", "One Two Three Four Five"),
        ("One Two Three Four Five", "1 2 3 4 5"),
        ("No. Yes. No. No. Yes.", "no yes no no yes"),
        ("No. Yes. No. No. Yes.", "No Yes No No Yes"),
        (
            "One. One. Two. Three. Four. Five."
            " Six. Six. Five. Four. Three. Two. One."
            " Yes. Yes. No.",
            "one one two three four five six six five four three two one yes yes no",
        ),
        (
            "One. One. Two. Three. Four. Five."
            " Six. Six. Five. Four. Three. Two. One."
            " Yes. Yes. No.",
            "1 1 2 3 4 5 6 6 5 4 3 2 1 yes yes no",
        ),
        ("0178 One. Two. Three.", "zero one seven eight one two three"),
        ("Zero One Seven Eight 1 2 3", "zero one seven eight one two three"),
        ("Please press 1 2 3 and confirm.", "please press one two three and confirm"),
        ("0 One 2 Three 4", "zero one two three four"),
        (
            "0176 4151 5589 Yes. No. Yes.",
            "zero one seven six four one five one five five eight nine yes no yes",
        ),
        ("0178 One. Two. Three. Yes.", "0 1 7 8 1 2 3 yes"),
    ],
)
def test_equivalent_number_formats_have_zero_wer(ref, hyp):
    assert wer(ref, hyp) == 0.0, (
        f"\nWER={wer(ref, hyp):.4f} (expected 0.0)"
        f"\n  ref: {ref!r} → {normalize_for_wer(ref)!r}"
        f"\n  hyp: {hyp!r} → {normalize_for_wer(hyp)!r}"
    )


@pytest.mark.parametrize(
    "ref, hyp, description",
    [
        ("one two three", "one two four", "last digit wrong"),
        ("one two three four", "one two three", "missing last digit"),
        ("one two three", "one two three four", "extra digit at end"),
        ("one two three", "four five six", "all digits wrong"),
        (
            "0176 4151 5589",
            "zero one seven six four one five two five five eight nine",
            "one digit swapped",
        ),
        ("0178 One. Two. Three.", "zero one seven eight one two four", "last word-digit wrong"),
        ("0178 One. Two. Three.", "zero one seven nine one two three", "one digit in block wrong"),
    ],
)
def test_actual_errors_have_nonzero_wer(ref, hyp, description):
    score = wer(ref, hyp)
    assert score > 0.0, f"[{description}] Expected WER > 0 but got {score}"


@pytest.mark.parametrize(
    "text",
    [
        "0176. 4151. 5589",
        "zero one seven six four one five one five five eight nine",
        "One Two Three Four Five",
        "No. Yes. No. No. Yes.",
        "in language infinitely many words can be written with a small set of letters",
    ],
)
def test_number_normalization_is_idempotent(text):
    once = normalize_for_wer(text)
    twice = normalize_for_wer(once)
    assert once == twice, f"Not idempotent:\n  once:  {once!r}\n  twice: {twice!r}"


# ===========================================================================
# EMAIL NORMALIZATION
# ===========================================================================


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("user@example.com", "user at example dot com"),
        ("info@ai-coustics.com", "info at ai dash coustics dot com"),
        ("first.last@sub.domain.org", "first dot last at sub dot domain dot org"),
        ("user+tag@example.com", "user plus tag at example dot com"),
    ],
)
def test_email_normalization(input_text, expected_fragment):
    t = NormalizeEmails()
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


def test_email_zero_wer():
    ref = "contact us at info at example dot com"
    hyp = "contact us at info@example.com"
    assert wer(ref, hyp) == 0.0, f"WER={wer(ref, hyp)}"


def test_email_idempotent():
    t = NormalizeEmails()
    text = "user@example.com"
    once = t.process_string(text)
    twice = t.process_string(once)
    assert once == twice


# ===========================================================================
# URL NORMALIZATION
# ===========================================================================


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("https://example.com", "example dot com"),
        ("http://www.example.com/path?q=1", "example dot com"),
        ("visit https://my-site.org/page", "visit my dash site dot org"),
    ],
)
def test_url_normalization(input_text, expected_fragment):
    t = NormalizeURLs()
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


def test_url_zero_wer():
    ref = "visit example dot com"
    hyp = "visit https://example.com"
    assert wer(ref, hyp) == 0.0, f"WER={wer(ref, hyp)}"


# ===========================================================================
# FILLER WORD REMOVAL
# ===========================================================================


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("um hello", "hello"),
        ("uh yes please", "yes please"),
        ("hmm I think so", "I think so"),
        ("well um uh you know", "well you know"),
        ("mhm sure", "sure"),
    ],
)
def test_filler_removal(input_text, expected):
    t = RemoveFillerWords()
    result = t.process_string(input_text)
    assert result.strip() == expected, f"{input_text!r} → {result!r}"


def test_filler_zero_wer():
    ref = "yes please"
    hyp = "um yes uh please"
    assert wer(ref, hyp) == 0.0, f"WER={wer(ref, hyp)}"


# ===========================================================================
# STUTTER / REPETITION COLLAPSE
# ===========================================================================


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("I I I think", "I think"),
        ("the the cat", "the cat"),
        ("yes yes yes", "yes"),
        ("no no", "no"),
        ("hello world", "hello world"),  # no repetition — unchanged
    ],
)
def test_collapse_repetitions(input_text, expected):
    t = CollapseRepetitions()
    assert t.process_string(input_text) == expected


def test_stutter_zero_wer():
    ref = "I think so"
    hyp = "I I I think so"
    assert wer(ref, hyp) == 0.0, f"WER={wer(ref, hyp)}"


# ===========================================================================
# ABBREVIATION EXPANSION
# ===========================================================================


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("Dr. Smith", "doctor Smith"),
        ("Mr. Jones", "mister Jones"),
        ("Mrs. Brown", "missus Brown"),
        ("vs. the other team", "versus the other team"),
        ("etc.", "et cetera"),
        ("approx. five", "approximately five"),
    ],
)
def test_abbreviation_expansion(input_text, expected_fragment):
    t = ExpandAbbreviations()
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


def test_abbreviation_zero_wer():
    ref = "doctor smith"
    hyp = "Dr. Smith"
    assert wer(ref, hyp) == 0.0, f"WER={wer(ref, hyp)}"


# ===========================================================================
# CURRENCY NORMALIZATION
# ===========================================================================


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("$5", "five dollars"),
        ("$5.99", "five dollars ninety nine cents"),
        ("$1", "one dollar"),
        ("€3", "three euros"),
        ("£10.50", "ten pounds fifty pennies"),
    ],
)
def test_currency_normalization(input_text, expected_fragment):
    t = NormalizeCurrency()
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


def test_currency_zero_wer():
    ref = "it costs five dollars ninety nine cents"
    hyp = "it costs $5.99"
    assert wer(ref, hyp) == 0.0, f"WER={wer(ref, hyp)}"


# ===========================================================================
# PERCENTAGE NORMALIZATION
# ===========================================================================


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("50%", "fifty percent"),
        ("100%", "one hundred percent"),
        ("3%", "three percent"),
    ],
)
def test_percentage_normalization(input_text, expected):
    t = NormalizePercentages()
    result = t.process_string(input_text)
    assert expected in result, f"{input_text!r} → {result!r}"


def test_percentage_zero_wer():
    ref = "fifty percent accuracy"
    hyp = "50% accuracy"
    assert wer(ref, hyp) == 0.0, f"WER={wer(ref, hyp)}"


# ===========================================================================
# ORDINAL NORMALIZATION
# ===========================================================================


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("1st place", "first place"),
        ("2nd attempt", "second attempt"),
        ("3rd floor", "third floor"),
        ("15th anniversary", "fifteenth anniversary"),
    ],
)
def test_ordinal_normalization(input_text, expected_fragment):
    t = NormalizeOrdinals()
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


def test_ordinal_zero_wer():
    ref = "first place"
    hyp = "1st place"
    assert wer(ref, hyp) == 0.0, f"WER={wer(ref, hyp)}"


# ===========================================================================
# SYMBOL NORMALIZATION
# ===========================================================================


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("cats & dogs", "cats and dogs"),
        ("rock & roll", "rock and roll"),
    ],
)
def test_symbol_normalization(input_text, expected_fragment):
    t = NormalizeSymbols()
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


def test_symbol_zero_wer():
    ref = "cats and dogs"
    hyp = "cats & dogs"
    assert wer(ref, hyp) == 0.0, f"WER={wer(ref, hyp)}"


# ===========================================================================
# IDEMPOTENCE — all new transforms applied twice = applied once
# ===========================================================================


@pytest.mark.parametrize(
    "text",
    [
        "email me at user@example.com for $5.99 or 50% off",
        "1st place Dr. Smith with 100% at https://example.com",
        "um I I think it costs $3.50",
        "visit https://my-site.org or call us",
    ],
)
def test_full_pipeline_idempotent(text):
    once = normalize_for_wer(text)
    twice = normalize_for_wer(once)
    assert once == twice, f"Not idempotent:\n  once:  {once!r}\n  twice: {twice!r}"
