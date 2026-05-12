"""Unit tests for extended-wer-normalizer.

Number normalization tests ported from aicpy/tests/test_wer_normalization.py.
Additional tests cover email, URL, filler, abbreviation, currency, ordinal,
percentage, and symbol normalizations.
"""

import jiwer
import pytest

from extended_wer_normalizer import normalize_for_wer
from extended_wer_normalizer.transforms import (
    CollapseRepetitions,
    ExpandAbbreviations,
    NormalizeCurrency,
    NormalizeEmails,
    NormalizeOrdinals,
    NormalizePercentages,
    NormalizeSymbols,
    NormalizeURLs,
    RemoveFillerWords,
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
        # Compound spoken numbers must collapse to the same digit form as the
        # numeric representation (both end up split into space-separated digits
        # by ExpandDigitRuns).
        ("twenty one", "21"),
        ("thirty five", "35"),
        ("one hundred twenty three", "123"),
        ("two hundred twenty one", "221"),
        ("one thousand", "1000"),
        # Compound numbers embedded in sentences.
        ("I have twenty one apples", "I have 21 apples"),
        ("call extension thirty five please", "call extension 35 please"),
    ],
)
def test_number_normalization_produces_equal_strings(a, b):
    assert normalize_for_wer(a) == normalize_for_wer(b), (
        f"\n  normalize_for_wer({a!r}) → {normalize_for_wer(a)!r}"
        f"\n  normalize_for_wer({b!r}) → {normalize_for_wer(b)!r}"
    )


@pytest.mark.parametrize(
    "a, b",
    [
        # German compound numbers — single-word "<units>und<tens>".
        ("einundzwanzig", "21"),
        ("fünfunddreißig", "35"),
        ("einhundertdreiundzwanzig", "123"),
        ("zweihunderteinundzwanzig", "221"),
        ("eintausend", "1000"),
        ("vierundneunzig", "94"),
        ("zwölf", "12"),
        # Embedded in sentences.
        ("ich habe einundzwanzig Äpfel", "ich habe 21 Äpfel"),
    ],
)
def test_number_normalization_de_compound_equal_strings(a, b):
    assert normalize_for_wer(a, language="de") == normalize_for_wer(b, language="de"), (
        f"\n  normalize_for_wer({a!r}, 'de') → {normalize_for_wer(a, language='de')!r}"
        f"\n  normalize_for_wer({b!r}, 'de') → {normalize_for_wer(b, language='de')!r}"
    )


@pytest.mark.parametrize(
    "a, b",
    [
        # French compound numbers — `et`-joined, hyphenated, `quatre-vingt` multiplicative.
        ("vingt et un", "21"),
        ("vingt-et-un", "21"),
        ("trente-cinq", "35"),
        ("cent vingt-trois", "123"),
        ("deux cent vingt et un", "221"),
        ("mille", "1000"),
        ("quatre-vingt-quatorze", "94"),
        ("quatre-vingts", "80"),
        ("soixante-dix", "70"),
        ("douze", "12"),
        # Embedded in sentences.
        ("j'ai vingt et un ans", "j'ai 21 ans"),
    ],
)
def test_number_normalization_fr_compound_equal_strings(a, b):
    assert normalize_for_wer(a, language="fr") == normalize_for_wer(b, language="fr"), (
        f"\n  normalize_for_wer({a!r}, 'fr') → {normalize_for_wer(a, language='fr')!r}"
        f"\n  normalize_for_wer({b!r}, 'fr') → {normalize_for_wer(b, language='fr')!r}"
    )


@pytest.mark.parametrize(
    "ref, hyp",
    [
        (
            "0176. 4151. 5589",
            "zero one seven six four one five one five five eight nine",
        ),
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
        (
            "0178 One. Two. Three.",
            "zero one seven eight one two four",
            "last word-digit wrong",
        ),
        (
            "0178 One. Two. Three.",
            "zero one seven nine one two three",
            "one digit in block wrong",
        ),
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


# ===========================================================================
# EDGE CASES
# ===========================================================================


def test_empty_string():
    assert normalize_for_wer("") == ""


def test_whitespace_only():
    assert normalize_for_wer("   ") == ""


def test_single_digit_unchanged():
    from extended_wer_normalizer.transforms import ExpandDigitRuns

    t = ExpandDigitRuns()
    assert t.process_string("5") == "5"


def test_digit_run_already_separated():
    from extended_wer_normalizer.transforms import ExpandDigitRuns

    t = ExpandDigitRuns()
    assert t.process_string("0 1 7 6") == "0 1 7 6"


def test_decimal_digits_not_split():
    from extended_wer_normalizer.transforms import ExpandDigitRuns

    t = ExpandDigitRuns()
    assert t.process_string("5.99") == "5.99"
    assert t.process_string("3.14159") == "3.14159"


def test_digit_words_mixed_case():
    assert normalize_for_wer("ZERO ONE SEVEN SIX") == normalize_for_wer("0 1 7 6")


def test_digit_word_preserves_adjacent_punctuation():
    from extended_wer_normalizer.transforms import DigitWordsToChars

    t = DigitWordsToChars()
    assert t.process_string("one.") == "1."
    assert t.process_string("two,") == "2,"


def test_digit_words_to_chars_skips_compositional_neighbors():
    """`DigitWordsToChars` in isolation leaves digit-words next to a
    compositional number word alone. The pipeline-level conversion of
    "twenty one" → "21" is handled by `CompoundSpokenNumbersToDigits`.
    """
    from extended_wer_normalizer.transforms import DigitWordsToChars

    t = DigitWordsToChars()
    assert t.process_string("twenty one") == "twenty one"
    assert t.process_string("one hundred") == "one hundred"


def test_compound_spoken_numbers_to_digits_en():
    """Direct unit tests for `CompoundSpokenNumbersToDigits` in English."""
    from extended_wer_normalizer.transforms import CompoundSpokenNumbersToDigits

    t = CompoundSpokenNumbersToDigits("en")
    # Compositional spoken numbers → digit strings.
    assert t.process_string("twenty one") == "21"
    assert t.process_string("thirty five") == "35"
    assert t.process_string("one hundred twenty three") == "123"
    assert t.process_string("two hundred twenty one") == "221"
    assert t.process_string("one thousand") == "1000"
    assert t.process_string("one thousand two hundred") == "1200"
    # Solo teen / tens / scale words also convert.
    assert t.process_string("twelve") == "12"
    assert t.process_string("twenty") == "20"
    # Embedded in surrounding text.
    assert t.process_string("call ext twenty one now") == "call ext 21 now"
    # Trailing punctuation preserved.
    assert t.process_string("twenty one.") == "21."
    # Isolated single-digit words: left alone (handled downstream by DigitWordsToChars).
    assert t.process_string("one") == "one"


def test_compound_spoken_numbers_to_digits_de():
    """German compound numbers: `<units>und<tens>` is a single word."""
    from extended_wer_normalizer.transforms import CompoundSpokenNumbersToDigits

    t = CompoundSpokenNumbersToDigits("de")
    assert t.process_string("einundzwanzig") == "21"
    assert t.process_string("fünfunddreißig") == "35"
    assert t.process_string("einhundertdreiundzwanzig") == "123"
    assert t.process_string("zweihunderteinundzwanzig") == "221"
    assert t.process_string("eintausend") == "1000"
    assert t.process_string("vierundneunzig") == "94"
    assert t.process_string("zwölf") == "12"


def test_compound_spoken_numbers_to_digits_fr():
    """French: hyphenated and `et`-joined forms, `quatre-vingt` multiplication."""
    from extended_wer_normalizer.transforms import CompoundSpokenNumbersToDigits

    t = CompoundSpokenNumbersToDigits("fr")
    assert t.process_string("vingt et un") == "21"
    assert t.process_string("vingt-et-un") == "21"
    assert t.process_string("trente-cinq") == "35"
    assert t.process_string("cent vingt-trois") == "123"
    assert t.process_string("deux cent vingt et un") == "221"
    assert t.process_string("mille") == "1000"
    assert t.process_string("quatre-vingt-quatorze") == "94"
    assert t.process_string("quatre-vingts") == "80"
    assert t.process_string("douze") == "12"


def test_compound_spoken_numbers_to_digits_unsupported_lang_noop():
    """Languages outside text2num's set fall back to no-op."""
    from extended_wer_normalizer.transforms import CompoundSpokenNumbersToDigits

    # Build a transform for a supported language, then flip the language code
    # to simulate an unsupported one. Avoids polluting the global LanguageData
    # registry that other tests rely on.
    t = CompoundSpokenNumbersToDigits("en")
    t._code = "xx"  # not in text2num's supported set
    t._enabled = False
    assert t.process_string("twenty one") == "twenty one"
    assert t.process_string("einundzwanzig") == "einundzwanzig"


def test_email_no_match_unchanged():
    t = NormalizeEmails()
    assert t.process_string("hello world") == "hello world"


def test_url_no_match_unchanged():
    t = NormalizeURLs()
    assert t.process_string("visit the website") == "visit the website"


def test_filler_consecutive():
    t = RemoveFillerWords()
    assert t.process_string("um uh yeah") == "yeah"


def test_collapse_three_repetitions():
    from extended_wer_normalizer.transforms import CollapseRepetitions

    t = CollapseRepetitions()
    assert t.process_string("yes yes yes") == "yes"


def test_collapse_mixed_case_repetitions():
    from extended_wer_normalizer.transforms import CollapseRepetitions

    t = CollapseRepetitions()
    assert t.process_string("Yes yes YES") == "Yes"


def test_currency_zero_major():
    t = NormalizeCurrency()
    assert "zero" in t.process_string("$0")


def test_currency_no_minor_when_zero():
    t = NormalizeCurrency()
    result = t.process_string("$5.00")
    assert "five dollars" in result
    assert "cent" not in result


def test_ordinal_large():
    t = NormalizeOrdinals()
    result = t.process_string("100th anniversary")
    assert "hundredth" in result


def test_percentage_zero():
    t = NormalizePercentages()
    assert "zero percent" in t.process_string("0%")


def test_percentage_decimal():
    t = NormalizePercentages()
    result = t.process_string("3.5%")
    assert "percent" in result


# ===========================================================================
# NON-ENGLISH PATH
# ===========================================================================


@pytest.mark.parametrize(
    "text, language, expected_fragment",
    [
        ("Bonjour monde!", "fr", "bonjour monde"),
        ("Das ist gut.", "de", "das ist gut"),
        ("HELLO WORLD", "es", "hello world"),
    ],
)
def test_non_english_lowercases_and_strips_punctuation(text, language, expected_fragment):
    assert normalize_for_wer(text, language=language) == expected_fragment


# ===========================================================================
# JIWER PIPELINE INTEGRATION
# ===========================================================================


def test_english_pipeline_usable_with_jiwer_wer():
    """english_wer_pipeline + ReduceToListOfListOfWords can be used with jiwer.wer."""
    from extended_wer_normalizer import english_wer_pipeline

    # english_wer_pipeline produces strings; append word-reduction for jiwer.wer
    complete = jiwer.Compose([english_wer_pipeline, jiwer.ReduceToListOfListOfWords()])
    score = jiwer.wer(
        "0176",
        "zero one seven six",
        reference_transform=complete,
        hypothesis_transform=complete,
    )
    assert score == 0.0


def test_transforms_composable_with_jiwer_builtins():
    """Individual transforms can be freely mixed with jiwer built-ins."""
    from extended_wer_normalizer.transforms import ExpandDigitRuns

    pipeline = jiwer.Compose(
        [
            ExpandDigitRuns(),
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    score = jiwer.wer(
        "0 1 7 6", "0176", reference_transform=pipeline, hypothesis_transform=pipeline
    )
    assert score == 0.0


# ===========================================================================
# LANGUAGE REGISTRY
# ===========================================================================


def test_supported_languages_registry():
    from extended_wer_normalizer import supported_languages

    assert supported_languages() == ["de", "en", "fr"]


def test_unsupported_language_for_full_pipeline_falls_back_to_minimal():
    # Spanish has no full pipeline; minimal lowercase + punctuation removal applies.
    assert normalize_for_wer("HOLA, ¿qué tal?", language="es") == "hola qué tal"


def test_get_language_data_raises_on_unknown_code():
    from extended_wer_normalizer.languages import get_language_data

    with pytest.raises(ValueError, match="Unsupported language"):
        get_language_data("xx")


# ===========================================================================
# GERMAN — full pipeline
# ===========================================================================


def wer_de(ref: str, hyp: str) -> float:
    return jiwer.wer(
        normalize_for_wer(ref, language="de"),
        normalize_for_wer(hyp, language="de"),
    )


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("Hr. Müller", "herr müller"),
        ("Fr. Schmidt", "frau schmidt"),
        ("Dr. Schulz", "doktor schulz"),
        ("z.B. das Auto", "zum beispiel das auto"),
        ("usw.", "und so weiter"),
        ("ca. fünf", "circa fünf"),
        ("d.h. niemand", "das heißt niemand"),
        ("Str. 5", "straße 5"),
    ],
)
def test_german_abbreviation_expansion(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import ExpandAbbreviations

    t = ExpandAbbreviations(language="de")
    result = t.process_string(input_text)
    assert expected_fragment in result.lower(), f"{input_text!r} → {result!r}"


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("max@beispiel.de", "max at beispiel punkt de"),
        ("info@ai-coustics.de", "info at ai bindestrich coustics punkt de"),
    ],
)
def test_german_email_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeEmails

    t = NormalizeEmails(language="de")
    assert expected_fragment in t.process_string(input_text)


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("https://www.beispiel.de/x", "beispiel punkt de"),
        ("https://my-site.de", "my bindestrich site punkt de"),
    ],
)
def test_german_url_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeURLs

    t = NormalizeURLs(language="de")
    assert expected_fragment in t.process_string(input_text)


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("ähm hallo", "hallo"),
        ("äh ja bitte", "ja bitte"),
        ("hmm ich denke", "ich denke"),
        ("tja schade", "schade"),
        ("also gut", "gut"),
    ],
)
def test_german_filler_removal(input_text, expected):
    from extended_wer_normalizer.transforms import RemoveFillerWords

    t = RemoveFillerWords(language="de")
    assert t.process_string(input_text).strip() == expected


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("€5", "fünf euro"),
        ("€3", "drei euro"),
        ("$5", "fünf dollar"),
    ],
)
def test_german_currency_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeCurrency

    t = NormalizeCurrency(language="de")
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


def test_german_currency_no_plural_inflection():
    """German currency units don't pluralize (Euro stays Euro)."""
    from extended_wer_normalizer.transforms import NormalizeCurrency

    t = NormalizeCurrency(language="de")
    assert "euros" not in t.process_string("€5")
    assert "dollars" not in t.process_string("$5")


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("50%", "fünfzig prozent"),
        ("100%", "einhundert prozent"),
        ("3,5%", "prozent"),  # comma-decimal accepted
        ("3.5%", "prozent"),  # period-decimal also accepted
    ],
)
def test_german_percentage_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizePercentages

    t = NormalizePercentages(language="de")
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("am 1. Januar treffen", "am erste Januar treffen"),
        ("am 31. Dezember", "am einunddreißigste Dezember"),
        ("100. Geburtstag", "hundertste Geburtstag"),
    ],
)
def test_german_ordinal_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeOrdinals

    t = NormalizeOrdinals(language="de")
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


def test_german_ordinal_does_not_match_decimals():
    from extended_wer_normalizer.transforms import NormalizeOrdinals

    t = NormalizeOrdinals(language="de")
    # "1.5" is a decimal, not an ordinal — must stay untouched.
    assert t.process_string("Wert: 1.5 Liter") == "Wert: 1.5 Liter"


def test_german_ordinal_does_not_match_sentence_end():
    from extended_wer_normalizer.transforms import NormalizeOrdinals

    t = NormalizeOrdinals(language="de")
    # Period followed by end-of-string is not an ordinal.
    assert t.process_string("Es war 5.") == "Es war 5."


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("Katzen & Hunde", "katzen und hunde"),
        ("rock & roll", "rock und roll"),
    ],
)
def test_german_symbol_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeSymbols

    t = NormalizeSymbols(language="de")
    assert expected_fragment in t.process_string(input_text).lower()


@pytest.mark.parametrize(
    "ref, hyp",
    [
        ("Hr. Müller", "herr müller"),
        ("Es kostet €5", "es kostet fünf euro"),
        ("50% Rabatt", "fünfzig prozent rabatt"),
        ("ähm ja", "ja"),
        ("ich ich denke", "ich denke"),
        ("am 1. Januar", "am erste Januar"),
    ],
)
def test_german_zero_wer(ref, hyp):
    score = wer_de(ref, hyp)
    assert score == 0.0, (
        f"WER={score} (expected 0.0)\n"
        f"  ref: {ref!r} → {normalize_for_wer(ref, language='de')!r}\n"
        f"  hyp: {hyp!r} → {normalize_for_wer(hyp, language='de')!r}"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Hr. Müller, am 1. Januar, ähm, ungefähr 50% Rabatt.",
        "Schreib an info@beispiel.de oder besuche https://beispiel.de",
        "Das kostet €5 und €3,50",
    ],
)
def test_german_pipeline_idempotent(text):
    once = normalize_for_wer(text, language="de")
    twice = normalize_for_wer(once, language="de")
    assert once == twice, f"Not idempotent:\n  once:  {once!r}\n  twice: {twice!r}"


# ===========================================================================
# FRENCH — full pipeline
# ===========================================================================


def wer_fr(ref: str, hyp: str) -> float:
    return jiwer.wer(
        normalize_for_wer(ref, language="fr"),
        normalize_for_wer(hyp, language="fr"),
    )


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("M. Dupont", "monsieur dupont"),
        ("Mme Dubois", "madame dubois"),
        ("Mlle Martin", "mademoiselle martin"),
        ("Dr. Petit", "docteur petit"),
        ("Pr. Lefèvre", "professeur lefèvre"),
        ("etc.", "et cetera"),
        ("env. cinq", "environ cinq"),
        ("av. de la République", "avenue de la république"),
    ],
)
def test_french_abbreviation_expansion(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import ExpandAbbreviations

    t = ExpandAbbreviations(language="fr")
    result = t.process_string(input_text)
    assert expected_fragment in result.lower(), f"{input_text!r} → {result!r}"


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("max@exemple.fr", "max arobase exemple point fr"),
        ("info@mon-site.fr", "info arobase mon tiret site point fr"),
    ],
)
def test_french_email_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeEmails

    t = NormalizeEmails(language="fr")
    assert expected_fragment in t.process_string(input_text)


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("https://exemple.fr", "exemple point fr"),
        ("https://exemple-site.fr/page", "exemple tiret site point fr"),
    ],
)
def test_french_url_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeURLs

    t = NormalizeURLs(language="fr")
    assert expected_fragment in t.process_string(input_text)


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("euh bonjour", "bonjour"),
        ("ben oui", "oui"),
        ("hum d'accord", "d'accord"),
        ("alors voilà", "voilà"),
    ],
)
def test_french_filler_removal(input_text, expected):
    from extended_wer_normalizer.transforms import RemoveFillerWords

    t = RemoveFillerWords(language="fr")
    assert t.process_string(input_text).strip() == expected


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("€5", "cinq euros"),
        ("€1", "un euro"),
        ("€3.50", "cinquante centimes"),  # period also accepted
        ("€3,50", "cinquante centimes"),  # French comma-decimal
        ("$5", "cinq dollars"),
    ],
)
def test_french_currency_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeCurrency

    t = NormalizeCurrency(language="fr")
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("50%", "cinquante pour cent"),
        ("100%", "cent pour cent"),
        ("3,5%", "pour cent"),
        ("3.5%", "pour cent"),
    ],
)
def test_french_percentage_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizePercentages

    t = NormalizePercentages(language="fr")
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("1er janvier", "premier janvier"),
        ("2e étage", "deuxième étage"),
        ("15e siècle", "quinzième siècle"),
        ("100e fois", "centième fois"),
    ],
)
def test_french_ordinal_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeOrdinals

    t = NormalizeOrdinals(language="fr")
    result = t.process_string(input_text)
    assert expected_fragment in result, f"{input_text!r} → {result!r}"


@pytest.mark.parametrize(
    "input_text, expected_fragment",
    [
        ("chats & chiens", "chats et chiens"),
        ("noir & blanc", "noir et blanc"),
    ],
)
def test_french_symbol_normalization(input_text, expected_fragment):
    from extended_wer_normalizer.transforms import NormalizeSymbols

    t = NormalizeSymbols(language="fr")
    assert expected_fragment in t.process_string(input_text).lower()


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("j'aime", "j aime"),
        ("l'eau", "l eau"),
        ("d'accord", "d accord"),
        ("qu'il pense", "qu il pense"),
        ("jusqu'ici", "jusqu ici"),
        # Typographic apostrophe also handled
        ("c’est ça", "c est ça"),
    ],
)
def test_french_elision_expansion(input_text, expected):
    from extended_wer_normalizer.transforms import ExpandFrenchElisions

    t = ExpandFrenchElisions()
    assert t.process_string(input_text) == expected


@pytest.mark.parametrize(
    "ref, hyp",
    [
        ("M. Dupont", "monsieur dupont"),
        ("Il coûte €5", "il coûte cinq euros"),
        ("50% de réduction", "cinquante pour cent de réduction"),
        ("euh oui", "oui"),
        ("oui oui", "oui"),
        ("le 1er janvier", "le premier janvier"),
        # Elision drops the apostrophe → matches written form expanded with space
        ("j'aime ça", "j aime ça"),
    ],
)
def test_french_zero_wer(ref, hyp):
    score = wer_fr(ref, hyp)
    assert score == 0.0, (
        f"WER={score} (expected 0.0)\n"
        f"  ref: {ref!r} → {normalize_for_wer(ref, language='fr')!r}\n"
        f"  hyp: {hyp!r} → {normalize_for_wer(hyp, language='fr')!r}"
    )


@pytest.mark.parametrize(
    "text",
    [
        "M. Dupont, le 1er janvier, euh, environ 50% de réduction.",
        "Écrivez à info@exemple.fr ou visitez https://exemple.fr",
        "C'est €5,99 pour deux",
    ],
)
def test_french_pipeline_idempotent(text):
    once = normalize_for_wer(text, language="fr")
    twice = normalize_for_wer(once, language="fr")
    assert once == twice, f"Not idempotent:\n  once:  {once!r}\n  twice: {twice!r}"


# ===========================================================================
# CROSS-LANGUAGE — full pipelines all idempotent on the same input shape
# ===========================================================================


@pytest.mark.parametrize("language", ["en", "de", "fr"])
def test_full_pipeline_idempotent_per_language(language):
    samples = {
        "en": "Dr. Smith's 1st visit cost $5.99 — um, around 50% off.",
        "de": "Hr. Müller, am 1. Januar, ähm, ungefähr 50% Rabatt.",
        "fr": "M. Dupont, le 1er janvier, euh, environ 50% de réduction.",
    }
    text = samples[language]
    once = normalize_for_wer(text, language=language)
    twice = normalize_for_wer(once, language=language)
    assert once == twice, f"[{language}] not idempotent:\n  once:  {once!r}\n  twice: {twice!r}"


@pytest.mark.parametrize("language", ["en", "de", "fr"])
def test_pipeline_attached_to_module_export(language):
    """The module-level pipelines are exposed and runnable."""
    from extended_wer_normalizer import (
        english_wer_pipeline,
        french_wer_pipeline,
        german_wer_pipeline,
    )

    pipelines = {
        "en": english_wer_pipeline,
        "de": german_wer_pipeline,
        "fr": french_wer_pipeline,
    }
    out = pipelines[language](["Hello world"])
    assert isinstance(out, list) and len(out) == 1
