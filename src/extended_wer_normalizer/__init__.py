"""extended-wer-normalizer: jiwer-compatible WER normalization for voice AI."""

from __future__ import annotations

import jiwer

from .transforms import (
    CollapseRepetitions,
    DigitWordsToChars,
    ExpandAbbreviations,
    ExpandDigitRuns,
    NormalizeCurrency,
    NormalizeEmails,
    NormalizeOrdinals,
    NormalizePercentages,
    NormalizeSymbols,
    NormalizeURLs,
    RemoveFillerWords,
)

__all__ = [
    "normalize_for_wer",
    "english_wer_pipeline",
    "CollapseRepetitions",
    "DigitWordsToChars",
    "ExpandAbbreviations",
    "ExpandDigitRuns",
    "NormalizeCurrency",
    "NormalizeEmails",
    "NormalizeOrdinals",
    "NormalizePercentages",
    "NormalizeSymbols",
    "NormalizeURLs",
    "RemoveFillerWords",
]

english_wer_pipeline = jiwer.Compose([
    # Pattern-specific normalizations run first, before punctuation is stripped
    NormalizeEmails(),                          # user@x.com → user at x dot com
    NormalizeURLs(),                            # https://x.com → x dot com
    NormalizeSymbols(),                         # & → and  (before RemovePunctuation drops &)
    ExpandAbbreviations(),                      # Dr. → doctor  (before punct removal)
    NormalizeCurrency(),                        # $5.99 → five dollars ninety nine cents
    NormalizePercentages(),                     # 50% → fifty percent
    NormalizeOrdinals(),                        # 1st → first
    # Core text normalization
    jiwer.ExpandCommonEnglishContractions(),    # I'm → i am
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    # Digit normalization
    ExpandDigitRuns(),                          # "0176" → "0 1 7 6"
    DigitWordsToChars(),                        # "zero one" → "0 1"
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    # Speech artifact cleanup
    RemoveFillerWords(),                        # um, uh, hmm, …
    CollapseRepetitions(),                      # I I I → I
])


def normalize_for_wer(text: str, language: str = "en") -> str:
    """Normalize text for WER comparison.

    English pipeline (in order):
      email → URL → symbol → abbreviation → currency → percentage → ordinal
      → contractions → lowercase → punctuation removal
      → digit-run expansion → digit-word-to-char
      → filler removal → repetition collapse

    All digit sequences collapse to space-separated single digits:
      "0176", "zero one seven six", "0 1 7 6" all become "0 1 7 6"

    For non-English text (language != "en"), applies only lowercase,
    punctuation removal, and whitespace normalization.
    """
    if language == "en":
        result = english_wer_pipeline([text])
        return result[0].strip()
    # Minimal language-agnostic normalization
    pipeline = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    return pipeline([text])[0].strip()
