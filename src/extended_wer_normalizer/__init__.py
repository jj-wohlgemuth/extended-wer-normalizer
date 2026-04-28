"""extended-wer-normalizer: jiwer-compatible WER normalization for voice AI."""

from __future__ import annotations

import jiwer

from .transforms import (
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
    WhisperBasicNormalize,
    WhisperEnglishNormalize,
)

__all__ = [
    "normalize_for_wer",
    "english_wer_pipeline",
    "CollapseRepetitions",
    "DigitWordsToChars",
    "ExpandAbbreviations",
    "ExpandDigitRuns",
    "FinalDigitWordCleanup",
    "NormalizeCurrency",
    "NormalizeEmails",
    "NormalizeOrdinals",
    "NormalizePercentages",
    "NormalizeSymbols",
    "NormalizeURLs",
    "RemoveFillerWords",
    "WhisperBasicNormalize",
    "WhisperEnglishNormalize",
]

english_wer_pipeline = jiwer.Compose([
    # Pre-WhisperEN: handle patterns WhisperEN would mangle or remove
    NormalizeEmails(),        # must be first: WhisperEN strips @ and dots in emails
    NormalizeURLs(),          # WhisperEN strips slashes and scheme prefixes
    NormalizeSymbols(),       # WhisperEN removes & and + as punctuation
    ExpandAbbreviations(),    # expand Dr./Mr. before WhisperEN lowercases
    # Core Whisper normalization (lowercase, punctuation, contractions, compound numbers)
    ExpandDigitRuns(),        # preserve leading zeros before WhisperEN sees e.g. "0176"
    DigitWordsToChars(),      # "zero one seven" → "0 1 7" before compound resolution
    WhisperEnglishNormalize(),
    # Post-WhisperEN: handle patterns WhisperEN preserves as-is or converts to compact form
    # (WhisperEN preserves $5.99, 50%, 1st; and converts spoken forms back to compact)
    ExpandDigitRuns(),        # split any remaining digit runs (e.g. from compound numbers)
    FinalDigitWordCleanup(),  # final digit-word sweep — must run before currency/% so
                              # it doesn't convert "nine" inside "ninety nine cents"
    NormalizeCurrency(),      # $5.99 → five dollars ninety nine cents
    NormalizePercentages(),   # 50% → fifty percent
    NormalizeOrdinals(),      # 1st → first
    RemoveFillerWords(),
    CollapseRepetitions(),
])


def normalize_for_wer(text: str, language: str = "en") -> str:
    """Normalize text for WER comparison.

    English pipeline applies, in order:
      email → URL → symbol → currency → percentage → ordinal → abbreviation
      → digit-run expansion → digit-word-to-char → Whisper EnglishTextNormalizer
      → digit-run expansion → final digit cleanup → filler removal → repetition collapse

    All number representations collapse to space-separated single digits, so
    "0176", "zero one seven six", and "21" all compare correctly.

    For non-English text, Whisper's BasicTextNormalizer is applied instead.
    """
    if language == "en":
        result = english_wer_pipeline([text])
        return result[0].strip()
    from whisper_normalizer.basic import BasicTextNormalizer
    return BasicTextNormalizer()(text).strip()
