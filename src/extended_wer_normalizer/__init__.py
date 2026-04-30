"""extended-wer-normalizer: jiwer-compatible WER normalization for voice AI."""

from __future__ import annotations

import jiwer

from .languages import get_language_data, supported_languages
from .transforms import (
    CollapseRepetitions,
    DigitWordsToChars,
    ExpandAbbreviations,
    ExpandDigitRuns,
    ExpandFrenchElisions,
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
    "supported_languages",
    "english_wer_pipeline",
    "german_wer_pipeline",
    "french_wer_pipeline",
    "CollapseRepetitions",
    "DigitWordsToChars",
    "ExpandAbbreviations",
    "ExpandDigitRuns",
    "ExpandFrenchElisions",
    "NormalizeCurrency",
    "NormalizeEmails",
    "NormalizeOrdinals",
    "NormalizePercentages",
    "NormalizeSymbols",
    "NormalizeURLs",
    "RemoveFillerWords",
]


def _build_pipeline(language: str) -> jiwer.Compose:
    """Compose a full WER-normalization pipeline for `language`."""
    # Validate the language registers (will raise if unsupported).
    get_language_data(language)

    steps: list = [
        # Pattern-specific normalizations run first, before punctuation is stripped.
        NormalizeEmails(language),
        NormalizeURLs(language),
        NormalizeSymbols(language),
        ExpandAbbreviations(language),
        NormalizeCurrency(language),
        NormalizePercentages(language),
        NormalizeOrdinals(language),
    ]

    # Language-specific contraction handling.
    if language == "en":
        steps.append(jiwer.ExpandCommonEnglishContractions())
    elif language == "fr":
        steps.append(ExpandFrenchElisions())
    # German has no separate contraction expansion step.

    steps.extend(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            # Digit normalization (after lowercase + punctuation removal).
            ExpandDigitRuns(),
            DigitWordsToChars(language),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            # Speech artifact cleanup.
            RemoveFillerWords(language),
            CollapseRepetitions(),
        ]
    )
    return jiwer.Compose(steps)


english_wer_pipeline = _build_pipeline("en")
german_wer_pipeline = _build_pipeline("de")
french_wer_pipeline = _build_pipeline("fr")

_PIPELINES: dict[str, jiwer.Compose] = {
    "en": english_wer_pipeline,
    "de": german_wer_pipeline,
    "fr": french_wer_pipeline,
}


def normalize_for_wer(text: str, language: str = "en") -> str:
    """Normalize text for WER comparison.

    Supported full-pipeline languages: "en", "de", "fr". Each runs the same
    structure (pattern normalization → lowercase → punctuation removal → digit
    normalization → cleanup) with language-specific lexicons and a custom
    contraction step (English contractions for "en", elision splitting for "fr",
    none for "de").

    For any other `language` value, applies a minimal language-agnostic pipeline:
    lowercase, punctuation removal, whitespace normalization. Useful as a
    fallback for languages that don't yet have a tuned data module.
    """
    if language in _PIPELINES:
        result = _PIPELINES[language]([text])
        return result[0].strip()
    pipeline = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    return pipeline([text])[0].strip()
