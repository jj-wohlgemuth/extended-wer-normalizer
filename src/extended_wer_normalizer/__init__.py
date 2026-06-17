"""extended-wer-normalizer: jiwer-compatible WER normalization for voice AI."""

from __future__ import annotations

import jiwer

from .languages import supported_languages
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


DEFAULT_MAX_REPEATS = 3


def _build_pipeline(language: str, max_repeats: int = DEFAULT_MAX_REPEATS) -> jiwer.Compose:
    """Compose a WER-normalization pipeline for `language`.

    Supported languages ("en", "de", "fr") get the full structure. Any other
    language falls back to a minimal language-agnostic pipeline: lowercase,
    punctuation/whitespace cleanup, and repetition collapse.
    """
    if language not in supported_languages():
        return jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                CollapseRepetitions(max_repeats),
            ]
        )

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
            CollapseRepetitions(max_repeats),
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


def normalize_for_wer(
    text: str, language: str = "en", max_repeats: int = DEFAULT_MAX_REPEATS
) -> str:
    """Normalize text for WER comparison.

    Supported full-pipeline languages: "en", "de", "fr". Each runs the same
    structure (pattern normalization → lowercase → punctuation removal → digit
    normalization → cleanup) with language-specific lexicons and a custom
    contraction step (English contractions for "en", elision splitting for "fr",
    none for "de").

    For any other `language` value, applies a minimal language-agnostic pipeline:
    lowercase, punctuation removal, whitespace normalization, and repetition
    collapse. Useful as a fallback for languages that don't yet have a tuned
    data module.

    `max_repeats` controls the repetition-collapse threshold: a word must repeat
    more than this many times in a row before the run is collapsed to a single
    occurrence (default 3, i.e. runs of 4+ collapse). See `CollapseRepetitions`.
    """
    # Reuse the prebuilt pipelines for the default threshold; build a one-off
    # only when a custom max_repeats or an unsupported language is requested.
    if max_repeats == DEFAULT_MAX_REPEATS and language in _PIPELINES:
        pipeline = _PIPELINES[language]
    else:
        pipeline = _build_pipeline(language, max_repeats)
    return pipeline([text])[0].strip()
