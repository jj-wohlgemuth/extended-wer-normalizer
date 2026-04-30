"""Per-language data registry for transforms.

Every language plugs in a `LanguageData` instance describing the lexicons,
regexes, and lookup tables that the parameterized transforms in
`extended_wer_normalizer.transforms` consume. Currently registered: en, de, fr.
"""

from __future__ import annotations

from dataclasses import dataclass
from re import Pattern


@dataclass(frozen=True)
class LanguageData:
    """All language-specific data the transform classes need.

    Currency plural maps are explicit (no "+s" fallback) so each language can
    encode its own pluralization rules — e.g. German leaves most currency units
    invariant ("fünf Euro", not "fünf Euros").
    """

    code: str
    digit_words: dict[str, str]
    compositional_number_words: frozenset[str]
    filler_re: Pattern[str]
    abbreviations: dict[str, str]
    symbol_map: list[tuple[str, str]]
    email_lexicon: dict[str, str]
    url_lexicon: dict[str, str]
    currency_symbols: dict[str, tuple[str, str]]
    currency_plurals: dict[str, str]
    word_for_percent: str
    ordinal_re: Pattern[str]


_REGISTRY: dict[str, LanguageData] = {}


def register(data: LanguageData) -> None:
    _REGISTRY[data.code] = data


def get_language_data(lang: str) -> LanguageData:
    if lang not in _REGISTRY:
        raise ValueError(f"Unsupported language: {lang!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[lang]


def supported_languages() -> list[str]:
    return sorted(_REGISTRY)


# Trigger registration of the bundled languages.
from . import de, en, fr  # noqa: E402, F401
