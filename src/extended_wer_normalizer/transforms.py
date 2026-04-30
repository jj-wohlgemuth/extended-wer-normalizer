"""jiwer-compatible AbstractTransform subclasses for WER normalization.

Every transform that needs language-specific data accepts `language="en"` and
reads its lexicon from `extended_wer_normalizer.languages.get_language_data`.
The English defaults preserve the package's pre-0.3 behavior bit-for-bit.
"""

from __future__ import annotations

import re

from jiwer import AbstractTransform

from .languages import get_language_data

# ---------------------------------------------------------------------------
# Number normalization
# ---------------------------------------------------------------------------


class ExpandDigitRuns(AbstractTransform):
    """Split runs of 2+ consecutive digits into space-separated single digits.

    Preserves leading zeros and handles phone-number sequences:
      "0176"  → "0 1 7 6"
      "5589"  → "5 5 8 9"
      "5.99"  → unchanged  (decimal part after "." is not split)

    Language-agnostic.
    """

    def process_string(self, s: str) -> str:
        # (?<![.,\d]) skips digit runs preceded by "." or "," (decimal separators
        # in either English or continental notation) or another digit.
        return re.sub(r"(?<![.,\d])\d{2,}", lambda m: " ".join(m.group()), s)


class DigitWordsToChars(AbstractTransform):
    """Convert isolated single-digit words to digit characters.

    Leaves digit words adjacent to compositional number words (twenty, hundred…)
    unchanged to avoid producing inconsistent mixed forms like "twenty 1".
    """

    def __init__(self, language: str = "en") -> None:
        data = get_language_data(language)
        self._digit_words = data.digit_words
        self._compositional = data.compositional_number_words

    def process_string(self, s: str) -> str:
        words = s.split()
        out: list[str] = []
        for i, word in enumerate(words):
            bare = word.lower().rstrip(".,!?;:")
            if bare in self._digit_words:
                left = words[i - 1].lower().rstrip(".,!?;:") if i > 0 else ""
                right = words[i + 1].lower().rstrip(".,!?;:") if i < len(words) - 1 else ""
                if left in self._compositional or right in self._compositional:
                    out.append(word)
                else:
                    punct = word[len(word.rstrip(".,!?;:")) :]
                    out.append(self._digit_words[bare] + punct)
            else:
                out.append(word)
        return " ".join(out)


# ---------------------------------------------------------------------------
# Email normalization
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)


class NormalizeEmails(AbstractTransform):
    """Convert email addresses to spoken-word form.

    English example: user.name+tag@sub.example.com
      → "user dot name plus tag at sub dot example dot com"
    """

    def __init__(self, language: str = "en") -> None:
        self._lex = get_language_data(language).email_lexicon

    def _expand(self, part: str) -> str:
        part = re.sub(r"\.", f" {self._lex['dot']} ", part)
        part = re.sub(r"-", f" {self._lex['dash']} ", part)
        part = re.sub(r"\+", f" {self._lex['plus']} ", part)
        part = re.sub(r"_", f" {self._lex['underscore']} ", part)
        return part

    def process_string(self, s: str) -> str:
        def _sub(match: re.Match[str]) -> str:
            email = match.group()
            local, domain = email.split("@", 1)
            return f"{self._expand(local)} {self._lex['at']} {self._expand(domain)}"

        return _EMAIL_RE.sub(_sub, s)


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://(?:www\.)?([A-Za-z0-9.\-]+)(?:/[^\s]*)?",
)


class NormalizeURLs(AbstractTransform):
    """Convert URLs to their spoken domain form, dropping scheme and path.

    English example: "https://www.example-site.com/some/path?q=1" → "example dash site dot com"
    """

    def __init__(self, language: str = "en") -> None:
        self._lex = get_language_data(language).url_lexicon

    def process_string(self, s: str) -> str:
        def _sub(match: re.Match[str]) -> str:
            domain = match.group(1)
            domain = re.sub(r"\.", f" {self._lex['dot']} ", domain)
            domain = re.sub(r"-", f" {self._lex['dash']} ", domain)
            return domain.strip()

        return _URL_RE.sub(_sub, s)


# ---------------------------------------------------------------------------
# Filler words & stutter/repetition
# ---------------------------------------------------------------------------

_MULTI_SPACE_RE = re.compile(r" {2,}")


class RemoveFillerWords(AbstractTransform):
    """Remove spoken filler words (language-specific patterns).

    Common in spontaneous speech and often transcribed by ASR models.
    """

    def __init__(self, language: str = "en") -> None:
        self._filler_re = get_language_data(language).filler_re

    def process_string(self, s: str) -> str:
        s = self._filler_re.sub("", s)
        return _MULTI_SPACE_RE.sub(" ", s).strip()


class CollapseRepetitions(AbstractTransform):
    """Collapse consecutive identical words to a single occurrence.

    Models ASR output stuttering: "I I I think" → "I think".
    Case-insensitive comparison; preserves the first occurrence's casing.
    Language-agnostic.
    """

    def process_string(self, s: str) -> str:
        words = s.split()
        if not words:
            return s
        out = [words[0]]
        for w in words[1:]:
            if w.lower() != out[-1].lower():
                out.append(w)
        return " ".join(out)


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------


class ExpandAbbreviations(AbstractTransform):
    """Expand common spoken abbreviations to their full form.

    English examples: "Dr. Smith" → "doctor Smith", "vs." → "versus"
    """

    def __init__(self, language: str = "en") -> None:
        data = get_language_data(language)
        self._abbreviations = data.abbreviations
        if data.abbreviations:
            self._re: re.Pattern[str] | None = re.compile(
                r"(?<!\w)(" + "|".join(re.escape(k) for k in data.abbreviations) + r")",
                re.IGNORECASE,
            )
        else:
            self._re = None

    def process_string(self, s: str) -> str:
        if self._re is None:
            return s
        return self._re.sub(lambda m: self._abbreviations[m.group().lower()], s)


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------


class NormalizeSymbols(AbstractTransform):
    """Replace common written symbols with their spoken equivalents.

    Only replaces symbols that appear as standalone tokens (surrounded by spaces
    or text boundaries) to avoid corrupting emails and URLs.
    """

    def __init__(self, language: str = "en") -> None:
        self._symbol_map = get_language_data(language).symbol_map

    def process_string(self, s: str) -> str:
        for symbol, word in self._symbol_map:
            s = s.replace(symbol, word)
        return s


# ---------------------------------------------------------------------------
# Currency normalization
# ---------------------------------------------------------------------------

# Accept either "." or "," as decimal separator so French "5,99 €" and
# English "$5.99" both parse. The integer-part regex captures one currency symbol
# and the digits to either side of an optional fractional separator.
_CURRENCY_RE = re.compile(
    r"([$€£¥₹])\s*(\d+)(?:[.,](\d{1,2}))?",
)


def _pluralize(unit: str, count: int, plurals: dict[str, str]) -> str:
    if count == 1:
        return unit
    return plurals.get(unit, unit + "s")


class NormalizeCurrency(AbstractTransform):
    """Convert currency amounts to spoken form.

    English examples:
      "$5"    → "five dollars"
      "$5.99" → "five dollars ninety nine cents"
      "€3.50" → "three euros fifty cents"
    """

    def __init__(self, language: str = "en") -> None:
        data = get_language_data(language)
        self._language = language
        self._symbols = data.currency_symbols
        self._plurals = data.currency_plurals

    def process_string(self, s: str) -> str:
        from num2words import num2words

        def n2w(n: int) -> str:
            return num2words(n, lang=self._language).replace("-", " ")

        def _sub(match: re.Match[str]) -> str:
            symbol = match.group(1)
            major = int(match.group(2))
            minor_str = match.group(3)
            unit_major, unit_minor = self._symbols.get(symbol, ("dollar", "cent"))

            parts = [f"{n2w(major)} {_pluralize(unit_major, major, self._plurals)}"]
            if minor_str:
                minor = int(minor_str.ljust(2, "0"))
                if minor > 0 and unit_minor:
                    parts.append(f"{n2w(minor)} {_pluralize(unit_minor, minor, self._plurals)}")
            return " ".join(parts)

        return _CURRENCY_RE.sub(_sub, s)


# ---------------------------------------------------------------------------
# Percentage normalization
# ---------------------------------------------------------------------------

# Accept either "." or "," as decimal separator (French uses comma).
_PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")


class NormalizePercentages(AbstractTransform):
    """Convert percentage notation to spoken form.

    English examples: "50%" → "fifty percent", "3.5%" → "three point five percent"
    French: "3,5%" → "trois virgule cinq pour cent"
    """

    def __init__(self, language: str = "en") -> None:
        self._language = language
        self._word = get_language_data(language).word_for_percent

    def process_string(self, s: str) -> str:
        from num2words import num2words

        def _replace(m: re.Match[str]) -> str:
            raw = m.group(1).replace(",", ".")
            val = float(raw)
            if val == int(val):
                spoken = num2words(int(val), lang=self._language)
            else:
                spoken = num2words(val, lang=self._language)
            return f"{spoken.replace('-', ' ')} {self._word}"

        return _PERCENT_RE.sub(_replace, s)


# ---------------------------------------------------------------------------
# Ordinal normalization
# ---------------------------------------------------------------------------


class NormalizeOrdinals(AbstractTransform):
    """Convert ordinal numerals to spoken word form.

    English: "1st" → "first", "15th" → "fifteenth"
    German:  "1. Januar" → "erste januar"
    French:  "1er" → "premier", "2e" → "deuxième"
    """

    def __init__(self, language: str = "en") -> None:
        self._language = language
        self._re = get_language_data(language).ordinal_re

    def process_string(self, s: str) -> str:
        from num2words import num2words

        def _replace(m: re.Match[str]) -> str:
            spoken = num2words(int(m.group(1)), to="ordinal", lang=self._language)
            return spoken.replace("-", " ")

        return self._re.sub(_replace, s)


# ---------------------------------------------------------------------------
# French elision contractions
# ---------------------------------------------------------------------------

# Matches the common French elision prefixes followed by an apostrophe
# (ASCII or typographic). Maps "j'aime" → "j aime", mirroring what
# jiwer.ExpandCommonEnglishContractions does for English contractions.
_FR_ELISIONS_RE = re.compile(
    r"\b(l|d|n|s|m|t|j|c|qu|jusqu|lorsqu|puisqu|quoiqu)['’]",
    re.IGNORECASE,
)


class ExpandFrenchElisions(AbstractTransform):
    """Expand French elision contractions: j'aime → j aime, l'eau → l eau.

    The contracted prefix is split off the following word with a space so that
    ASR output (which often expands or omits the apostrophe) aligns with the
    reference. Matches both ASCII (') and typographic (’) apostrophes.
    """

    def process_string(self, s: str) -> str:
        return _FR_ELISIONS_RE.sub(lambda m: m.group(1) + " ", s)
