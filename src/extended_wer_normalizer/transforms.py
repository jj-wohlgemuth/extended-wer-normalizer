"""jiwer-compatible AbstractTransform subclasses for WER normalization."""

from __future__ import annotations

import re

from jiwer import AbstractTransform

# ---------------------------------------------------------------------------
# Number normalization helpers
# ---------------------------------------------------------------------------

_DIGIT_WORDS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

# Words that form part of compound numbers ("twenty one", "one hundred …").
# A single-digit word adjacent to any of these is left as-is to avoid producing
# mixed forms like "twenty 1".
_COMPOSITIONAL_NUMBER_WORDS: frozenset[str] = frozenset(
    {
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
    }
)


class ExpandDigitRuns(AbstractTransform):
    """Split runs of 2+ consecutive digits into space-separated single digits.

    Preserves leading zeros and handles phone-number sequences:
      "0176"  → "0 1 7 6"
      "5589"  → "5 5 8 9"
      "5.99"  → unchanged  (decimal part after "." is not split)
    """

    def process_string(self, s: str) -> str:
        # (?<![.\d]) skips digit runs preceded by "." or another digit,
        # which protects ALL decimal digits in numbers like "5.99" and "3.14159".
        # Currency, % and ordinals are already converted to words upstream.
        return re.sub(r"(?<![.\d])\d{2,}", lambda m: " ".join(m.group()), s)


class DigitWordsToChars(AbstractTransform):
    """Convert isolated single-digit words (zero–nine) to digit characters.

    Leaves digit words adjacent to compositional number words (twenty, hundred…)
    unchanged to avoid producing inconsistent mixed forms like "twenty 1".
    """

    def process_string(self, s: str) -> str:
        words = s.split()
        out: list[str] = []
        for i, word in enumerate(words):
            bare = word.lower().rstrip(".,!?;:")
            if bare in _DIGIT_WORDS:
                left = words[i - 1].lower().rstrip(".,!?;:") if i > 0 else ""
                right = (
                    words[i + 1].lower().rstrip(".,!?;:") if i < len(words) - 1 else ""
                )
                if (
                    left in _COMPOSITIONAL_NUMBER_WORDS
                    or right in _COMPOSITIONAL_NUMBER_WORDS
                ):
                    out.append(word)
                else:
                    punct = word[len(word.rstrip(".,!?;:")) :]
                    out.append(_DIGIT_WORDS[bare] + punct)
            else:
                out.append(word)
        return " ".join(out)


# ---------------------------------------------------------------------------
# Email normalization
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)


def _email_to_words(match: re.Match) -> str:  # type: ignore[type-arg]
    email = match.group()
    local, domain = email.split("@", 1)

    def _expand(part: str) -> str:
        part = re.sub(r"\.", " dot ", part)
        part = re.sub(r"-", " dash ", part)
        part = re.sub(r"\+", " plus ", part)
        part = re.sub(r"_", " underscore ", part)
        return part

    return _expand(local) + " at " + _expand(domain)


class NormalizeEmails(AbstractTransform):
    """Convert email addresses to spoken-word form.

    Example: user.name+tag@sub.example.com
      → "user dot name plus tag at sub dot example dot com"
    """

    def process_string(self, s: str) -> str:
        return _EMAIL_RE.sub(_email_to_words, s)


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://(?:www\.)?([A-Za-z0-9.\-]+)(?:/[^\s]*)?",
)


def _url_to_words(match: re.Match) -> str:  # type: ignore[type-arg]
    domain = match.group(1)
    domain = re.sub(r"\.", " dot ", domain)
    domain = re.sub(r"-", " dash ", domain)
    return domain.strip()


class NormalizeURLs(AbstractTransform):
    """Convert URLs to their spoken domain form, dropping scheme and path.

    Example: "https://www.example-site.com/some/path?q=1" → "example dash site dot com"
    """

    def process_string(self, s: str) -> str:
        return _URL_RE.sub(_url_to_words, s)


# ---------------------------------------------------------------------------
# Filler words & stutter/repetition
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"\b(um+|uh+|hmm+|hm+|mm+|er+|ah+|eh+|mhm+|erm+)\b",
    re.IGNORECASE,
)

_MULTI_SPACE_RE = re.compile(r" {2,}")


class RemoveFillerWords(AbstractTransform):
    """Remove spoken filler words (um, uh, hmm, er, ah, …).

    Common in spontaneous speech and often transcribed by ASR models.
    """

    def process_string(self, s: str) -> str:
        s = _FILLER_RE.sub("", s)
        return _MULTI_SPACE_RE.sub(" ", s).strip()


class CollapseRepetitions(AbstractTransform):
    """Collapse consecutive identical words to a single occurrence.

    Models ASR output stuttering: "I I I think" → "I think".
    Case-insensitive comparison; preserves the first occurrence's casing.
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

_ABBREVIATIONS: dict[str, str] = {
    "dr.": "doctor",
    "mr.": "mister",
    "mrs.": "missus",
    "ms.": "miss",
    "prof.": "professor",
    "sr.": "senior",
    "jr.": "junior",
    "vs.": "versus",
    "etc.": "et cetera",
    "approx.": "approximately",
    "dept.": "department",
    "est.": "established",
    "inc.": "incorporated",
    "ltd.": "limited",
    "corp.": "corporation",
    "ave.": "avenue",
    "blvd.": "boulevard",
    "st.": "street",
    "jan.": "january",
    "feb.": "february",
    "mar.": "march",
    "apr.": "april",
    "aug.": "august",
    "sep.": "september",
    "oct.": "october",
    "nov.": "november",
    "dec.": "december",
}

_ABBREV_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in _ABBREVIATIONS) + r")",
    re.IGNORECASE,
)


class ExpandAbbreviations(AbstractTransform):
    """Expand common spoken abbreviations to their full form.

    Examples: "Dr. Smith" → "doctor Smith", "vs." → "versus"
    """

    def process_string(self, s: str) -> str:
        return _ABBREV_RE.sub(lambda m: _ABBREVIATIONS[m.group().lower()], s)


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------

_SYMBOL_MAP: list[tuple[str, str]] = [
    ("&amp;", "and"),
    ("&", "and"),
    (" + ", " plus "),
    (" @ ", " at "),
    (" # ", " number "),
    (" * ", " times "),
    (" = ", " equals "),
    (" > ", " greater than "),
    (" < ", " less than "),
    ("...", " "),
]


class NormalizeSymbols(AbstractTransform):
    """Replace common written symbols with their spoken equivalents.

    Only replaces symbols that appear as standalone tokens (surrounded by spaces
    or text boundaries) to avoid corrupting emails and URLs.
    """

    def process_string(self, s: str) -> str:
        for symbol, word in _SYMBOL_MAP:
            s = s.replace(symbol, word)
        return s


# ---------------------------------------------------------------------------
# Currency normalization
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS: dict[str, tuple[str, str]] = {
    "$": ("dollar", "cent"),
    "€": ("euro", "cent"),
    "£": ("pound", "penny"),
    "¥": ("yen", ""),
    "₹": ("rupee", "paisa"),
}

_CURRENCY_RE = re.compile(
    r"([$€£¥₹])\s*(\d+)(?:\.(\d{1,2}))?",
)

_IRREGULAR_PLURALS: dict[str, str] = {"penny": "pennies", "paisa": "paise"}


def _pluralize(unit: str, count: int) -> str:
    if count == 1:
        return unit
    return _IRREGULAR_PLURALS.get(unit, unit + "s")


def _currency_to_words(match: re.Match) -> str:  # type: ignore[type-arg]
    from num2words import num2words

    symbol = match.group(1)
    major = int(match.group(2))
    minor_str = match.group(3)
    unit_major, unit_minor = _CURRENCY_SYMBOLS.get(symbol, ("dollar", "cent"))

    def n2w(n: int) -> str:
        return num2words(n).replace("-", " ")

    parts = [f"{n2w(major)} {_pluralize(unit_major, major)}"]
    if minor_str:
        minor = int(minor_str.ljust(2, "0"))
        if minor > 0 and unit_minor:
            parts.append(f"{n2w(minor)} {_pluralize(unit_minor, minor)}")
    return " ".join(parts)


class NormalizeCurrency(AbstractTransform):
    """Convert currency amounts to spoken form.

    Examples:
      "$5"    → "five dollars"
      "$5.99" → "five dollars ninety nine cents"
      "€3.50" → "three euros fifty cents"
    """

    def process_string(self, s: str) -> str:
        return _CURRENCY_RE.sub(_currency_to_words, s)


# ---------------------------------------------------------------------------
# Percentage normalization
# ---------------------------------------------------------------------------

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


class NormalizePercentages(AbstractTransform):
    """Convert percentage notation to spoken form.

    Examples: "50%" → "fifty percent", "3.5%" → "three point five percent"
    """

    def process_string(self, s: str) -> str:
        from num2words import num2words

        def _replace(m: re.Match) -> str:  # type: ignore[type-arg]
            val = float(m.group(1))
            if val == int(val):
                return f"{num2words(int(val)).replace('-', ' ')} percent"
            return f"{num2words(val).replace('-', ' ')} percent"

        return _PERCENT_RE.sub(_replace, s)


# ---------------------------------------------------------------------------
# Ordinal normalization
# ---------------------------------------------------------------------------

_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)


class NormalizeOrdinals(AbstractTransform):
    """Convert ordinal numerals to spoken word form.

    Examples: "1st" → "first", "2nd" → "second", "15th" → "fifteenth"
    """

    def process_string(self, s: str) -> str:
        from num2words import num2words

        def _replace(m: re.Match) -> str:  # type: ignore[type-arg]
            return num2words(int(m.group(1)), to="ordinal").replace("-", " ")

        return _ORDINAL_RE.sub(_replace, s)
