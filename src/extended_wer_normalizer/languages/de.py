"""German language data."""

from __future__ import annotations

import re

from . import LanguageData, register

DE = LanguageData(
    code="de",
    digit_words={
        "null": "0",
        "eins": "1",
        "zwei": "2",
        "drei": "3",
        "vier": "4",
        "fünf": "5",
        "sechs": "6",
        "sieben": "7",
        "acht": "8",
        "neun": "9",
    },
    compositional_number_words=frozenset(
        {
            "elf",
            "zwölf",
            "dreizehn",
            "vierzehn",
            "fünfzehn",
            "sechzehn",
            "siebzehn",
            "achtzehn",
            "neunzehn",
            "zwanzig",
            "dreißig",
            "vierzig",
            "fünfzig",
            "sechzig",
            "siebzig",
            "achtzig",
            "neunzig",
            "hundert",
            "tausend",
            "million",
            "millionen",
            "milliarde",
            "milliarden",
            "billion",
            "billionen",
        }
    ),
    filler_re=re.compile(
        r"\b(ähm+|äh+|öh+|hm+|hmm+|mhm+|mh+|na+|tja+|also)\b",
        re.IGNORECASE,
    ),
    abbreviations={
        # Titles
        "dr.": "doktor",
        "prof.": "professor",
        "hr.": "herr",
        "fr.": "frau",
        "frl.": "fräulein",
        "dipl.": "diplom",
        # Generic
        "etc.": "et cetera",
        "ca.": "circa",
        "ggf.": "gegebenenfalls",
        "bzw.": "beziehungsweise",
        "evtl.": "eventuell",
        "usw.": "und so weiter",
        "z.b.": "zum beispiel",
        "d.h.": "das heißt",
        "u.a.": "unter anderem",
        "i.d.r.": "in der regel",
        # Address
        "str.": "straße",
        "nr.": "nummer",
        "pl.": "platz",
        # Months
        "jan.": "januar",
        "feb.": "februar",
        "mär.": "märz",
        "apr.": "april",
        "aug.": "august",
        "sep.": "september",
        "okt.": "oktober",
        "nov.": "november",
        "dez.": "dezember",
        # Numeric magnitudes
        "mio.": "millionen",
        "mrd.": "milliarden",
    },
    symbol_map=[
        ("&amp;", "und"),
        ("&", "und"),
        (" + ", " plus "),
        (" @ ", " at "),
        (" # ", " nummer "),
        (" * ", " mal "),
        (" = ", " gleich "),
        (" > ", " größer als "),
        (" < ", " kleiner als "),
        ("...", " "),
    ],
    email_lexicon={
        "dot": "punkt",
        "dash": "bindestrich",
        "at": "at",
        "plus": "plus",
        "underscore": "unterstrich",
    },
    url_lexicon={"dot": "punkt", "dash": "bindestrich"},
    currency_symbols={
        "$": ("dollar", "cent"),
        "€": ("euro", "cent"),
        "£": ("pfund", "penny"),
        "¥": ("yen", ""),
        "₹": ("rupie", "paisa"),
    },
    # Most German currency units don't take plural forms.
    currency_plurals={
        "dollar": "dollar",
        "cent": "cent",
        "euro": "euro",
        "pfund": "pfund",
        "penny": "pennys",
        "yen": "yen",
        "rupie": "rupien",
        "paisa": "paisa",
    },
    word_for_percent="prozent",
    # German ordinal: 1-3 digits followed by "." and a word ("1. Januar" → "erste januar").
    # The 1-3 digit cap keeps years like "1990. " out. The lookbehind blocks decimals,
    # the lookahead blocks sentence-ending periods (no following word).
    ordinal_re=re.compile(r"(?<!\w)(\d{1,3})\.(?=\s+\w)"),
)

register(DE)
