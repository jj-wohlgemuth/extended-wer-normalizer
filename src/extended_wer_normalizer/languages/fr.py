"""French language data."""

from __future__ import annotations

import re

from . import LanguageData, register

FR = LanguageData(
    code="fr",
    digit_words={
        "zéro": "0",
        "un": "1",
        "deux": "2",
        "trois": "3",
        "quatre": "4",
        "cinq": "5",
        "six": "6",
        "sept": "7",
        "huit": "8",
        "neuf": "9",
    },
    compositional_number_words=frozenset(
        {
            # teens
            "onze",
            "douze",
            "treize",
            "quatorze",
            "quinze",
            "seize",
            "dix",  # also forms 17 (dix-sept), 18 (dix-huit), 19 (dix-neuf), 70 (soixante-dix)
            # tens
            "vingt",
            "vingts",
            "trente",
            "quarante",
            "cinquante",
            "soixante",
            # large
            "cent",
            "cents",
            "mille",
            "million",
            "millions",
            "milliard",
            "milliards",
            # connector for "vingt et un", "trente et un", …
            "et",
        }
    ),
    filler_re=re.compile(
        r"\b(euh+|heu+|hum+|hm+|mh+|bah+|ben+|ah+|oh+|alors)\b",
        re.IGNORECASE,
    ),
    abbreviations={
        # Titles
        "m.": "monsieur",
        "mm.": "messieurs",
        "mme": "madame",
        "mmes": "mesdames",
        "mlle": "mademoiselle",
        "mlles": "mesdemoiselles",
        "dr.": "docteur",
        "pr.": "professeur",
        # Generic
        "etc.": "et cetera",
        "env.": "environ",
        "ex.": "exemple",
        "p.ex.": "par exemple",
        "c.-à-d.": "c'est-à-dire",
        "qqch.": "quelque chose",
        "qqn.": "quelqu'un",
        # Address
        "av.": "avenue",
        "bd.": "boulevard",
        "bld.": "boulevard",
        "rte.": "route",
        "pl.": "place",
        # Months
        "janv.": "janvier",
        "févr.": "février",
        "avr.": "avril",
        "juil.": "juillet",
        "sept.": "septembre",
        "oct.": "octobre",
        "nov.": "novembre",
        "déc.": "décembre",
    },
    symbol_map=[
        ("&amp;", "et"),
        ("&", "et"),
        (" + ", " plus "),
        (" @ ", " arobase "),
        (" # ", " dièse "),
        (" * ", " fois "),
        (" = ", " égale "),
        (" > ", " supérieur à "),
        (" < ", " inférieur à "),
        ("...", " "),
    ],
    email_lexicon={
        "dot": "point",
        "dash": "tiret",
        "at": "arobase",
        "plus": "plus",
        "underscore": "souligné",
    },
    url_lexicon={"dot": "point", "dash": "tiret"},
    currency_symbols={
        "$": ("dollar", "cent"),
        "€": ("euro", "centime"),
        "£": ("livre", "penny"),
        "¥": ("yen", ""),
        "₹": ("roupie", "paisa"),
    },
    currency_plurals={
        "dollar": "dollars",
        "cent": "cents",
        "euro": "euros",
        "centime": "centimes",
        "livre": "livres",
        "penny": "pennies",
        "yen": "yens",
        "roupie": "roupies",
        "paisa": "paisas",
    },
    word_for_percent="pour cent",
    # French ordinal suffixes: 1er, 1ère, 2e, 2ème, 2es, 2èmes, 2nde, 2nds, 2nd.
    # Order longer alternatives first so e.g. "èmes" wins over "ème" + leftover "s",
    # though backtracking makes most orderings work in practice.
    ordinal_re=re.compile(
        r"\b(\d+)(?:èmes|ème|ères|ère|ers|er|nde|nds|nd|es|e)\b",
        re.IGNORECASE,
    ),
)

register(FR)
