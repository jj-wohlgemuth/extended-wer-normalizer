# extended-wer-normalizer

jiwer-compatible text normalizer for Word Error Rate (WER) evaluation in voice AI.

Extends [jiwer](https://github.com/jitsi/jiwer)'s built-in transforms with normalizations that matter for real-world ASR evaluation: phone numbers, emails, URLs, currency, percentages, ordinals, filler words, and stuttering.

## Installation

```bash
pip install extended-wer-normalizer
```

## Quick start

```python
from extended_wer_normalizer import normalize_for_wer

normalize_for_wer("Call 0176 or email info@example.com, it costs $5.99")
# → "call 0 1 7 6 or email info at example dot com it costs five dollars ninety nine cents"

normalize_for_wer("Um, 1st place goes to Dr. Smith with 50% accuracy")
# → "first place goes to doctor smith with fifty percent accuracy"
```

## jiwer integration

Every normalization is a `jiwer.AbstractTransform` subclass — compose them freely:

```python
import jiwer
from extended_wer_normalizer.transforms import NormalizeEmails, ExpandDigitRuns

pipeline = jiwer.Compose([
    NormalizeEmails(),
    ExpandDigitRuns(),
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])

wer = jiwer.wer("info at example dot com", "info@example.com", hypothesis_transform=pipeline)
```

Use the pre-built pipeline directly with `jiwer.wer`:

```python
import jiwer
from extended_wer_normalizer import english_wer_pipeline

wer = jiwer.wer(reference, hypothesis, reference_transform=english_wer_pipeline, hypothesis_transform=english_wer_pipeline)
```

## Available transforms

| Transform | Example |
|---|---|
| `ExpandDigitRuns` | `"0176"` → `"0 1 7 6"` |
| `DigitWordsToChars` | `"zero one seven"` → `"0 1 7"` |
| `NormalizeEmails` | `"user@example.com"` → `"user at example dot com"` |
| `NormalizeURLs` | `"https://example.com/path"` → `"example dot com"` |
| `NormalizeCurrency` | `"$5.99"` → `"five dollars ninety nine cents"` |
| `NormalizePercentages` | `"50%"` → `"fifty percent"` |
| `NormalizeOrdinals` | `"1st"` → `"first"`, `"15th"` → `"fifteenth"` |
| `ExpandAbbreviations` | `"Dr."` → `"doctor"`, `"vs."` → `"versus"` |
| `NormalizeSymbols` | `"cats & dogs"` → `"cats and dogs"` |
| `RemoveFillerWords` | removes `um`, `uh`, `hmm`, `er`, `ah`, … |
| `CollapseRepetitions` | `"I I I think"` → `"I think"` |
| `ExpandFrenchElisions` | `"j'aime"` → `"j aime"`, `"qu'il"` → `"qu il"` (French only) |

Every transform that consumes language-specific data accepts a `language="en"` keyword (default English): `NormalizeEmails(language="fr")`, `ExpandAbbreviations(language="de")`, etc.

## Pipeline design

The English pipeline applies transforms left-to-right in a single pass:

1. **Pattern-specific** (before punctuation is stripped): email, URL, symbol, abbreviation, currency, percentage, ordinal
2. **Core**: contractions (`I'm` → `i am`), lowercase, punctuation removal
3. **Digit normalization**: expand digit runs (`0176` → `0 1 7 6`), convert digit words (`zero` → `0`)
4. **Cleanup**: filler words, repetition collapse

## Supported languages

Full pipelines (with language-specific abbreviations, fillers, lexicons, and number/ordinal/percentage word forms via `num2words`) ship for **English**, **German**, and **French**. Pass any other `language` value for the minimal fallback (lowercase + punctuation + whitespace).

```python
from extended_wer_normalizer import normalize_for_wer

# German: titles, fillers, ordinals, currency
normalize_for_wer("Hr. Müller, am 1. Januar, ähm, ungefähr 50% Rabatt", language="de")
# → "herr müller am erste januar ungefähr fünfzig prozent rabatt"

# French: elision contractions, ordinals, comma-decimal currency
normalize_for_wer("M. Dupont, le 1er janvier, c'est €5,99", language="fr")
# → "monsieur dupont le premier janvier c est cinq euros quatre vingt dix neuf centimes"

# Spanish, Italian, … fall through to the minimal pipeline
normalize_for_wer("¡Hola, mundo!", language="es")
# → "hola mundo"
```

Per-language pipelines are also exposed for direct use with `jiwer.wer`:

```python
from extended_wer_normalizer import (
    english_wer_pipeline,
    german_wer_pipeline,
    french_wer_pipeline,
)
```

To inspect or extend the language data:

```python
from extended_wer_normalizer.languages import get_language_data, supported_languages

supported_languages()              # ["de", "en", "fr"]
get_language_data("de").abbreviations["hr."]  # "herr"
```

### Quirks worth knowing

- **Comma vs. period decimals**: French uses `,` (`€5,99`, `3,5%`); the currency and percentage transforms accept either separator regardless of language.
- **German ordinals**: matched as 1- to 3-digit numbers followed by `. ` and a word (e.g. `"1. Januar"` but not `"Es war 1990."` or `"1.5 Liter"`). 4+ digits and decimals are skipped to avoid false positives on years.
- **French ordinals**: matched as `1er`, `1ère`, `2e`, `2es`, `2ème`, `2èmes`, `2nde`, `2nds`, `2nd`. `num2words` returns masculine forms (`premier`, `deuxième`); feminine variants like `première` or `seconde` are not produced.
- **Contractions**: `jiwer.ExpandCommonEnglishContractions` runs only for English. French has a custom `ExpandFrenchElisions` that splits `j'`, `l'`, `d'`, `n'`, `s'`, `m'`, `t'`, `c'`, `qu'`, `jusqu'`, `lorsqu'`, `puisqu'`, `quoiqu'` from the following word. German has no contraction step.
- **German pluralization**: most currency units stay invariant (`fünf Euro`, not `fünf Euros`).
