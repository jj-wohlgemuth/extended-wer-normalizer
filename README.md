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

## Pipeline design

The English pipeline applies transforms left-to-right in a single pass:

1. **Pattern-specific** (before punctuation is stripped): email, URL, symbol, abbreviation, currency, percentage, ordinal
2. **Core**: contractions (`I'm` → `i am`), lowercase, punctuation removal
3. **Digit normalization**: expand digit runs (`0176` → `0 1 7 6`), convert digit words (`zero` → `0`)
4. **Cleanup**: filler words, repetition collapse

## Non-English

For non-English text, pass any `language` value other than `"en"` to apply a minimal language-agnostic pipeline (lowercase, punctuation removal, whitespace normalization):

```python
normalize_for_wer("Das kostet fünf Euro", language="de")
# → "das kostet fünf euro"
```
