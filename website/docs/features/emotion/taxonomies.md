---
title: Emotion Taxonomies
description: Different emotion classification systems
---

# Emotion Taxonomies

Sentimatrix supports three emotion taxonomies with different granularity levels.

## GoEmotions (28 Classes)

The default taxonomy with fine-grained emotion detection:

```python
result = await sm.detect_emotions(
    "I'm so excited about this!",
    taxonomy="goemotion"  # default
)

print(result.primary)  # "excitement"
```

**Categories:**
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

## Ekman's 6 Basic Emotions

Universal emotions recognized across cultures:

```python
result = await sm.detect_emotions(
    "I can't believe they did that!",
    taxonomy="ekman"
)

print(result.primary)  # "anger" or "surprise"
```

**Categories:**
- Anger
- Disgust
- Fear
- Joy
- Sadness
- Surprise

## Plutchik's Wheel (8 Primary)

Emotions with opposite pairs and intensity variations:

```python
result = await sm.detect_emotions(
    "This is terrifying!",
    taxonomy="plutchik"
)

print(result.primary)    # "fear"
print(result.intensity)  # "high"
```

**Emotion Pairs:**
- Joy ↔ Sadness
- Trust ↔ Disgust
- Fear ↔ Anger
- Surprise ↔ Anticipation

## Choosing a Taxonomy

| Use Case | Recommended |
|----------|-------------|
| Fine-grained analysis | GoEmotions |
| Simple categorization | Ekman |
| Intensity analysis | Plutchik |
| Cross-cultural | Ekman |
| Customer feedback | GoEmotions |

## Related

- [Emotion Overview](index.md)
- [Detection Modes](modes.md)
- [Intensity](intensity.md)
