---
title: Emotion Intensity
description: Detect intensity levels of emotions
---

# Emotion Intensity

Detect not just what emotion is present, but how intense it is.

## Usage

```python
result = await sm.detect_emotions(
    "I'm absolutely FURIOUS about this!!!",
    include_intensity=True
)

print(f"Emotion: {result.primary}")    # "anger"
print(f"Intensity: {result.intensity}")  # "high"
```

## Intensity Levels

| Level | Description | Example |
|-------|-------------|---------|
| `low` | Mild emotion | "I'm a bit annoyed" |
| `medium` | Moderate emotion | "I'm upset about this" |
| `high` | Intense emotion | "I'm absolutely FURIOUS!!!" |

## Intensity Indicators

The model considers:

- **Intensifiers**: very, extremely, absolutely, really
- **Punctuation**: !!!, ???, CAPS
- **Repetition**: sooooo, nooooo
- **Emojis**: Usage and count
- **Word choice**: furious vs annoyed

## Use Cases

- **Crisis detection**: High-intensity negative emotions
- **Customer satisfaction**: High-intensity positive
- **Sentiment triage**: Priority by intensity
- **Mental health**: Concerning patterns

## Example: Intensity-Based Triage

```python
async def triage_reviews(reviews):
    urgent = []
    normal = []

    for review in reviews:
        result = await sm.detect_emotions(
            review.text,
            include_intensity=True
        )

        if result.primary in ["anger", "fear"] and result.intensity == "high":
            urgent.append(review)
        else:
            normal.append(review)

    return urgent, normal
```

## Related

- [Emotion Overview](index.md)
- [Detection Modes](modes.md)
- [Taxonomies](taxonomies.md)
