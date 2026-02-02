---
title: Detection Modes
description: Single-label, multi-label, and top-k emotion detection
---

# Detection Modes

Different modes for emotion detection depending on your needs.

## Single-Label Mode (Default)

Returns the dominant emotion:

```python
result = await sm.detect_emotions(
    "I'm happy but also a bit nervous",
    mode="single_label"
)

print(result.primary)  # "joy" (highest score)
```

## Multi-Label Mode

Returns all emotions above a threshold:

```python
result = await sm.detect_emotions(
    "I'm happy but also a bit nervous",
    mode="multi_label",
    threshold=0.3
)

print(result.labels)  # ["joy", "nervousness"]
```

## Top-K Mode

Returns the top K emotions:

```python
result = await sm.detect_emotions(
    "This is absolutely amazing!",
    mode="top_k",
    k=3
)

print(result.top_k(3))
# [("excitement", 0.85), ("joy", 0.72), ("admiration", 0.45)]
```

## Choosing a Mode

| Scenario | Mode | Why |
|----------|------|-----|
| Simple classification | single_label | One clear answer |
| Complex emotions | multi_label | Capture nuances |
| Ranked list | top_k | See alternatives |
| Analytics | top_k | Full distribution |

## Threshold Tuning

For multi-label mode, adjust threshold based on needs:

```python
# Strict (fewer labels)
result = await sm.detect_emotions(text, mode="multi_label", threshold=0.5)

# Lenient (more labels)
result = await sm.detect_emotions(text, mode="multi_label", threshold=0.2)
```

## Related

- [Emotion Overview](index.md)
- [Taxonomies](taxonomies.md)
- [Intensity](intensity.md)
