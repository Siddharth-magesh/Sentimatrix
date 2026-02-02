---
title: Structured Analysis
description: Detailed sentiment scores and confidence
---

# Structured Analysis

Get detailed per-class scores along with the sentiment label.

## Usage

```python
result = await sm.analyze(
    "Good product but shipping was slow",
    structured=True
)

print(f"Label: {result.sentiment}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Scores:")
print(f"  Positive: {result.scores['positive']:.3f}")
print(f"  Negative: {result.scores['negative']:.3f}")
print(f"  Neutral: {result.scores['neutral']:.3f}")
```

Output:

```
Label: positive
Confidence: 54.23%
Scores:
  Positive: 0.542
  Negative: 0.312
  Neutral: 0.146
```

## When to Use

- Analytics dashboards
- Confidence thresholds
- Mixed sentiment detection
- Score-based filtering

## Score Interpretation

| Scenario | Positive | Negative | Neutral |
|----------|----------|----------|---------|
| Clear positive | 0.95 | 0.03 | 0.02 |
| Clear negative | 0.02 | 0.96 | 0.02 |
| Mixed sentiment | 0.45 | 0.40 | 0.15 |
| Neutral | 0.15 | 0.10 | 0.75 |

## Related

- [Quick Sentiment](quick.md)
- [Aspect-Based](aspect-based.md)
