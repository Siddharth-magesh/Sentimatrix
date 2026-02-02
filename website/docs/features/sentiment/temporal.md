---
title: Temporal Sentiment
description: Track sentiment changes over time
---

# Temporal Sentiment

Track how sentiment changes across time periods.

## Usage

```python
reviews_over_time = [
    {"text": "Great product!", "date": "2024-01-01"},
    {"text": "Quality seems worse", "date": "2024-03-01"},
    {"text": "Terrible now", "date": "2024-06-01"},
]

timeline = await sm.analyze_temporal(reviews_over_time)

for period in timeline:
    print(f"{period.date}: {period.sentiment} ({period.avg_score:.2f})")
```

Output:

```
2024-01: positive (0.85)
2024-03: negative (-0.42)
2024-06: negative (-0.89)
```

## Use Cases

- Product quality tracking
- Brand perception monitoring
- Campaign impact analysis
- Seasonal trends

## Aggregation Periods

```python
# Monthly aggregation
timeline = await sm.analyze_temporal(reviews, period="month")

# Weekly aggregation
timeline = await sm.analyze_temporal(reviews, period="week")

# Daily aggregation
timeline = await sm.analyze_temporal(reviews, period="day")
```

## Detecting Trends

```python
# Identify sentiment shifts
for i, period in enumerate(timeline[1:], 1):
    prev = timeline[i-1]
    change = period.avg_score - prev.avg_score

    if change < -0.3:
        print(f"Significant drop at {period.date}")
    elif change > 0.3:
        print(f"Significant improvement at {period.date}")
```

## Related

- [Sentiment Overview](index.md)
- [Comparative](comparative.md)
