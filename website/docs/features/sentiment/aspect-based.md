---
title: Aspect-Based Sentiment
description: Sentiment analysis per aspect/feature
---

# Aspect-Based Sentiment

Analyze sentiment for specific product aspects or features.

## Usage

```python
result = await sm.analyze_aspects(
    "The camera is excellent but battery life is disappointing",
    aspects=["camera", "battery", "design", "price"]
)

for aspect, sentiment in result.items():
    print(f"{aspect}: {sentiment}")
```

Output:

```
camera: positive
battery: negative
design: neutral
price: neutral
```

## With Confidence Scores

```python
result = await sm.analyze_aspects(
    "Great camera, terrible battery",
    aspects=["camera", "battery"],
    include_scores=True
)

# {
#   'camera': {'sentiment': 'positive', 'confidence': 0.92},
#   'battery': {'sentiment': 'negative', 'confidence': 0.88}
# }
```

## Common Aspects by Domain

### Electronics
- camera, battery, screen, performance, price, design

### Restaurants
- food, service, ambiance, price, cleanliness

### Hotels
- room, location, staff, amenities, value

### Software
- usability, features, performance, support, price

## Related

- [Sentiment Overview](index.md)
- [Comparative](comparative.md)
