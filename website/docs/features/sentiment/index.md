---
title: Sentiment Analysis
description: Comprehensive sentiment analysis with multiple analysis modes
---

# Sentiment Analysis

Sentimatrix provides multiple sentiment analysis modes for different use cases.

## Analysis Modes

| Mode | Output | Speed | Use Case |
|------|--------|-------|----------|
| **Quick** | Label only | Fastest | High-volume filtering |
| **Structured** | Label + scores | Fast | Analytics |
| **Fine-Grained** | 5 classes | Medium | Nuanced analysis |
| **Aspect-Based** | Per-aspect | Medium | Product feedback |
| **Comparative** | Comparison | Slower | Competition analysis |
| **Temporal** | Over time | Medium | Trend tracking |

## Quick Sentiment

Fast classification into positive/negative/neutral:

```python
async with Sentimatrix() as sm:
    result = await sm.analyze("This product is amazing!")

    print(result.sentiment)   # "positive"
    print(result.confidence)  # 0.967
```

### Batch Processing

```python
texts = [
    "Great quality!",
    "Terrible experience",
    "It's okay",
]

results = await sm.analyze_batch(texts)

for text, result in zip(texts, results):
    print(f"{result.sentiment:>10} ({result.confidence:.0%}): {text}")
```

Output:

```
  positive (97%): Great quality!
  negative (94%): Terrible experience
   neutral (82%): It's okay
```

## Structured Analysis

Get detailed scores for each class:

```python
result = await sm.analyze("Good product but shipping was slow", structured=True)

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

## Fine-Grained Sentiment

5-class classification for nuanced understanding:

```python
result = await sm.analyze(
    "I absolutely love this product!",
    mode="fine_grained"
)

print(result.sentiment)  # "very_positive"
```

Classes:
- `very_negative`
- `negative`
- `neutral`
- `positive`
- `very_positive`

## Aspect-Based Sentiment

Analyze sentiment for specific product aspects:

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

### With Confidence Scores

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

## Comparative Sentiment

Compare sentiment across products or time periods:

```python
products = {
    "Product A": [
        "Great quality product",
        "Love this item",
        "Some issues with durability"
    ],
    "Product B": [
        "Okay product",
        "Not worth the price",
        "Returned it"
    ]
}

comparison = await sm.compare_sentiment(products)

print(comparison)
# {
#   'Product A': {'positive': 0.67, 'negative': 0.33, 'avg_score': 0.45},
#   'Product B': {'positive': 0.00, 'negative': 0.67, 'avg_score': -0.52},
#   'winner': 'Product A'
# }
```

## Temporal Sentiment

Track sentiment changes over time:

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

## Domain-Specific Analysis

Use domain-specific models for better accuracy:

```python
# Financial sentiment
result = await sm.analyze(
    "Stock prices fell sharply today",
    domain="financial"
)
# Correctly identifies as negative (financial context)

# Social media
result = await sm.analyze(
    "This is so bad it's good! ",
    domain="social_media"
)
# Handles sarcasm and emojis better
```

Available domains:
- `general` (default)
- `financial`
- `social_media`
- `product_reviews`
- `healthcare`

## Model Configuration

### Default Model

```python
# Uses cardiffnlp/twitter-roberta-base-sentiment-latest
async with Sentimatrix() as sm:
    result = await sm.analyze("Hello world")
```

### Custom Model

```python
from sentimatrix.config import SentimatrixConfig, ModelConfig

config = SentimatrixConfig(
    model=ModelConfig(
        sentiment_model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze("Bonjour le monde")
```

### Available Models

| Model | Languages | Classes | Accuracy |
|-------|-----------|---------|----------|
| `twitter-roberta-base-sentiment` | English | 3 | 94.8% |
| `bert-base-multilingual-sentiment` | 100+ | 5 | 89.2% |
| `distilbert-base-sentiment` | English | 2 | 91.3% |
| `xlm-roberta-sentiment` | 100+ | 3 | 92.1% |

## Performance Tips

1. **Use Batch Processing**
    ```python
    # Slower
    for text in texts:
        result = await sm.analyze(text)

    # Faster (3-10x)
    results = await sm.analyze_batch(texts)
    ```

2. **Enable GPU Acceleration**
    ```python
    config = SentimatrixConfig(
        model=ModelConfig(device="cuda")  # or "mps" for Apple Silicon
    )
    ```

3. **Use Appropriate Mode**
    - Quick: High volume, don't need scores
    - Structured: Need confidence scores
    - Fine-grained: Need nuanced classes

4. **Cache Results**
    ```python
    config = SentimatrixConfig(
        cache=CacheConfig(enabled=True)
    )
    ```

## Related

- [Emotion Detection](../emotion/index.md)
- [LLM Features](../llm/index.md)
- [Configuration](../../configuration/index.md)
