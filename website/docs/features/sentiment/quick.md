---
title: Quick Sentiment
description: Fast sentiment classification
---

# Quick Sentiment

The fastest mode for high-volume sentiment classification into positive/negative/neutral.

## Usage

```python
async with Sentimatrix() as sm:
    result = await sm.analyze("This product is amazing!")

    print(result.sentiment)   # "positive"
    print(result.confidence)  # 0.967
```

## When to Use

- High-volume filtering
- Real-time analysis
- Simple classification needs
- Resource-constrained environments

## Batch Processing

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

## Performance

| Metric | Value |
|--------|-------|
| Latency (CPU) | ~25ms |
| Latency (GPU) | ~5ms |
| Throughput | 400+ texts/sec |

## Related

- [Structured Analysis](structured.md)
- [Fine-Grained](../sentiment/index.md#fine-grained-sentiment)
