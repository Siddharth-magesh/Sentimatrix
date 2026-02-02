---
title: Emotion Models
description: Emotion detection model reference
---

# Emotion Models

Available models for emotion detection.

## Implemented Models (5)

| Model | Emotions | Best For |
|-------|----------|----------|
| `SamLowe/roberta-base-go_emotions` | 28 | Default, fine-grained |
| `bhadresh-savani/distilbert-base-uncased-emotion` | 6 | Fast, Ekman |
| `j-hartmann/emotion-english-distilroberta-base` | 7 | General use |
| `cardiffnlp/twitter-roberta-base-emotion` | 4 | Twitter/social |
| `mrm8488/t5-base-finetuned-emotion` | 6 | T5-based |

## Configuration

```python
ModelConfig(
    emotion_model="SamLowe/roberta-base-go_emotions",
    device="auto",
    batch_size=32
)
```

## Taxonomies

### GoEmotions (28 classes)

```python
result = await sm.detect_emotions(text, taxonomy="goemotion")
```

### Ekman (6 classes)

```python
result = await sm.detect_emotions(text, taxonomy="ekman")
```

### Plutchik (8 classes)

```python
result = await sm.detect_emotions(text, taxonomy="plutchik")
```

## Performance

| Model | Accuracy | CPU Latency | GPU Latency |
|-------|----------|-------------|-------------|
| GoEmotions | 65.2% | 45ms | 8ms |
| Ekman DistilBERT | 78.3% | 25ms | 5ms |
| Twitter Emotion | 72.1% | 30ms | 6ms |

## Related

- [API Overview](index.md)
- [Sentiment Models](sentiment-models.md)
- [Emotion Detection](../features/emotion/index.md)
