---
title: Emotion Detection
description: Detect emotions using multiple taxonomies and modes
---

# Emotion Detection

Sentimatrix provides comprehensive emotion detection with support for multiple emotion taxonomies and detection modes.

## Emotion Taxonomies

### GoEmotions (28 Classes)

The default taxonomy with fine-grained emotion detection:

```python
async with Sentimatrix() as sm:
    result = await sm.detect_emotions("I'm so excited about this!")

    print(f"Primary: {result.primary}")      # "excitement"
    print(f"Confidence: {result.confidence}")  # 0.89
    print(f"All scores: {result.scores}")    # {'excitement': 0.89, 'joy': 0.45, ...}
```

**GoEmotions Categories:**
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

### Ekman's 6 Basic Emotions

Universal emotions recognized across cultures:

```python
result = await sm.detect_emotions(
    "I can't believe they did that!",
    taxonomy="ekman"
)

print(result.primary)  # "anger" or "surprise"
```

**Ekman Emotions:**
- Anger
- Disgust
- Fear
- Joy
- Sadness
- Surprise

### Plutchik's Wheel (8 Primary)

Emotions with intensity variations:

```python
result = await sm.detect_emotions(
    "This is terrifying!",
    taxonomy="plutchik"
)

print(result.primary)  # "fear"
print(result.intensity)  # "high"
```

**Plutchik Emotions:**
- Joy ↔ Sadness
- Trust ↔ Disgust
- Fear ↔ Anger
- Surprise ↔ Anticipation

## Detection Modes

### Single-Label Mode (Default)

Returns the dominant emotion:

```python
result = await sm.detect_emotions(
    "I'm happy but also a bit nervous",
    mode="single_label"
)

print(result.primary)  # "joy" (highest score)
```

### Multi-Label Mode

Returns all emotions above threshold:

```python
result = await sm.detect_emotions(
    "I'm happy but also a bit nervous",
    mode="multi_label",
    threshold=0.3
)

print(result.labels)  # ["joy", "nervousness"]
```

### Top-K Mode

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

## Batch Processing

Efficiently process multiple texts:

```python
texts = [
    "I'm so happy today!",
    "This makes me really angry",
    "I'm worried about the future",
]

results = await sm.detect_emotions_batch(texts)

for text, result in zip(texts, results):
    print(f"{result.primary}: {text}")
```

## Emotion Intensity

Detect the intensity level of emotions:

```python
result = await sm.detect_emotions(
    "I'm absolutely FURIOUS about this!!!",
    include_intensity=True
)

print(f"Emotion: {result.primary}")    # "anger"
print(f"Intensity: {result.intensity}")  # "high"
```

Intensity levels:
- `low` - Mild emotion
- `medium` - Moderate emotion
- `high` - Intense emotion

## Emotion Timeline

Track emotional changes across a sequence:

```python
messages = [
    "Starting the day feeling great!",
    "Work is getting stressful...",
    "Finally done, so relieved!",
]

timeline = await sm.analyze_emotion_timeline(messages)

for i, point in enumerate(timeline):
    print(f"Message {i+1}: {point.primary} ({point.confidence:.0%})")
```

Output:

```
Message 1: joy (87%)
Message 2: nervousness (72%)
Message 3: relief (81%)
```

## Available Models (5 Implemented)

| Model | Emotions | Best For |
|-------|----------|----------|
| `SamLowe/roberta-base-go_emotions` | 28 | Default, fine-grained |
| `bhadresh-savani/distilbert-base-uncased-emotion` | 6 | Fast, Ekman emotions |
| `j-hartmann/emotion-english-distilroberta-base` | 7 | General use |
| `cardiffnlp/twitter-roberta-base-emotion` | 4 | Twitter/social media |
| `mrm8488/t5-base-finetuned-emotion` | 6 | T5-based detection |

## Model Configuration

```python
from sentimatrix.config import SentimatrixConfig, ModelConfig

config = SentimatrixConfig(
    model=ModelConfig(
        emotion_model="SamLowe/roberta-base-go_emotions",
        device="cuda",  # GPU acceleration
        batch_size=32,
    )
)

async with Sentimatrix(config) as sm:
    result = await sm.detect_emotions("Hello world!")
```

## Combining with Sentiment

Analyze both sentiment and emotions together:

```python
async with Sentimatrix() as sm:
    # Combined analysis
    result = await sm.analyze_full("I'm thrilled but also a bit anxious about this!")

    print(f"Sentiment: {result.sentiment.label}")  # "positive"
    print(f"Emotions: {result.emotions.top_k(3)}")
    # [("excitement", 0.78), ("nervousness", 0.45), ("joy", 0.42)]
```

## Use Cases

| Use Case | Recommended Config |
|----------|-------------------|
| Customer feedback | GoEmotions, multi_label |
| Social media monitoring | Ekman, single_label |
| Chat bot emotions | GoEmotions, top_k(3) |
| Survey analysis | Plutchik, include_intensity |
| Crisis detection | Ekman, threshold=0.5 |

## Performance

| Model | Accuracy | Latency (CPU) | Latency (GPU) |
|-------|----------|---------------|---------------|
| GoEmotions (RoBERTa) | 65.2% | 45ms | 8ms |
| Ekman (DistilBERT) | 78.3% | 25ms | 5ms |
| Twitter Emotion | 72.1% | 30ms | 6ms |

## Related

- [Sentiment Analysis](../sentiment/index.md)
- [Multi-Modal Analysis](../multimodal/index.md)
- [Configuration](../../configuration/index.md)
