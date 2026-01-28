# Sentimatrix V2 - Emotion Detection Features

## Overview

Emotion detection goes beyond positive/negative classification to identify specific emotional states in text. V2 supports multiple emotion taxonomies and detection approaches.

---

## Emotion Taxonomies

### 1. Ekman's Basic Emotions (6 classes)
- Joy
- Sadness
- Anger
- Fear
- Surprise
- Disgust

### 2. GoEmotions (28 classes)
Full taxonomy from Google's GoEmotions dataset:

| Positive | Negative | Ambiguous |
|----------|----------|-----------|
| Admiration | Anger | Confusion |
| Amusement | Annoyance | Curiosity |
| Approval | Disappointment | Realization |
| Caring | Disapproval | Surprise |
| Desire | Disgust | Neutral |
| Excitement | Embarrassment | |
| Gratitude | Fear | |
| Joy | Grief | |
| Love | Nervousness | |
| Optimism | Remorse | |
| Pride | Sadness | |
| Relief | | |

### 3. Plutchik's Wheel (8 primary + combinations)
- Joy - Sadness
- Trust - Disgust
- Fear - Anger
- Surprise - Anticipation

---

## Detection Modes

### 1. Single-Label Classification
Returns the dominant emotion only.

```python
{
    "emotion": "joy",
    "score": 0.89
}
```

### 2. Multi-Label Classification
Returns all detected emotions above threshold.

```python
{
    "emotions": [
        {"label": "joy", "score": 0.85},
        {"label": "gratitude", "score": 0.72},
        {"label": "excitement", "score": 0.45}
    ]
}
```

### 3. Top-K Emotions
Returns top K emotions by confidence.

```python
# top_k=3
{
    "emotions": [
        {"label": "anger", "score": 0.78},
        {"label": "disappointment", "score": 0.65},
        {"label": "annoyance", "score": 0.52}
    ],
    "dominant": "anger"
}
```

---

## Supported Models

| Model | Classes | Source | Notes |
|-------|---------|--------|-------|
| `SamLowe/roberta-base-go_emotions` | 28 | HuggingFace | Default |
| `bhadresh-savani/distilbert-base-uncased-emotion` | 6 | HuggingFace | Fast |
| `j-hartmann/emotion-english-distilroberta-base` | 7 | HuggingFace | Balanced |
| `cardiffnlp/twitter-roberta-base-emotion` | 4 | HuggingFace | Social media |
| Custom fine-tuned | Variable | Local | Domain-specific |

---

## Emotion Intensity

Beyond classification, V2 can estimate emotion intensity:

**Scale:** 0.0 (absent) to 1.0 (intense)

```python
{
    "emotion": "anger",
    "presence_score": 0.92,
    "intensity": "high",  # low, medium, high
    "intensity_score": 0.85
}
```

---

## Emotion Timeline

Track emotional changes across a sequence of texts:

```python
{
    "timeline": [
        {"index": 0, "text": "First review", "dominant_emotion": "joy"},
        {"index": 1, "text": "Second review", "dominant_emotion": "disappointment"},
        {"index": 2, "text": "Third review", "dominant_emotion": "anger"}
    ],
    "trend": "declining",
    "emotion_shift_points": [1]
}
```

---

## Emotion Aggregation

For collections of reviews/texts:

```python
{
    "total_reviews": 500,
    "emotion_distribution": {
        "joy": 0.35,
        "satisfaction": 0.25,
        "disappointment": 0.15,
        "anger": 0.10,
        "neutral": 0.15
    },
    "dominant_emotion": "joy",
    "emotional_polarity": 0.60  # -1 to 1
}
```

---

## LLM-Enhanced Emotion Analysis

### Emotion Explanation
Use LLM to explain why certain emotions are detected:

```python
{
    "emotion": "disappointment",
    "score": 0.87,
    "llm_explanation": "The reviewer expresses disappointment due to
                        product quality not meeting expectations set
                        by marketing claims."
}
```

### Emotion-Driven Recommendations
Generate recommendations based on emotional patterns:

```python
{
    "dominant_negative_emotions": ["frustration", "disappointment"],
    "recommendations": [
        "Improve product documentation to set clear expectations",
        "Enhance customer support response time",
        "Address quality control issues"
    ]
}
```

---

## Configuration

```yaml
emotion:
  model: "SamLowe/roberta-base-go_emotions"
  taxonomy: "goemotion"  # ekman, goemotion, plutchik
  mode: "multi_label"    # single, multi_label, top_k
  top_k: 5
  threshold: 0.3
  include_intensity: true
  device: "auto"
```

---

## Use Cases

| Use Case | Recommended Config |
|----------|-------------------|
| Customer feedback | GoEmotions, multi-label |
| Social media monitoring | Ekman, single-label |
| Product reviews | GoEmotions, top_k=3 |
| Support ticket triage | Ekman, with intensity |
| Brand sentiment | GoEmotions, aggregated |

---

## Performance

| Model | Accuracy | F1 Score | Speed (CPU) |
|-------|----------|----------|-------------|
| GoEmotions RoBERTa | 0.51 | 0.48 | 55ms |
| DistilBERT Emotion | 0.93 | 0.92 | 18ms |
| Twitter RoBERTa | 0.78 | 0.75 | 50ms |

Note: GoEmotions has lower accuracy due to 28-class complexity.
