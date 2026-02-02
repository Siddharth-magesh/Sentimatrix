---
title: Features
description: Comprehensive sentiment analysis, emotion detection, and multi-modal capabilities
---

# Features

Sentimatrix provides a comprehensive suite of text analysis features.

## Feature Categories

<div class="grid">

<div class="card">
<h3>:material-text-box-search: Sentiment Analysis</h3>
<p>Multiple analysis modes from quick classification to aspect-based analysis.</p>
<p><a href="sentiment/">Learn more →</a></p>
</div>

<div class="card">
<h3>:material-emoticon: Emotion Detection</h3>
<p>Detect emotions using multiple taxonomies with intensity analysis.</p>
<p><a href="emotion/">Learn more →</a></p>
</div>

<div class="card">
<h3>:material-image-multiple: Multi-Modal</h3>
<p>Analyze text, audio, images, and video for comprehensive understanding.</p>
<p><a href="multimodal/">Learn more →</a></p>
</div>

<div class="card">
<h3>:material-brain: LLM Features</h3>
<p>Summarization, insight generation, and reasoning with 19 LLM providers.</p>
<p><a href="llm/">Learn more →</a></p>
</div>

</div>

## Feature Overview

### Sentiment Analysis

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Quick Sentiment** | Fast positive/negative/neutral | High-volume processing |
| **Structured Analysis** | Detailed scores and confidence | Analytics dashboards |
| **Fine-Grained** | 5-class classification | Nuanced understanding |
| **Aspect-Based** | Per-aspect sentiment | Product feedback |
| **Comparative** | Compare across items | Competitor analysis |
| **Temporal** | Track changes over time | Trend monitoring |
| **Domain-Specific** | Specialized for industry | Finance, healthcare |

### Emotion Detection

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Ekman Basic** | 6 universal emotions | General use |
| **GoEmotions** | 28 fine-grained emotions | Detailed analysis |
| **Plutchik Wheel** | 8 primary + combinations | Psychological research |
| **Intensity** | Low/medium/high levels | Customer satisfaction |
| **Timeline** | Emotion changes over text | Narrative analysis |

### Multi-Modal Analysis

| Modality | Capabilities | Formats |
|----------|--------------|---------|
| **Text** | Full sentiment/emotion | Any text |
| **Audio** | Speech-to-text + analysis | WAV, MP3, FLAC |
| **Image** | Visual sentiment, OCR | PNG, JPEG, WEBP |
| **Video** | Frame extraction + audio | MP4, AVI, MOV |

### LLM-Powered Features

| Feature | Description | Requires LLM |
|---------|-------------|--------------|
| **Summarization** | Generate review summaries | Yes |
| **Insights** | Extract pros/cons/themes | Yes |
| **Reasoning** | Chain-of-thought analysis | Yes |
| **Comparison** | Compare products/services | Yes |

## Quick Examples

### Sentiment Analysis

```python
async with Sentimatrix() as sm:
    # Quick sentiment
    result = await sm.analyze("Great product!")
    print(f"{result.sentiment}: {result.confidence:.0%}")

    # Aspect-based
    aspects = await sm.analyze_aspects(
        "Great camera but battery life is poor",
        aspects=["camera", "battery", "design"]
    )
    # {'camera': 'positive', 'battery': 'negative', 'design': 'neutral'}
```

### Emotion Detection

```python
async with Sentimatrix() as sm:
    emotions = await sm.detect_emotions("I'm thrilled about this!")

    print(f"Primary: {emotions.primary}")  # joy
    print(f"Top 3: {emotions.top_k(3)}")   # [('joy', 0.89), ('surprise', 0.12), ...]
```

### LLM Features

```python
async with Sentimatrix(config) as sm:
    # Summarize reviews
    summary = await sm.summarize_reviews(reviews)

    # Generate insights
    insights = await sm.generate_insights(reviews)
    print(f"Pros: {insights.pros}")
    print(f"Cons: {insights.cons}")
```

## Performance Benchmarks

### Sentiment Analysis

| Model | Accuracy | Latency (CPU) | Latency (GPU) |
|-------|----------|---------------|---------------|
| RoBERTa-base | 94.8% | 50ms | 9ms |
| DistilBERT | 92.1% | 25ms | 5ms |
| BERT-base | 93.5% | 60ms | 12ms |

### Emotion Detection

| Model | Accuracy | Classes | Latency |
|-------|----------|---------|---------|
| GoEmotions | 65.2% | 28 | 45ms |
| Ekman | 78.3% | 6 | 35ms |

### Throughput

| Mode | Batch Size | Throughput |
|------|------------|------------|
| Single | 1 | 20/sec |
| Batch | 32 | 400/sec |
| Batch (GPU) | 32 | 2000/sec |

## Feature Documentation

### Sentiment Analysis
- [Overview](sentiment/index.md)
- [Quick Sentiment](sentiment/quick.md)
- [Structured Analysis](sentiment/structured.md)
- [Aspect-Based](sentiment/aspect-based.md)
- [Comparative](sentiment/comparative.md)
- [Temporal](sentiment/temporal.md)

### Emotion Detection
- [Overview](emotion/index.md)
- [Taxonomies](emotion/taxonomies.md)
- [Detection Modes](emotion/modes.md)
- [Intensity Analysis](emotion/intensity.md)

### Multi-Modal
- [Overview](multimodal/index.md)
- [Audio Analysis](multimodal/audio.md)
- [Image Analysis](multimodal/image.md)
- [Video Analysis](multimodal/video.md)

### LLM Features
- [Overview](llm/index.md)
- [Summarization](llm/summarization.md)
- [Insights](llm/insights.md)
- [Reasoning](llm/reasoning.md)
