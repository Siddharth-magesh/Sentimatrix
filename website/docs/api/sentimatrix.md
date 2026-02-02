---
title: Sentimatrix Class
description: Main entry point API reference
---

# Sentimatrix Class

The main entry point for all Sentimatrix operations.

## Class Definition

```python
class Sentimatrix:
    """Main entry point for Sentimatrix operations."""

    def __init__(
        self,
        config: SentimatrixConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Initialize Sentimatrix.

        Args:
            config: Configuration object. If None, loads from file or defaults.
            config_path: Path to YAML config file.
        """
```

## Context Manager

```python
async with Sentimatrix(config) as sm:
    result = await sm.analyze("Hello world!")
```

## Manual Lifecycle

```python
sm = Sentimatrix(config)
await sm.initialize()

try:
    result = await sm.analyze("Hello")
finally:
    await sm.close()
```

## Methods

### Sentiment Analysis

| Method | Description |
|--------|-------------|
| `analyze()` | Analyze single text |
| `analyze_batch()` | Analyze multiple texts |
| `analyze_aspects()` | Aspect-based analysis |
| `compare_sentiment()` | Compare across groups |
| `analyze_temporal()` | Time-series analysis |

### Emotion Detection

| Method | Description |
|--------|-------------|
| `detect_emotions()` | Detect emotions |
| `detect_emotions_batch()` | Batch emotion detection |
| `analyze_emotion_timeline()` | Track emotions over time |

### Web Scraping

| Method | Description |
|--------|-------------|
| `scrape_reviews()` | Scrape reviews from platform |

### LLM Features

| Method | Description |
|--------|-------------|
| `summarize_reviews()` | Generate summary |
| `generate_insights()` | Extract insights |
| `compare_products()` | Compare products |
| `llm_analyze()` | Custom LLM analysis |
| `stream_summary()` | Streaming summary |

### Multimodal

| Method | Description |
|--------|-------------|
| `analyze_audio()` | Analyze audio |
| `analyze_image()` | Analyze image |
| `analyze_video()` | Analyze video |
| `analyze_multimodal()` | Combined analysis |

## Related

- [API Overview](index.md)
- [Configuration](config.md)
