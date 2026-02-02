---
title: Image Analysis
description: Visual sentiment and emotion analysis
---

# Image Analysis

Analyze images for visual sentiment and emotional content.

## Usage

```python
async with Sentimatrix() as sm:
    result = await sm.analyze_image("product_photo.jpg")

    print(f"Caption: {result.caption}")
    print(f"Sentiment: {result.sentiment.label}")
    print(f"Visual mood: {result.mood}")
```

## Supported Formats

- PNG
- JPEG
- WEBP
- GIF
- BMP

## Vision Models

| Model | Provider | Features |
|-------|----------|----------|
| `llava` | Ollama | General vision |
| `blip-base` | Salesforce | Fast captioning |
| `blip-large` | Salesforce | Better captions |
| `blip2-opt-2.7b` | Salesforce | Best quality |
| `gpt-4-vision` | OpenAI | Premium |
| `claude-vision` | Anthropic | Safety-focused |
| `gemini-vision` | Google | Multimodal |

## Configuration

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4-vision-preview"
    )
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze_image(
        "product.jpg",
        prompt="What emotions does this product image convey?"
    )
```

## Use Cases

- Product photo analysis
- Social media sentiment
- Ad effectiveness
- Brand perception

## Related

- [Multimodal Overview](index.md)
- [Audio Analysis](audio.md)
- [Video Analysis](video.md)
