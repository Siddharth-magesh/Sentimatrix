---
title: Multi-Modal Analysis
description: Analyze text, audio, images, and video for comprehensive understanding
---

# Multi-Modal Analysis

Sentimatrix supports analyzing multiple modalities - text, audio, images, and video - with fusion strategies for comprehensive sentiment understanding.

## Supported Modalities

<div class="grid">

<div class="card">
<h3>:material-text: Text</h3>
<p>Standard text sentiment and emotion analysis with 48 models.</p>
</div>

<div class="card">
<h3>:material-microphone: Audio</h3>
<p>Speech-to-text transcription followed by sentiment analysis.</p>
</div>

<div class="card">
<h3>:material-image: Image</h3>
<p>Image captioning and visual sentiment analysis.</p>
</div>

<div class="card">
<h3>:material-video: Video</h3>
<p>Frame extraction + audio analysis with result fusion.</p>
</div>

</div>

## Audio Analysis

Transcribe audio and analyze the sentiment of spoken content:

```python
async with Sentimatrix() as sm:
    result = await sm.analyze_audio("customer_call.wav")

    print(f"Transcription: {result.transcription}")
    print(f"Sentiment: {result.sentiment.label}")
    print(f"Emotions: {result.emotions.top_k(3)}")
```

### Supported Audio Formats

- WAV, MP3, FLAC, OGG, M4A

### Transcription Engines (4 Implemented)

| Engine | Provider | Speed | Quality |
|--------|----------|-------|---------|
| `whisper-base` | OpenAI (local) | Fast | Good |
| `whisper-medium` | OpenAI (local) | Medium | Better |
| `whisper-large-v3` | OpenAI (local) | Slow | Best |
| `groq-whisper` | Groq API | Fast | Excellent |

### Configuration

```python
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",  # Use Groq for fast transcription
    )
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze_audio(
        "call.wav",
        engine="groq-whisper"
    )
```

## Image Analysis

Analyze images for visual sentiment and emotional content:

```python
async with Sentimatrix() as sm:
    result = await sm.analyze_image("product_photo.jpg")

    print(f"Caption: {result.caption}")
    print(f"Sentiment: {result.sentiment.label}")
    print(f"Visual mood: {result.mood}")
```

### Supported Image Formats

- PNG, JPEG, WEBP, GIF, BMP

### Vision Models (5 Implemented)

| Model | Provider | Features |
|-------|----------|----------|
| `llava` | Ollama (local) | General vision |
| `blip-base` | Salesforce | Fast captioning |
| `blip-large` | Salesforce | Better captions |
| `blip2-opt-2.7b` | Salesforce | Best quality |
| `gpt-4-vision` | OpenAI API | Premium quality |
| `claude-vision` | Anthropic API | Safety-focused |
| `gemini-vision` | Google API | Multimodal |

### Configuration

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

## Video Analysis

Analyze videos by extracting frames and audio:

```python
async with Sentimatrix() as sm:
    result = await sm.analyze_video("review_video.mp4")

    print(f"Duration: {result.duration}s")
    print(f"Frames analyzed: {result.frame_count}")
    print(f"Overall sentiment: {result.sentiment.label}")
    print(f"Audio sentiment: {result.audio_sentiment.label}")
    print(f"Visual sentiment: {result.visual_sentiment.label}")
```

### Supported Video Formats

- MP4, AVI, MOV, WEBM, MKV

### Frame Extraction Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `uniform` | Extract every N frames | General analysis |
| `keyframe` | Extract key frames only | Efficient processing |
| `scene` | Detect scene changes | Narrative analysis |
| `custom` | User-defined intervals | Specific timestamps |

### Configuration

```python
async with Sentimatrix() as sm:
    result = await sm.analyze_video(
        "video.mp4",
        frame_method="uniform",
        frame_interval=5,  # Every 5 seconds
        analyze_audio=True,
        max_frames=20,
    )
```

## Fusion Strategies

Combine results from multiple modalities:

### Late Fusion (Default)

Analyze each modality separately, then combine:

```python
result = await sm.analyze_video(
    "video.mp4",
    fusion_strategy="late"
)

# Result combines audio + visual sentiments
print(f"Combined: {result.sentiment}")
print(f"Audio: {result.audio_sentiment}")
print(f"Visual: {result.visual_sentiment}")
```

### Weighted Fusion

Apply custom weights to each modality:

```python
result = await sm.analyze_video(
    "video.mp4",
    fusion_strategy="weighted",
    weights={"audio": 0.6, "visual": 0.4}
)
```

### Dominant Fusion

Use the modality with highest confidence:

```python
result = await sm.analyze_video(
    "video.mp4",
    fusion_strategy="dominant"
)

print(f"Dominant modality: {result.dominant_modality}")
```

## Combined Analysis

Analyze multiple inputs together:

```python
async with Sentimatrix() as sm:
    result = await sm.analyze_multimodal(
        text="Customer review: Great product!",
        audio="voicemail.wav",
        image="product_photo.jpg",
    )

    print(f"Text sentiment: {result.text_sentiment}")
    print(f"Audio sentiment: {result.audio_sentiment}")
    print(f"Image sentiment: {result.image_sentiment}")
    print(f"Overall: {result.combined_sentiment}")
```

## Hardware Requirements

| Modality | CPU | GPU (Recommended) |
|----------|-----|-------------------|
| Text | 4GB RAM | Optional |
| Audio (Whisper base) | 4GB RAM | 4GB VRAM |
| Audio (Whisper large) | 8GB RAM | 8GB VRAM |
| Image (BLIP) | 4GB RAM | 4GB VRAM |
| Image (LLaVA) | 16GB RAM | 12GB VRAM |
| Video | 8GB+ RAM | 8GB+ VRAM |

## Performance Tips

1. **Use GPU Acceleration**
    ```python
    config = SentimatrixConfig(
        model=ModelConfig(device="cuda")
    )
    ```

2. **Batch Process Frames**
    - Process multiple frames in batches
    - Reduces overhead

3. **Choose Appropriate Models**
    - Use smaller models for speed
    - Larger models for quality

4. **Limit Frame Count**
    - More frames = slower processing
    - 10-20 frames often sufficient

## Use Cases

| Use Case | Modalities | Recommended Setup |
|----------|------------|-------------------|
| Call center analysis | Audio | Whisper + sentiment |
| Product reviews | Text + Image | BLIP + RoBERTa |
| Video testimonials | Audio + Video | Whisper + LLaVA |
| Social media | Text + Image | Fast models |
| Customer support | Audio | Groq Whisper |

## Related

- [Sentiment Analysis](../sentiment/index.md)
- [Emotion Detection](../emotion/index.md)
- [LLM Features](../llm/index.md)
