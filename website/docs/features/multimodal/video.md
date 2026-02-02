---
title: Video Analysis
description: Combined audio and visual sentiment analysis
---

# Video Analysis

Analyze videos by extracting frames and audio for comprehensive sentiment analysis.

## Usage

```python
async with Sentimatrix() as sm:
    result = await sm.analyze_video("review_video.mp4")

    print(f"Duration: {result.duration}s")
    print(f"Frames analyzed: {result.frame_count}")
    print(f"Overall sentiment: {result.sentiment.label}")
    print(f"Audio sentiment: {result.audio_sentiment.label}")
    print(f"Visual sentiment: {result.visual_sentiment.label}")
```

## Supported Formats

- MP4
- AVI
- MOV
- WEBM
- MKV

## Frame Extraction Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `uniform` | Every N frames | General |
| `keyframe` | Key frames only | Efficient |
| `scene` | Scene changes | Narrative |
| `custom` | User-defined | Specific timestamps |

## Configuration

```python
result = await sm.analyze_video(
    "video.mp4",
    frame_method="uniform",
    frame_interval=5,      # Every 5 seconds
    analyze_audio=True,
    max_frames=20,
)
```

## Fusion Strategies

```python
# Late fusion (default)
result = await sm.analyze_video(video, fusion_strategy="late")

# Weighted fusion
result = await sm.analyze_video(
    video,
    fusion_strategy="weighted",
    weights={"audio": 0.6, "visual": 0.4}
)

# Dominant fusion
result = await sm.analyze_video(video, fusion_strategy="dominant")
```

## Use Cases

- Video testimonial analysis
- YouTube content analysis
- Product demo sentiment
- Social media videos

## Related

- [Multimodal Overview](index.md)
- [Audio Analysis](audio.md)
- [Image Analysis](image.md)
