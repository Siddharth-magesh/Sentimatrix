---
title: Audio Analysis
description: Transcription and sentiment analysis of audio
---

# Audio Analysis

Transcribe audio and analyze sentiment from spoken content.

## Usage

```python
async with Sentimatrix() as sm:
    result = await sm.analyze_audio("customer_call.wav")

    print(f"Transcription: {result.transcription}")
    print(f"Sentiment: {result.sentiment.label}")
    print(f"Emotions: {result.emotions.top_k(3)}")
```

## Supported Formats

- WAV
- MP3
- FLAC
- OGG
- M4A

## Transcription Engines

| Engine | Speed | Quality | Provider |
|--------|-------|---------|----------|
| `whisper-base` | Fast | Good | Local |
| `whisper-medium` | Medium | Better | Local |
| `whisper-large-v3` | Slow | Best | Local |
| `groq-whisper` | Fast | Excellent | Groq API |

## Configuration

```python
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(provider="groq")  # For Groq Whisper
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze_audio(
        "call.wav",
        engine="groq-whisper"
    )
```

## Use Cases

- Call center analysis
- Voice feedback processing
- Podcast sentiment
- Meeting summaries

## Related

- [Multimodal Overview](index.md)
- [Image Analysis](image.md)
- [Video Analysis](video.md)
