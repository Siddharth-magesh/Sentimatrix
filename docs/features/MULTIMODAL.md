# Sentimatrix V2 - Multi-Modal Analysis Features

**Status: IMPLEMENTED** ✅

## Overview

V2 extends sentiment analysis beyond text to support audio, image, and video inputs. Each modality has specialized processing pipelines that ultimately produce sentiment/emotion outputs.

### Implementation Files
- `sentimatrix/input/handlers.py` - Audio, Image, Video handlers (~800 lines)
- `sentimatrix/analysis/multimodal.py` - Multi-modal analyzer (~850 lines)
- `tests/unit/input/test_handlers.py` - Handler tests (33 tests)
- `tests/unit/analysis/test_multimodal.py` - Multi-modal tests (27 tests)

---

## 1. Audio Analysis ✅

### Pipeline

```
Audio File → Speech-to-Text → Text Analysis → Results
     │              │
     └──────────────┴── Audio Features (optional)
```

### Supported Formats
- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A

### Speech-to-Text Engines (Implemented)

| Engine | Type | Quality | Speed | Cost | Status |
|--------|------|---------|-------|------|--------|
| OpenAI Whisper (local) | Local | Excellent | Medium | Free | ✅ Implemented |
| OpenAI Whisper API | Cloud | Excellent | Fast | $0.006/min | ✅ Implemented |
| Groq Whisper | Cloud | Excellent | Very Fast | Free tier | ✅ Implemented |
| Google Speech-to-Text | Cloud | Excellent | Fast | $0.006/15s | Planned |
| AssemblyAI | Cloud | Excellent | Fast | $0.00025/s | Planned |
| Deepgram | Cloud | Good | Very Fast | $0.0043/min | Planned |
| Vosk | Local | Good | Fast | Free | Planned |

### Audio-Specific Features
- Speaker diarization (who said what)
- Emotion from voice (tone analysis)
- Transcription with timestamps
- Language detection

### Configuration

```yaml
audio:
  engine: "whisper"  # whisper, google, assemblyai, deepgram, vosk
  model: "base"      # tiny, base, small, medium, large
  language: "auto"   # auto-detect or specify
  include_timestamps: true
  speaker_diarization: false
  analyze_tone: false
```

---

### Usage Example

```python
from sentimatrix import Sentimatrix

async with Sentimatrix() as sm:
    # Analyze audio file
    result = await sm.analyze_audio("review.mp3", language="en")
    print(result.transcription.text)
    print(result.sentiment.sentiment)

    # Just transcribe (no sentiment)
    transcript = await sm.transcribe_audio("podcast.wav")
    print(transcript.text)
```

---

## 2. Image Analysis ✅

### Pipeline

```
Image File → Image Captioning → Caption Analysis → Results
     │              │
     └──────────────┴── Visual Emotion Detection (optional)
```

### Supported Formats
- PNG
- JPEG/JPG
- WEBP
- GIF (first frame)
- BMP

### Image Understanding Models (Implemented)

| Model | Type | Capabilities | Status |
|-------|------|--------------|--------|
| GPT-4 Vision | Cloud | Advanced understanding | ✅ Implemented |
| Claude Vision | Cloud | Advanced understanding | ✅ Implemented |
| Gemini Vision | Cloud | Advanced understanding | ✅ Implemented |
| BLIP | Local | Captioning | ✅ Implemented |
| LLaVA | Local | General captioning | Planned |
| BLIP-2 | Local | Advanced captioning | Planned |

### Image-Specific Features
- Scene description
- Object detection
- Text extraction (OCR)
- Facial emotion detection
- Brand/logo detection
- Screenshot analysis

### Configuration

```yaml
image:
  captioning_model: "llava"  # llava, blip, gpt4v, claude, gemini
  include_ocr: true
  detect_faces: false
  detect_objects: true
  max_resolution: 1024
```

### Usage Example

```python
from sentimatrix import Sentimatrix

async with Sentimatrix() as sm:
    # Analyze image
    result = await sm.analyze_image("product.jpg", detailed=True)
    print(result.caption.caption)
    print(result.sentiment.sentiment)

    # Just caption (no sentiment)
    caption = await sm.caption_image("screenshot.png")
    print(caption.caption)
```

---

## 3. Video Analysis ✅

### Pipeline

```
Video File → Frame Extraction → Per-Frame Analysis → Aggregation → Results
     │              │                    │
     └── Audio ─────┴── Audio Analysis ──┘
```

### Supported Formats
- MP4
- AVI
- MOV
- WEBM
- MKV

### Video Processing Approaches

| Approach | Description | Use Case |
|----------|-------------|----------|
| Key Frame | Extract representative frames | Long videos |
| Uniform Sampling | Extract every N frames | General analysis |
| Scene Detection | Extract on scene changes | Movies, ads |
| Full Audio | Complete audio analysis | Podcasts, interviews |

### Video-Specific Features
- Frame-by-frame sentiment
- Audio track analysis
- Subtitle/caption extraction
- Scene-level aggregation
- Temporal sentiment mapping

### Configuration

```yaml
video:
  frame_extraction: "keyframe"  # keyframe, uniform, scene
  sample_rate: 1.0              # frames per second (for uniform)
  max_frames: 100
  analyze_audio: true
  extract_subtitles: true
  aggregation: "weighted_average"
```

### Usage Example

```python
from sentimatrix import Sentimatrix

async with Sentimatrix() as sm:
    # Full video analysis
    result = await sm.analyze_video(
        "review_video.mp4",
        analyze_audio=True,
        analyze_frames=True,
        max_frames=50
    )
    print(f"Frames analyzed: {result.frames_analyzed}")
    print(f"Duration: {result.duration_seconds}s")
    if result.audio_result:
        print(f"Audio transcript: {result.audio_result.transcription.text}")
    print(f"Combined sentiment: {result.combined_sentiment.sentiment}")
```

---

## 4. Combined Multi-Modal Analysis ✅

When multiple modalities are present (e.g., video with audio):

### Fusion Strategies (Implemented)

| Strategy | Description | Status |
|----------|-------------|--------|
| Late Fusion | Analyze separately, merge by majority vote | ✅ Implemented |
| Weighted | Weight by configurable modality weights | ✅ Implemented |
| Dominant | Use highest-confidence modality | ✅ Implemented |
| Early Fusion | Combine features before analysis | Planned |

### Configuration

```yaml
multimodal:
  fusion_strategy: "late"
  weights:
    text: 0.5
    audio: 0.3
    image: 0.2
  confidence_threshold: 0.6
```

### Usage Example

```python
from sentimatrix import Sentimatrix
from sentimatrix.analysis.multimodal import MultiModalAnalyzer, FusionStrategy

# Using main Sentimatrix class
async with Sentimatrix() as sm:
    result = await sm.analyze_multimodal(
        text="This product is amazing!",
        audio="review.mp3",
        image="product.jpg"
    )
    print(f"Modalities: {result.modalities_analyzed}")
    print(f"Combined: {result.combined_sentiment.sentiment}")

# Using MultiModalAnalyzer directly
analyzer = MultiModalAnalyzer(
    fusion_strategy=FusionStrategy.WEIGHTED,
    weights={"text": 0.6, "audio": 0.2, "image": 0.2}
)
async with analyzer:
    result = await analyzer.analyze_multimodal(
        text="Great quality!",
        audio="feedback.wav"
    )
```

---

## 5. Multi-Modal Output Format

```python
{
    "input_type": "video",
    "modalities_analyzed": ["video_frames", "audio"],
    "results": {
        "video_frames": {
            "frames_analyzed": 50,
            "dominant_sentiment": "positive",
            "sentiment_timeline": [...],
            "key_frames": [...]
        },
        "audio": {
            "transcription": "...",
            "sentiment": {"label": "positive", "score": 0.82},
            "duration_seconds": 120
        }
    },
    "combined": {
        "sentiment": {"label": "positive", "score": 0.78},
        "confidence": 0.85,
        "fusion_method": "weighted_average"
    }
}
```

---

## Hardware Requirements

| Modality | Minimum RAM | Recommended RAM | GPU Benefit |
|----------|-------------|-----------------|-------------|
| Text | 4GB | 8GB | Moderate |
| Audio (Whisper base) | 4GB | 8GB | High |
| Audio (Whisper large) | 10GB | 16GB | Very High |
| Image (LLaVA) | 8GB | 16GB | High |
| Video | 8GB | 32GB | Very High |

---

## Cloud vs Local Trade-offs

| Aspect | Local | Cloud |
|--------|-------|-------|
| Cost | Hardware only | Per-request |
| Latency | Low | Network dependent |
| Privacy | Full control | Data leaves system |
| Quality | Model dependent | Generally higher |
| Scalability | Hardware limited | Elastic |

---

## Use Cases

| Use Case | Recommended Setup |
|----------|-------------------|
| Product review videos | Video + Audio, late fusion |
| Customer call analysis | Audio only, Whisper |
| Social media images | Image + OCR, LLaVA/GPT-4V |
| Surveillance sentiment | Video keyframes |
| Podcast analysis | Audio only, large Whisper |
