# Sentimatrix V2 - Input/Output Formats

## Overview

This document defines all supported input and output formats for LLM interactions in Sentimatrix V2.

---

## Input Types

### 1. Text Input

**Direct String:**
```python
result = await sm.analyze("This product is amazing!")
```

**List of Strings:**
```python
results = await sm.analyze_batch([
    "Great product!",
    "Terrible experience",
    "It's okay"
])
```

**With Metadata:**
```python
result = await sm.analyze(
    TextInput(
        text="Great product!",
        metadata={
            "source": "amazon",
            "product_id": "B0123",
            "timestamp": "2024-01-15"
        }
    )
)
```

### 2. Structured Input

**Dictionary:**
```python
result = await sm.analyze({
    "text": "Review text here",
    "rating": 4,
    "author": "John",
    "date": "2024-01-15"
})
```

**Pydantic Model:**
```python
class ReviewInput(BaseModel):
    text: str
    rating: Optional[int]
    author: Optional[str]
    product_name: Optional[str]

review = ReviewInput(
    text="Great product!",
    rating=5,
    author="Jane"
)
result = await sm.analyze(review)
```

**JSON String:**
```python
result = await sm.analyze(
    '{"text": "Great!", "rating": 5}',
    input_format="json"
)
```

### 3. URL Input

**Single URL:**
```python
result = await sm.analyze_url("https://amazon.com/product/B0123")
```

**Multiple URLs:**
```python
results = await sm.analyze_urls([
    "https://amazon.com/product/A",
    "https://amazon.com/product/B"
])
```

**With Options:**
```python
result = await sm.analyze_url(
    url="https://amazon.com/product/B0123",
    options=UrlOptions(
        limit=100,
        sort="recent",
        scraper="playwright"
    )
)
```

### 4. File Input

**Text File:**
```python
result = await sm.analyze_file("reviews.txt")
```

**CSV File:**
```python
result = await sm.analyze_file(
    "reviews.csv",
    options=CsvOptions(
        text_column="review_text",
        rating_column="stars",
        delimiter=","
    )
)
```

**JSON File:**
```python
result = await sm.analyze_file(
    "reviews.json",
    options=JsonOptions(
        text_path="$.reviews[*].text"
    )
)
```

### 5. Audio Input

**Single File:**
```python
result = await sm.analyze_audio("recording.wav")
```

**With Transcription Options:**
```python
result = await sm.analyze_audio(
    "recording.mp3",
    options=AudioOptions(
        model="whisper-large",
        language="en",
        include_timestamps=True
    )
)
```

### 6. Image Input

**Single Image:**
```python
result = await sm.analyze_image("screenshot.png")
```

**With Analysis Options:**
```python
result = await sm.analyze_image(
    "product_photo.jpg",
    options=ImageOptions(
        model="gpt-4-vision",
        prompt="Analyze the sentiment expressed in this image",
        include_ocr=True
    )
)
```

**Base64 Encoded:**
```python
result = await sm.analyze_image(
    image_data=base64_string,
    input_format="base64"
)
```

### 7. Video Input

**Video File:**
```python
result = await sm.analyze_video(
    "review_video.mp4",
    options=VideoOptions(
        frame_extraction="keyframe",
        analyze_audio=True,
        max_frames=50
    )
)
```

### 8. Mixed/Multi-Modal Input

```python
result = await sm.analyze_multimodal(
    MultiModalInput(
        text="Check out my review",
        images=["photo1.jpg", "photo2.jpg"],
        audio="voice_note.wav"
    )
)
```

---

## Output Formats

### 1. Simple Output

**Sentiment Result:**
```python
@dataclass
class SentimentResult:
    label: str          # "positive", "negative", "neutral"
    score: float        # 0.0 - 1.0
    confidence: float   # Optional confidence measure

# Usage
result = await sm.analyze_sentiment(text)
print(result.label)     # "positive"
print(result.score)     # 0.92
```

### 2. Structured Output

**Full Analysis Result:**
```python
@dataclass
class AnalysisResult:
    input: str
    sentiment: SentimentResult
    emotions: Optional[List[EmotionScore]]
    aspects: Optional[List[AspectSentiment]]
    summary: Optional[str]
    reasoning: Optional[str]
    metadata: Dict[str, Any]
    processing_time_ms: float
```

**JSON Output:**
```python
result = await sm.analyze(
    text,
    output_format="json"
)
# Returns:
{
    "input": "Great product!",
    "sentiment": {
        "label": "positive",
        "score": 0.95
    },
    "emotions": [
        {"label": "joy", "score": 0.85},
        {"label": "satisfaction", "score": 0.72}
    ],
    "aspects": [
        {"aspect": "quality", "sentiment": "positive", "score": 0.9}
    ]
}
```

### 3. Streaming Output

**Text Streaming:**
```python
async for chunk in sm.analyze_stream(text):
    print(chunk, end="", flush=True)
```

**Structured Streaming:**
```python
async for event in sm.analyze_stream(text, structured=True):
    if event.type == "thinking":
        print(f"Thinking: {event.content}")
    elif event.type == "result":
        print(f"Result: {event.sentiment}")
```

### 4. Batch Output

**List of Results:**
```python
results = await sm.analyze_batch(texts)
# Returns: List[SentimentResult]

for text, result in zip(texts, results):
    print(f"{text}: {result.label}")
```

**With Errors:**
```python
@dataclass
class BatchResult:
    results: List[Optional[SentimentResult]]
    errors: List[Optional[Error]]
    success_count: int
    error_count: int
```

### 5. Report Output

**Summary Report:**
```python
@dataclass
class SummaryReport:
    total_reviews: int
    sentiment_distribution: Dict[str, int]
    average_score: float
    top_aspects: List[AspectSummary]
    summary_text: str
    recommendations: List[str]
```

**Comparison Report:**
```python
@dataclass
class ComparisonReport:
    products: List[ProductSummary]
    winner: str
    comparison_matrix: Dict[str, Dict[str, float]]
    narrative: str
```

### 6. Export Formats

**CSV Export:**
```python
await sm.export(results, "output.csv", format="csv")

# Output:
# text,sentiment,score,confidence
# "Great product!",positive,0.95,0.92
# "Terrible",negative,0.88,0.85
```

**JSON Export:**
```python
await sm.export(results, "output.json", format="json")

# Output: Array of full result objects
```

**Excel Export:**
```python
await sm.export(results, "output.xlsx", format="xlsx")

# Output: Excel file with formatted sheets
```

### 7. Visualization Output

**Chart Data:**
```python
chart = await sm.visualize(
    results,
    chart_type="bar",
    return_data=True
)
# Returns: ChartData with labels, values, colors
```

**Image Output:**
```python
await sm.visualize(
    results,
    chart_type="pie",
    save_path="sentiment_distribution.png"
)
```

---

## Output Customization

### Include/Exclude Fields

```python
result = await sm.analyze(
    text,
    include=[
        "sentiment",
        "emotions",
        "reasoning"
    ],
    exclude=[
        "metadata",
        "processing_time"
    ]
)
```

### Custom Output Schema

```python
from pydantic import BaseModel

class CustomOutput(BaseModel):
    sentiment: str
    confidence: float
    key_phrases: List[str]
    recommendation: str

result = await sm.analyze(
    text,
    output_schema=CustomOutput
)
```

### Output Transformation

```python
result = await sm.analyze(
    text,
    transform=lambda r: {
        "label": r.sentiment.label,
        "score": round(r.sentiment.score, 2)
    }
)
```

---

## Error Handling in Outputs

### Error Types

```python
@dataclass
class AnalysisError:
    error_type: str
    message: str
    input_index: Optional[int]
    recoverable: bool
    suggestion: Optional[str]
```

### Error Responses

```python
try:
    result = await sm.analyze(text)
except ValidationError as e:
    # Input validation failed
    print(f"Invalid input: {e.message}")
except ProviderError as e:
    # LLM provider error
    print(f"Provider error: {e.provider} - {e.message}")
except TimeoutError as e:
    # Operation timed out
    print(f"Timeout after {e.seconds}s")
```

### Partial Results

```python
result = await sm.analyze_batch(
    texts,
    on_error="continue"  # or "stop"
)

for i, (res, err) in enumerate(zip(result.results, result.errors)):
    if err:
        print(f"Error at {i}: {err.message}")
    else:
        print(f"Result at {i}: {res.label}")
```

---

## Configuration

```yaml
input:
  # Default input handling
  max_text_length: 10000
  truncation: "end"  # end, start, middle
  encoding: "utf-8"

  # File handling
  max_file_size_mb: 100
  allowed_extensions:
    text: [".txt", ".md"]
    csv: [".csv", ".tsv"]
    json: [".json", ".jsonl"]
    audio: [".wav", ".mp3", ".flac"]
    image: [".png", ".jpg", ".webp"]
    video: [".mp4", ".avi", ".mov"]

output:
  # Default format
  format: "structured"

  # Include by default
  default_fields:
    - sentiment
    - confidence
    - processing_time

  # JSON options
  json:
    indent: 2
    ensure_ascii: false

  # Streaming
  streaming:
    chunk_size: 100
    buffer_size: 1000
```

---

## Type Definitions

```python
# Input types
TextInput = Union[str, List[str], Dict[str, Any]]
UrlInput = Union[str, List[str], HttpUrl]
FileInput = Union[str, Path, BinaryIO]
AudioInput = Union[str, Path, bytes]
ImageInput = Union[str, Path, bytes]
VideoInput = Union[str, Path]

# Output types
SentimentLabel = Literal["positive", "negative", "neutral"]
EmotionLabel = Literal["joy", "sadness", "anger", "fear", "surprise", "disgust", ...]
OutputFormat = Literal["json", "csv", "xlsx", "html"]
ChartType = Literal["bar", "pie", "line", "histogram", "heatmap"]
```
