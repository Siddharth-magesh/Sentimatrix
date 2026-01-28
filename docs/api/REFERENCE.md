# Sentimatrix V2 - API Reference

## Main Class

### Sentimatrix

```python
class Sentimatrix:
    """Main entry point for Sentimatrix functionality."""

    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[str] = None
    ) -> None:
        """
        Initialize Sentimatrix instance.

        Args:
            config: Configuration object
            config_path: Path to YAML config file
        """
```

---

## Sentiment Analysis

### analyze_sentiment

```python
async def analyze_sentiment(
    self,
    text: str,
    model: Optional[str] = None,
    return_all_scores: bool = False
) -> SentimentResult:
    """
    Analyze sentiment of a single text.

    Args:
        text: Text to analyze
        model: Model name (uses default if None)
        return_all_scores: Return scores for all classes

    Returns:
        SentimentResult with label and score

    Raises:
        ValueError: If text is empty
        ModelError: If model fails

    Example:
        >>> result = await sm.analyze_sentiment("Great product!")
        >>> print(result.label)  # "positive"
        >>> print(result.score)  # 0.95
    """
```

### analyze_sentiment_batch

```python
async def analyze_sentiment_batch(
    self,
    texts: List[str],
    model: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = False
) -> List[SentimentResult]:
    """
    Analyze sentiment of multiple texts.

    Args:
        texts: List of texts to analyze
        model: Model name
        batch_size: Processing batch size
        show_progress: Show progress bar

    Returns:
        List of SentimentResult objects
    """
```

---

## Emotion Detection

### detect_emotions

```python
async def detect_emotions(
    self,
    text: str,
    top_k: int = 5,
    threshold: float = 0.0,
    model: Optional[str] = None
) -> EmotionResult:
    """
    Detect emotions in text.

    Args:
        text: Text to analyze
        top_k: Number of top emotions to return
        threshold: Minimum confidence threshold
        model: Model name

    Returns:
        EmotionResult with emotions list and dominant emotion

    Example:
        >>> result = await sm.detect_emotions("I'm so happy!")
        >>> print(result.dominant_emotion)  # "joy"
        >>> print(result.emotions[0])  # EmotionScore(label="joy", score=0.92)
    """
```

### detect_emotions_batch

```python
async def detect_emotions_batch(
    self,
    texts: List[str],
    top_k: int = 5,
    batch_size: int = 32
) -> List[EmotionResult]:
    """Detect emotions in multiple texts."""
```

---

## Web Scraping

### scrape_url

```python
async def scrape_url(
    self,
    url: str,
    provider: Optional[str] = None,
    wait_for: Optional[str] = None,
    timeout: int = 30000
) -> ScrapedContent:
    """
    Scrape content from a URL.

    Args:
        url: URL to scrape
        provider: Scraper provider to use
        wait_for: CSS selector to wait for
        timeout: Timeout in milliseconds

    Returns:
        ScrapedContent with HTML and extracted text
    """
```

### scrape_reviews

```python
async def scrape_reviews(
    self,
    url: str,
    platform: Optional[str] = None,
    limit: int = 100,
    sort: str = "recent"
) -> List[Review]:
    """
    Scrape reviews from a URL.

    Args:
        url: URL to scrape
        platform: Platform name (auto-detected if None)
        limit: Maximum reviews to scrape
        sort: Sort order (recent, helpful, rating)

    Returns:
        List of Review objects

    Example:
        >>> reviews = await sm.scrape_reviews(
        ...     "https://amazon.com/product/...",
        ...     limit=50
        ... )
    """
```

---

## Platform-Specific Methods

### analyze_amazon

```python
async def analyze_amazon(
    self,
    url: str,
    limit: int = 100,
    include_summary: bool = True,
    include_aspects: bool = True
) -> ProductAnalysis:
    """
    Full analysis of Amazon product reviews.

    Args:
        url: Amazon product URL
        limit: Maximum reviews to analyze
        include_summary: Generate LLM summary
        include_aspects: Perform aspect-based analysis

    Returns:
        ProductAnalysis with reviews, sentiment, aspects, summary
    """
```

### analyze_youtube

```python
async def analyze_youtube(
    self,
    video_id: str,
    limit: int = 100,
    include_transcript: bool = False
) -> VideoAnalysis:
    """
    Analyze YouTube video comments.

    Args:
        video_id: YouTube video ID
        limit: Maximum comments to analyze
        include_transcript: Include video transcript analysis

    Returns:
        VideoAnalysis with comments and sentiment
    """
```

### analyze_reddit

```python
async def analyze_reddit(
    self,
    query: str,
    subreddit: Optional[str] = None,
    limit: int = 50,
    time_filter: str = "month"
) -> RedditAnalysis:
    """
    Analyze Reddit discussions.

    Args:
        query: Search query
        subreddit: Subreddit to search (all if None)
        limit: Maximum posts to analyze
        time_filter: Time filter (hour, day, week, month, year, all)

    Returns:
        RedditAnalysis with posts and sentiment
    """
```

### analyze_steam

```python
async def analyze_steam(
    self,
    game_name: str,
    limit: int = 100,
    filter: str = "all"
) -> GameAnalysis:
    """
    Analyze Steam game reviews.

    Args:
        game_name: Name of the game
        limit: Maximum reviews to analyze
        filter: Review filter (all, positive, negative)

    Returns:
        GameAnalysis with reviews and sentiment
    """
```

---

## Comparison

### compare_products

```python
async def compare_products(
    self,
    urls: List[str],
    aspects: Optional[List[str]] = None
) -> ComparisonResult:
    """
    Compare sentiment across multiple products.

    Args:
        urls: List of product URLs to compare
        aspects: Specific aspects to compare

    Returns:
        ComparisonResult with winner and aspect comparisons

    Example:
        >>> comparison = await sm.compare_products([
        ...     "https://amazon.com/product/A",
        ...     "https://amazon.com/product/B"
        ... ])
        >>> print(comparison.winner)
    """
```

---

## LLM Integration

### summarize_reviews

```python
async def summarize_reviews(
    self,
    reviews: List[Review],
    sentiment_results: Optional[List[SentimentResult]] = None,
    provider: Optional[str] = None
) -> str:
    """
    Generate LLM summary of reviews.

    Args:
        reviews: List of reviews to summarize
        sentiment_results: Pre-computed sentiment results
        provider: LLM provider to use

    Returns:
        Natural language summary
    """
```

### generate_insights

```python
async def generate_insights(
    self,
    reviews: List[Review],
    analysis_type: str = "general"
) -> Dict[str, str]:
    """
    Generate AI-powered insights from reviews.

    Args:
        reviews: List of reviews
        analysis_type: Type of analysis
            - "general": Overall insights
            - "recommendations": Product recommendations
            - "improvements": Suggested improvements
            - "aspects": Aspect-specific insights

    Returns:
        Dictionary of insights
    """
```

---

## Multi-Modal

### analyze_audio

```python
async def analyze_audio(
    self,
    file_path: str,
    transcription_model: str = "whisper"
) -> AudioAnalysis:
    """
    Analyze sentiment from audio file.

    Args:
        file_path: Path to audio file
        transcription_model: Model for speech-to-text

    Returns:
        AudioAnalysis with transcript and sentiment
    """
```

### analyze_image

```python
async def analyze_image(
    self,
    file_path: str,
    model: Optional[str] = None
) -> ImageAnalysis:
    """
    Analyze sentiment from image.

    Args:
        file_path: Path to image file
        model: Vision model to use

    Returns:
        ImageAnalysis with description and sentiment
    """
```

---

## Export

### export_results

```python
def export_results(
    self,
    results: Union[List[SentimentResult], AnalysisResult],
    path: str,
    format: str = "json"
) -> None:
    """
    Export results to file.

    Args:
        results: Results to export
        path: Output file path
        format: Export format (json, csv, xlsx, html)
    """
```

### visualize

```python
def visualize(
    self,
    results: Union[List[SentimentResult], AnalysisResult],
    chart_type: str = "bar",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Generate visualization of results.

    Args:
        results: Results to visualize
        chart_type: Type of chart (bar, pie, line, histogram)
        title: Chart title
        save_path: Path to save chart
        show: Display chart
    """
```

---

## Data Models

### SentimentResult

```python
@dataclass
class SentimentResult:
    label: str          # "positive", "negative", "neutral"
    score: float        # 0.0 - 1.0
    confidence: float   # Optional confidence score
    model: str          # Model used
```

### EmotionResult

```python
@dataclass
class EmotionResult:
    emotions: List[EmotionScore]
    dominant_emotion: str
    model: str
```

### Review

```python
@dataclass
class Review:
    id: str
    text: str
    source: str
    platform: str
    author: Optional[str]
    rating: Optional[float]
    timestamp: Optional[datetime]
    metadata: Dict[str, Any]
```

### AnalysisResult

```python
@dataclass
class AnalysisResult:
    url: str
    reviews: List[Review]
    sentiment: SentimentDistribution
    emotions: Optional[EmotionDistribution]
    aspects: Optional[List[AspectResult]]
    summary: Optional[str]
    insights: Optional[Dict[str, str]]
    metadata: Dict[str, Any]
```
