---
title: API Reference
description: Complete API reference for Sentimatrix
---

# API Reference

Complete reference documentation for the Sentimatrix Python API.

## Core Classes

<div class="grid">

<div class="card">
<h3>:material-application: Sentimatrix</h3>
<p>Main entry point for all operations.</p>
<p><a href="sentimatrix/">View docs →</a></p>
</div>

<div class="card">
<h3>:material-cog: Config</h3>
<p>Configuration objects and schemas.</p>
<p><a href="config/">View docs →</a></p>
</div>

<div class="card">
<h3>:material-api: Providers</h3>
<p>LLM provider interfaces and managers.</p>
<p><a href="base-provider/">View docs →</a></p>
</div>

<div class="card">
<h3>:material-spider-web: Scrapers</h3>
<p>Web scraper interfaces and managers.</p>
<p><a href="base-scraper/">View docs →</a></p>
</div>

</div>

## Quick Reference

### Main Class

```python
from sentimatrix import Sentimatrix

async with Sentimatrix(config=None) as sm:
    # Sentiment Analysis
    result = await sm.analyze(text: str) -> SentimentResult
    results = await sm.analyze_batch(texts: list[str]) -> list[SentimentResult]

    # Emotion Detection
    emotions = await sm.detect_emotions(text: str) -> EmotionResult

    # Web Scraping
    reviews = await sm.scrape_reviews(url: str, platform: str) -> list[Review]

    # LLM Features (requires LLM config)
    summary = await sm.summarize_reviews(reviews: list) -> str
    insights = await sm.generate_insights(reviews: list) -> Insights
```

### Configuration

```python
from sentimatrix.config import (
    SentimatrixConfig,
    LLMConfig,
    ScraperConfig,
    ModelConfig,
    CacheConfig,
)

config = SentimatrixConfig(
    llm=LLMConfig(...),
    scraper=ScraperConfig(...),
    model=ModelConfig(...),
    cache=CacheConfig(...),
)
```

### Result Types

```python
from sentimatrix.models import (
    SentimentResult,
    EmotionResult,
    Review,
    Insights,
)
```

## Sentimatrix Class

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

    async def __aenter__(self) -> "Sentimatrix":
        """Async context manager entry."""

    async def __aexit__(self, *args) -> None:
        """Async context manager exit. Cleans up resources."""
```

### Sentiment Analysis Methods

```python
async def analyze(
    self,
    text: str,
    *,
    mode: str = "quick",
    structured: bool = False,
    domain: str = "general",
) -> SentimentResult:
    """
    Analyze sentiment of a single text.

    Args:
        text: Text to analyze.
        mode: Analysis mode ("quick", "fine_grained").
        structured: Include detailed scores.
        domain: Domain for specialized analysis.

    Returns:
        SentimentResult with sentiment label and confidence.
    """

async def analyze_batch(
    self,
    texts: list[str],
    *,
    mode: str = "quick",
    batch_size: int = 32,
) -> list[SentimentResult]:
    """
    Analyze sentiment of multiple texts efficiently.

    Args:
        texts: List of texts to analyze.
        mode: Analysis mode.
        batch_size: Processing batch size.

    Returns:
        List of SentimentResult objects.
    """

async def analyze_aspects(
    self,
    text: str,
    aspects: list[str],
    *,
    include_scores: bool = False,
) -> dict[str, str | dict]:
    """
    Analyze sentiment per aspect.

    Args:
        text: Text to analyze.
        aspects: List of aspects to extract sentiment for.
        include_scores: Include confidence scores.

    Returns:
        Dictionary mapping aspects to sentiments.
    """
```

### Emotion Detection Methods

```python
async def detect_emotions(
    self,
    text: str,
    *,
    taxonomy: str = "goemotion",
    mode: str = "single_label",
    threshold: float = 0.3,
) -> EmotionResult:
    """
    Detect emotions in text.

    Args:
        text: Text to analyze.
        taxonomy: Emotion taxonomy ("ekman", "goemotion", "plutchik").
        mode: Detection mode ("single_label", "multi_label", "top_k").
        threshold: Confidence threshold for multi_label mode.

    Returns:
        EmotionResult with detected emotions.
    """
```

### Web Scraping Methods

```python
async def scrape_reviews(
    self,
    url: str,
    platform: str,
    *,
    max_reviews: int = 50,
    use_browser: bool = False,
    **kwargs,
) -> list[Review]:
    """
    Scrape reviews from a platform.

    Args:
        url: URL or product identifier.
        platform: Platform name ("amazon", "steam", "youtube", etc.).
        max_reviews: Maximum reviews to scrape.
        use_browser: Enable browser for JS-rendered pages.
        **kwargs: Platform-specific options.

    Returns:
        List of Review objects.
    """
```

### LLM Feature Methods

```python
async def summarize_reviews(
    self,
    reviews: list[dict | Review],
    *,
    style: str = "professional",
    max_length: int = 500,
) -> str:
    """
    Generate a summary of reviews using LLM.

    Args:
        reviews: List of reviews to summarize.
        style: Summary style ("professional", "casual", "bullet_points").
        max_length: Approximate max length of summary.

    Returns:
        Summary string.

    Raises:
        ConfigurationError: If LLM not configured.
    """

async def generate_insights(
    self,
    reviews: list[dict | Review],
) -> Insights:
    """
    Generate insights from reviews using LLM.

    Args:
        reviews: List of reviews to analyze.

    Returns:
        Insights object with pros, cons, recommendation.

    Raises:
        ConfigurationError: If LLM not configured.
    """
```

## Result Types

### SentimentResult

```python
@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    sentiment: str
    """Sentiment label: "positive", "negative", or "neutral"."""

    confidence: float
    """Confidence score between 0 and 1."""

    scores: dict[str, float] | None = None
    """Per-class scores (if structured=True)."""

    text: str | None = None
    """Original text (if included)."""
```

### EmotionResult

```python
@dataclass
class EmotionResult:
    """Result of emotion detection."""

    primary: str
    """Primary detected emotion."""

    confidence: float
    """Confidence of primary emotion."""

    scores: dict[str, float]
    """All emotion scores."""

    labels: list[str] | None = None
    """Detected labels (multi_label mode)."""

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Get top k emotions by score."""
```

### Review

```python
@dataclass
class Review:
    """Scraped review data."""

    text: str
    """Review text content."""

    rating: int | float | bool | None = None
    """Review rating (format varies by platform)."""

    title: str | None = None
    """Review title (if available)."""

    author: str | None = None
    """Author name or ID."""

    posted_date: datetime | None = None
    """When the review was posted."""

    helpful_count: int = 0
    """Number of helpful votes."""

    platform: str = ""
    """Source platform."""

    metadata: dict = field(default_factory=dict)
    """Platform-specific metadata."""
```

### InsightsResult

```python
@dataclass
class InsightsResult:
    """LLM-generated insights."""

    summary: str
    """Brief overview."""

    key_points: list[str]
    """Main takeaways."""

    pros: list[str]
    """List of positive aspects."""

    cons: list[str]
    """List of negative aspects."""

    themes: list[str]
    """Common themes mentioned."""

    recommendations: list[str]
    """Actionable suggestions."""

    raw_response: str
    """Full LLM response."""
```

### LLMResponse

```python
@dataclass
class LLMResponse:
    """Response from LLM provider."""

    content: str
    """Generated text content."""

    model: str
    """Model used for generation."""

    provider: str
    """Provider name."""

    usage: TokenUsage
    """Token usage statistics."""

    finish_reason: str
    """Why generation stopped."""

    response_time_ms: float
    """Response time in milliseconds."""

    raw_response: Optional[dict]
    """Raw API response."""

    tool_calls: Optional[list]
    """Function/tool calls if any."""
```

### TokenUsage

```python
@dataclass
class TokenUsage:
    """Token usage for LLM request."""

    prompt_tokens: int
    """Tokens in prompt."""

    completion_tokens: int
    """Tokens in response."""

    total_tokens: int
    """Total tokens used."""
```

### ReviewAnalysisResult

```python
@dataclass
class ReviewAnalysisResult:
    """Aggregated analysis of multiple reviews."""

    reviews: list[AnalysisResult]
    """Individual review results."""

    sentiment_summary: Optional[dict]
    """Aggregated sentiment stats."""

    emotion_summary: Optional[dict]
    """Aggregated emotion stats."""

    total_count: int
    """Number of reviews analyzed."""

    @property
    def positive_ratio(self) -> float:
        """Ratio of positive reviews."""

    @property
    def negative_ratio(self) -> float:
        """Ratio of negative reviews."""

    @property
    def average_polarity(self) -> float:
        """Average polarity score (-1 to 1)."""
```

### ComparisonResult

```python
@dataclass
class ComparisonResult:
    """Product/service comparison result."""

    product_a_id: str
    """First product identifier."""

    product_b_id: str
    """Second product identifier."""

    analysis_a: ReviewAnalysisResult
    """Analysis of first product."""

    analysis_b: ReviewAnalysisResult
    """Analysis of second product."""

    comparison_summary: str
    """LLM-generated comparison."""

    winner: Optional[str]
    """Recommended product (if clear winner)."""
```

## Exceptions

Sentimatrix provides a comprehensive exception hierarchy with 30+ exception types and 50+ error codes:

```python
from sentimatrix.exceptions import (
    # Base
    SentimatrixError,           # Base exception

    # Configuration
    ConfigurationError,         # Invalid configuration
    ConfigNotFoundError,        # Config file not found

    # Validation
    ValidationError,            # Validation failed
    InvalidInputError,          # Invalid input data

    # Provider errors
    ProviderError,              # Base provider error
    ProviderNotFoundError,      # Provider not registered

    # LLM errors
    LLMProviderError,           # LLM provider error
    AuthenticationError,        # Invalid API key
    TokenLimitExceededError,    # Context too long
    QuotaExceededError,         # API quota exceeded

    # Scraper errors
    ScraperError,               # Base scraper error
    ScraperBlockedError,        # IP/access blocked
    CaptchaDetectedError,       # CAPTCHA encountered
    RateLimitError,             # Rate limit exceeded

    # Model errors
    ModelError,                 # Base model error
    ModelLoadError,             # Failed to load model
    ModelInferenceError,        # Inference failed

    # Cache errors
    CacheError,                 # Cache operation failed
    CacheReadError,             # Read from cache failed
    CacheWriteError,            # Write to cache failed

    # Pipeline errors
    PipelineError,              # Pipeline execution failed
    PipelineStepError,          # Step failed
)
```

### Error Codes

Error codes are standardized for programmatic handling:

```python
from sentimatrix.exceptions import ErrorCode

# Configuration errors: 1000-1099
ErrorCode.CONFIG_NOT_FOUND      # 1001
ErrorCode.CONFIG_INVALID        # 1002

# Validation errors: 1100-1199
ErrorCode.VALIDATION_ERROR      # 1100
ErrorCode.INVALID_INPUT         # 1101

# Provider errors: 1200-1299
ErrorCode.PROVIDER_NOT_FOUND    # 1200
ErrorCode.PROVIDER_ERROR        # 1201

# LLM errors: 1300-1399
ErrorCode.LLM_ERROR             # 1300
ErrorCode.AUTHENTICATION_ERROR  # 1301
ErrorCode.TOKEN_LIMIT_EXCEEDED  # 1302
ErrorCode.RATE_LIMIT_EXCEEDED   # 1303

# Scraper errors: 1400-1499
ErrorCode.SCRAPER_ERROR         # 1400
ErrorCode.BLOCKED_ERROR         # 1401
ErrorCode.CAPTCHA_DETECTED      # 1402

# Model errors: 1500-1599
ErrorCode.MODEL_ERROR           # 1500
ErrorCode.MODEL_LOAD_ERROR      # 1501
```

## Usage Patterns

### Basic Usage

```python
async with Sentimatrix() as sm:
    result = await sm.analyze("Great product!")
    print(result.sentiment)
```

### With Configuration

```python
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

### Manual Initialization

```python
sm = Sentimatrix(config)
await sm.initialize()

try:
    result = await sm.analyze("Hello")
finally:
    await sm.close()
```

### Error Handling

```python
from sentimatrix.exceptions import ScraperError, ProviderError

async with Sentimatrix(config) as sm:
    try:
        reviews = await sm.scrape_reviews(url, platform="amazon")
        summary = await sm.summarize_reviews(reviews)
    except ScraperError as e:
        print(f"Scraping failed: {e}")
    except ProviderError as e:
        print(f"LLM error: {e}")
```

## Related

- [Configuration](../configuration/index.md)
- [Providers](../providers/index.md)
- [Scrapers](../scrapers/index.md)
