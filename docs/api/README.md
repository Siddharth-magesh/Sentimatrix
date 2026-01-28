# Sentimatrix V2 API Reference

This document provides comprehensive API documentation for all Sentimatrix V2 modules.

## Table of Contents

1. [Core Module](#core-module)
2. [Analysis Module](#analysis-module)
3. [Providers Module](#providers-module)
4. [Input Module](#input-module)
5. [Output Module](#output-module)
6. [Utils Module](#utils-module)

---

## Core Module

### Pipeline (`sentimatrix.core.pipeline`)

The pipeline module provides a flexible system for chaining analysis steps.

#### Classes

##### `Pipeline`

Main pipeline orchestrator for chaining analysis steps.

```python
from sentimatrix.core.pipeline import Pipeline, FunctionStep

pipeline = Pipeline(name="my_analysis")
pipeline.add_step(FunctionStep("step1", my_function))
result = await pipeline.run()
```

**Constructor Parameters:**
- `name: str` - Pipeline identifier
- `config: Optional[PipelineConfig]` - Pipeline configuration

**Methods:**
- `add_step(step: PipelineStep) -> None` - Add a step to the pipeline
- `async run(initial_input: Any = None) -> PipelineResult` - Execute the pipeline
- `get_steps() -> List[PipelineStep]` - Get all pipeline steps

##### `PipelineStep`

Abstract base class for pipeline steps.

**Methods:**
- `async execute(input_data: Any) -> Any` - Execute the step

##### `FunctionStep`

Pipeline step that wraps an async function.

```python
async def my_analyzer(ctx: PipelineContext, prev_result):
    return {"analyzed": True}

step = FunctionStep("analyzer", my_analyzer)
```

**Constructor Parameters:**
- `name: str` - Step name
- `func: Callable` - Async function to execute
- `config: Optional[StepConfig]` - Step configuration

##### `ParallelSteps`

Execute multiple steps in parallel.

```python
parallel = ParallelSteps(
    name="parallel_analysis",
    steps=[step1, step2, step3]
)
```

##### `PipelineContext`

Shared context for passing data between pipeline steps.

**Methods:**
- `set(key: str, value: Any) -> None` - Store a value
- `get(key: str, default: Any = None) -> Any` - Retrieve a value
- `has(key: str) -> bool` - Check if key exists

##### `PipelineResult`

Result of pipeline execution.

**Attributes:**
- `success: bool` - Whether pipeline completed successfully
- `output: Any` - Final output from the last step
- `error: Optional[Exception]` - Error if pipeline failed
- `step_results: Dict[str, Any]` - Results from each step
- `execution_time: float` - Total execution time in seconds

---

### Configuration (`sentimatrix.core.config`)

Configuration management for Sentimatrix.

#### Classes

##### `SentimatrixConfig`

Main configuration class.

```python
from sentimatrix.core.config import SentimatrixConfig

config = SentimatrixConfig(
    cache_enabled=True,
    cache_ttl=3600,
    max_concurrent_requests=10
)
```

**Attributes:**
- `cache_enabled: bool` - Enable caching (default: True)
- `cache_ttl: int` - Cache time-to-live in seconds (default: 3600)
- `max_concurrent_requests: int` - Maximum concurrent requests (default: 10)
- `default_provider: str` - Default LLM provider (default: "openai")

---

### Exceptions (`sentimatrix.core.exceptions`)

Custom exceptions for Sentimatrix.

```python
from sentimatrix.core.exceptions import (
    SentimatrixError,
    PipelineError,
    PipelineStepError,
    ProviderError,
    ValidationError,
    CacheError,
)
```

| Exception | Description |
|-----------|-------------|
| `SentimatrixError` | Base exception for all Sentimatrix errors |
| `PipelineError` | Pipeline execution errors |
| `PipelineStepError` | Individual step failures |
| `ProviderError` | LLM provider errors |
| `ValidationError` | Input validation errors |
| `CacheError` | Cache operation errors |

---

## Analysis Module

### Sentiment Analysis (`sentimatrix.analysis.sentiment`)

Sentiment analysis functionality.

#### Classes

##### `SentimentAnalyzer`

Main class for sentiment analysis.

```python
from sentimatrix.analysis.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer(model="default")
result = await analyzer.analyze("This product is great!")
```

**Constructor Parameters:**
- `model: str` - Model to use (default: "default")
- `config: Optional[SentimentConfig]` - Analysis configuration

**Methods:**
- `async analyze(text: str) -> SentimentResult` - Analyze single text
- `async analyze_batch(texts: List[str]) -> List[SentimentResult]` - Batch analysis

##### `SentimentResult`

Result of sentiment analysis.

**Attributes:**
- `sentiment: SentimentLabel` - Sentiment label (POSITIVE, NEGATIVE, NEUTRAL)
- `confidence: float` - Confidence score (0.0 to 1.0)
- `scores: Dict[str, float]` - Score breakdown by label

##### `SentimentLabel`

Enum for sentiment labels.

```python
from sentimatrix.analysis.sentiment import SentimentLabel

SentimentLabel.POSITIVE
SentimentLabel.NEGATIVE
SentimentLabel.NEUTRAL
```

---

### Emotion Detection (`sentimatrix.analysis.emotion`)

Emotion detection functionality.

#### Classes

##### `EmotionDetector`

Main class for emotion detection.

```python
from sentimatrix.analysis.emotion import EmotionDetector

detector = EmotionDetector()
result = await detector.detect("I'm so happy today!")
```

**Methods:**
- `async detect(text: str) -> EmotionResult` - Detect emotions in text
- `async detect_batch(texts: List[str]) -> List[EmotionResult]` - Batch detection

##### `EmotionResult`

Result of emotion detection.

**Attributes:**
- `primary_emotion: str` - Primary detected emotion
- `emotions: List[EmotionScore]` - All detected emotions with scores
- `confidence: float` - Overall confidence

##### `EmotionScore`

Individual emotion score.

**Attributes:**
- `label: str` - Emotion label (joy, anger, sadness, fear, surprise, disgust)
- `score: float` - Score for this emotion (0.0 to 1.0)

---

### Aggregation (`sentimatrix.analysis.aggregator`)

Aggregate analysis results.

#### Classes

##### `ResultAggregator`

Aggregate multiple analysis results.

```python
from sentimatrix.analysis.aggregator import ResultAggregator

aggregator = ResultAggregator()
summary = aggregator.aggregate(results)
```

**Methods:**
- `aggregate(results: List[SentimentResult]) -> AggregatedResult` - Aggregate results
- `get_statistics(results: List[SentimentResult]) -> Dict` - Get statistical summary

---

## Providers Module

### LLM Providers (`sentimatrix.providers.llm`)

LLM provider integrations.

#### Base Class

##### `BaseLLMProvider`

Abstract base class for LLM providers.

```python
from sentimatrix.providers.llm.base import BaseLLMProvider

class CustomProvider(BaseLLMProvider):
    async def generate(self, prompt: str) -> str:
        ...
```

**Methods:**
- `async generate(prompt: str, **kwargs) -> str` - Generate text
- `async initialize() -> None` - Initialize the provider
- `async close() -> None` - Clean up resources

#### Available Providers

##### `OpenAIProvider`

```python
from sentimatrix.providers.llm import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
```

##### `AnthropicProvider`

```python
from sentimatrix.providers.llm import AnthropicProvider

provider = AnthropicProvider(api_key="...", model="claude-3-sonnet")
```

##### `OllamaProvider`

```python
from sentimatrix.providers.llm import OllamaProvider

provider = OllamaProvider(model="llama2", base_url="http://localhost:11434")
```

##### `GroqProvider`

```python
from sentimatrix.providers.llm import GroqProvider

provider = GroqProvider(api_key="...", model="mixtral-8x7b")
```

---

### Scrapers (`sentimatrix.providers.scrapers`)

Web scraping functionality.

#### Classes

##### `BaseScraper`

Base class for scrapers.

##### `AmazonScraper`

Scrape Amazon reviews.

```python
from sentimatrix.providers.scrapers import AmazonScraper

scraper = AmazonScraper()
reviews = await scraper.scrape("B08N5WRWNW", max_reviews=100)
```

##### `YelpScraper`

Scrape Yelp reviews.

```python
from sentimatrix.providers.scrapers import YelpScraper

scraper = YelpScraper()
reviews = await scraper.scrape("restaurant-name", max_reviews=50)
```

##### `GooglePlayScraper`

Scrape Google Play reviews.

```python
from sentimatrix.providers.scrapers import GooglePlayScraper

scraper = GooglePlayScraper()
reviews = await scraper.scrape("com.example.app", max_reviews=100)
```

##### `RateLimiter`

Rate limiting for scrapers.

```python
from sentimatrix.providers.scrapers.rate_limiter import RateLimiter, RateLimitStrategy

limiter = RateLimiter(
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    requests_per_second=5.0,
    burst_size=10
)

await limiter.acquire()  # Wait for rate limit
await limiter.acquire("example.com")  # Per-domain limiting
```

---

## Input Module

### Input Handlers (`sentimatrix.input`)

Input processing and validation.

#### Classes

##### `TextInput`

Handle text input.

```python
from sentimatrix.input import TextInput

input_handler = TextInput()
processed = input_handler.process("My review text")
```

##### `BatchInput`

Handle batch text input.

```python
from sentimatrix.input import BatchInput

batch = BatchInput()
processed = batch.process(["text1", "text2", "text3"])
```

##### `FileInput`

Handle file input.

```python
from sentimatrix.input import FileInput

file_input = FileInput()
texts = file_input.read("reviews.csv")
```

##### `InputValidator`

Validate input data.

```python
from sentimatrix.input import InputValidator

validator = InputValidator()
is_valid = validator.validate(text)
```

---

## Output Module

### Formatters (`sentimatrix.output`)

Output formatting and export.

#### Classes

##### `JSONFormatter`

Format results as JSON.

```python
from sentimatrix.output import JSONFormatter

formatter = JSONFormatter()
json_output = formatter.format(results)
formatter.write("output.json", results)
```

##### `CSVFormatter`

Format results as CSV.

```python
from sentimatrix.output import CSVFormatter

formatter = CSVFormatter()
csv_output = formatter.format(results)
formatter.write("output.csv", results)
```

##### `HTMLFormatter`

Format results as HTML report.

```python
from sentimatrix.output import HTMLFormatter

formatter = HTMLFormatter(template="report")
html_output = formatter.format(results)
```

##### `ReportGenerator`

Generate comprehensive reports.

```python
from sentimatrix.output import ReportGenerator

generator = ReportGenerator()
report = generator.generate(results, format="html")
```

---

## Utils Module

### Caching (`sentimatrix.utils.cache`)

Caching utilities.

#### Classes

##### `CacheManager`

Manage caching operations.

```python
from sentimatrix.utils.cache import CacheManager

cache = CacheManager(backend="memory", ttl=3600)
await cache.set("key", value)
result = await cache.get("key")
```

##### `RedisCache`

Redis-backed cache.

```python
from sentimatrix.utils.cache import RedisCache

cache = RedisCache(
    host="localhost",
    port=6379,
    db=0,
    ttl=3600
)
```

---

### Logging (`sentimatrix.utils.logging`)

Logging utilities.

```python
from sentimatrix.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
```

---

### Text Processing (`sentimatrix.utils.text`)

Text processing utilities.

```python
from sentimatrix.utils.text import (
    clean_text,
    tokenize,
    normalize,
    detect_language,
)

cleaned = clean_text(raw_text)
tokens = tokenize(cleaned)
normalized = normalize(cleaned)
lang = detect_language(cleaned)
```

---

## Type Definitions

Common types used throughout Sentimatrix.

```python
from sentimatrix.providers.models import Review, AnalysisResult

# Review data structure
review = Review(
    id="R123",
    text="Great product!",
    rating=5,
    author="John",
    source="amazon"
)

# Analysis result structure
result = AnalysisResult(
    text="Great product!",
    sentiment="positive",
    confidence=0.95,
    emotions={"joy": 0.8}
)
```

---

## Error Handling

All Sentimatrix operations can raise specific exceptions:

```python
from sentimatrix.core.exceptions import (
    SentimatrixError,
    ProviderError,
    ValidationError,
)

try:
    result = await analyzer.analyze(text)
except ValidationError as e:
    print(f"Invalid input: {e}")
except ProviderError as e:
    print(f"Provider failed: {e}")
except SentimatrixError as e:
    print(f"General error: {e}")
```

---

## Async Context Managers

Most Sentimatrix components support async context managers:

```python
async with SentimentAnalyzer() as analyzer:
    result = await analyzer.analyze(text)
# Resources automatically cleaned up
```

---

## Configuration via Environment Variables

Sentimatrix supports configuration via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SENTIMATRIX_CACHE_ENABLED` | Enable caching | `true` |
| `SENTIMATRIX_CACHE_TTL` | Cache TTL in seconds | `3600` |
| `SENTIMATRIX_LOG_LEVEL` | Logging level | `INFO` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GROQ_API_KEY` | Groq API key | - |
| `REDIS_URL` | Redis connection URL | - |
