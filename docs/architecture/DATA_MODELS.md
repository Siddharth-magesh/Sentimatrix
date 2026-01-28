# Sentimatrix V2 - Data Models

## Core Data Models

### 1. Review

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

### 2. SentimentResult

```python
@dataclass
class SentimentResult:
    label: str              # "positive", "negative", "neutral"
    score: float            # 0.0 - 1.0
    confidence: float       # 0.0 - 1.0
    model_used: str
    processing_time_ms: float
```

### 3. EmotionResult

```python
@dataclass
class EmotionResult:
    emotions: List[EmotionScore]
    dominant_emotion: str
    model_used: str
    processing_time_ms: float

@dataclass
class EmotionScore:
    label: str              # "joy", "anger", "sadness", etc.
    score: float            # 0.0 - 1.0
```

### 4. AspectResult

```python
@dataclass
class AspectResult:
    aspects: List[AspectSentiment]
    overall_sentiment: SentimentResult
    model_used: str

@dataclass
class AspectSentiment:
    aspect: str             # "battery", "screen", "price"
    sentiment: str          # "positive", "negative", "neutral"
    score: float
    mentions: List[str]     # Relevant text snippets
```

### 5. AnalysisResult

```python
@dataclass
class AnalysisResult:
    id: str
    input_type: str         # "text", "url", "audio", "image"
    input_data: Any
    sentiment: Optional[SentimentResult]
    emotions: Optional[EmotionResult]
    aspects: Optional[AspectResult]
    llm_summary: Optional[str]
    llm_insights: Optional[Dict[str, str]]
    raw_reviews: Optional[List[Review]]
    metadata: Dict[str, Any]
    created_at: datetime
    processing_time_ms: float
```

---

## Scraper Data Models

### 6. ScrapedContent

```python
@dataclass
class ScrapedContent:
    url: str
    title: Optional[str]
    content: str
    html: Optional[str]
    reviews: List[Review]
    metadata: ScraperMetadata
    scraped_at: datetime
```

### 7. ScraperMetadata

```python
@dataclass
class ScraperMetadata:
    provider: str           # "playwright", "selenium", "api"
    status_code: int
    response_time_ms: float
    proxy_used: Optional[str]
    user_agent: str
    cookies: Dict[str, str]
    headers: Dict[str, str]
```

---

## LLM Data Models

### 8. LLMRequest

```python
@dataclass
class LLMRequest:
    prompt: str
    system_prompt: Optional[str]
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    stream: bool = False
    functions: Optional[List[Dict]] = None
```

### 9. LLMResponse

```python
@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: TokenUsage
    finish_reason: str
    response_time_ms: float

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

---

## Configuration Data Models

### 10. SentimatrixConfig

```python
@dataclass
class SentimatrixConfig:
    llm: LLMConfig
    scrapers: ScraperConfig
    models: ModelConfig
    cache: CacheConfig
    logging: LogConfig
    output: OutputConfig
```

### 11. LLMConfig

```python
@dataclass
class LLMConfig:
    provider: str           # "openai", "anthropic", "groq", etc.
    model: str
    api_key: Optional[str]
    api_base: Optional[str]
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 1024
```

### 12. ScraperConfig

```python
@dataclass
class ScraperConfig:
    provider: str           # "playwright", "selenium", "api"
    headless: bool = True
    timeout: int = 30
    proxy: Optional[ProxyConfig]
    user_agent: Optional[str]
    rate_limit: RateLimitConfig
    retry: RetryConfig
```

### 13. ProxyConfig

```python
@dataclass
class ProxyConfig:
    enabled: bool = False
    provider: Optional[str]  # "brightdata", "oxylabs", "custom"
    url: Optional[str]
    username: Optional[str]
    password: Optional[str]
    rotation: bool = True
```

### 14. RateLimitConfig

```python
@dataclass
class RateLimitConfig:
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    concurrent_requests: int = 5
    backoff_factor: float = 2.0
```

### 15. CacheConfig

```python
@dataclass
class CacheConfig:
    enabled: bool = True
    backend: str = "memory"  # "memory", "redis", "sqlite"
    ttl: int = 3600          # seconds
    max_size: int = 1000     # entries
    redis_url: Optional[str]
    sqlite_path: Optional[str]
```

---

## Output Data Models

### 16. ExportOptions

```python
@dataclass
class ExportOptions:
    format: str             # "json", "csv", "html", "xlsx"
    path: str
    include_raw: bool = False
    include_metadata: bool = True
    compression: Optional[str]  # "gzip", "zip"
```

### 17. VisualizationOptions

```python
@dataclass
class VisualizationOptions:
    chart_type: str         # "bar", "pie", "line", "heatmap"
    title: str
    width: int = 800
    height: int = 600
    theme: str = "default"
    save_path: Optional[str]
    show: bool = True
```

---

## Batch Processing Models

### 18. BatchJob

```python
@dataclass
class BatchJob:
    id: str
    items: List[Any]
    status: str             # "pending", "running", "completed", "failed"
    progress: float         # 0.0 - 1.0
    results: List[AnalysisResult]
    errors: List[BatchError]
    created_at: datetime
    completed_at: Optional[datetime]
```

### 19. BatchError

```python
@dataclass
class BatchError:
    item_index: int
    error_type: str
    error_message: str
    traceback: Optional[str]
```

---

## Comparison Models

### 20. ComparisonResult

```python
@dataclass
class ComparisonResult:
    items: List[ComparisonItem]
    summary: str
    winner: Optional[str]
    comparison_aspects: List[AspectComparison]
    generated_at: datetime

@dataclass
class ComparisonItem:
    name: str
    url: Optional[str]
    overall_sentiment: SentimentResult
    review_count: int
    average_rating: Optional[float]

@dataclass
class AspectComparison:
    aspect: str
    scores: Dict[str, float]  # item_name -> score
    winner: str
```
