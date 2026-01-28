# Sentimatrix V2 - Module Specifications

## Core Modules

### 1. Configuration Module (`core/config.py`)

**Purpose:** Centralized configuration management

**Components:**
- `SentimatrixConfig`: Main configuration class
- `ProviderConfig`: Provider-specific settings
- `ModelConfig`: Model parameters
- `ScraperConfig`: Scraping settings

**Features:**
- YAML/JSON file loading
- Environment variable interpolation
- Runtime override support
- Validation via Pydantic
- Default fallbacks

**Interface:**
```python
class SentimatrixConfig:
    llm: LLMConfig
    scrapers: ScraperConfig
    models: ModelConfig
    cache: CacheConfig
    logging: LogConfig

    @classmethod
    def from_file(cls, path: str) -> "SentimatrixConfig"

    @classmethod
    def from_env(cls) -> "SentimatrixConfig"
```

---

### 2. Pipeline Module (`core/pipeline.py`)

**Purpose:** Orchestrate analysis workflows

**Components:**
- `Pipeline`: Main pipeline class
- `PipelineStep`: Individual step abstraction
- `PipelineResult`: Result container

**Features:**
- Step chaining
- Parallel execution
- Error handling and recovery
- Progress callbacks
- Middleware support

**Interface:**
```python
class Pipeline:
    def add_step(self, step: PipelineStep) -> "Pipeline"
    def remove_step(self, name: str) -> "Pipeline"
    async def execute(self, input_data: Any) -> PipelineResult
    def on_progress(self, callback: Callable) -> None
```

---

### 3. Cache Module (`core/cache.py`)

**Purpose:** Reduce redundant API calls and computations

**Components:**
- `CacheManager`: Cache orchestration
- `MemoryCache`: In-memory caching
- `RedisCache`: Distributed caching
- `SQLiteCache`: Persistent local caching

**Features:**
- TTL support
- Key namespacing
- Cache invalidation
- Compression support
- Cache statistics

**Interface:**
```python
class CacheManager:
    async def get(self, key: str) -> Optional[Any]
    async def set(self, key: str, value: Any, ttl: int = None) -> None
    async def delete(self, key: str) -> None
    async def clear(self, namespace: str = None) -> None
```

---

### 4. Logger Module (`core/logger.py`)

**Purpose:** Structured logging throughout the application

**Components:**
- `LogManager`: Logger factory
- `StructuredLogger`: JSON-formatted logger
- `LogContext`: Context propagation

**Features:**
- Structured JSON output
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Context propagation
- File and console handlers
- Log rotation

---

## Provider Modules

### 5. LLM Provider Module (`providers/llm/`)

**Purpose:** Unified interface for all LLM providers

**Supported Providers:**
| Provider | Module | Status |
|----------|--------|--------|
| OpenAI | `openai_provider.py` | Planned |
| Anthropic | `anthropic_provider.py` | Planned |
| Google Gemini | `gemini_provider.py` | Planned |
| Groq | `groq_provider.py` | Planned |
| Mistral | `mistral_provider.py` | Planned |
| Cohere | `cohere_provider.py` | Planned |
| Together AI | `together_provider.py` | Planned |
| Fireworks AI | `fireworks_provider.py` | Planned |
| Cerebras | `cerebras_provider.py` | Planned |
| DeepSeek | `deepseek_provider.py` | Planned |
| Ollama (Local) | `ollama_provider.py` | Planned |
| vLLM (Local) | `vllm_provider.py` | Planned |
| HuggingFace | `huggingface_provider.py` | Planned |

**Base Interface:**
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]

    @abstractmethod
    async def embed(self, text: str) -> List[float]

    @abstractmethod
    def supports_vision(self) -> bool

    @abstractmethod
    def supports_function_calling(self) -> bool
```

---

### 6. Scraper Provider Module (`providers/scrapers/`)

**Purpose:** Unified interface for all scraping methods

**Supported Providers:**

| Category | Provider | Module |
|----------|----------|--------|
| Browser | Playwright | `playwright_scraper.py` |
| Browser | Selenium | `selenium_scraper.py` |
| HTTP | Requests | `requests_scraper.py` |
| HTTP | HTTPX | `httpx_scraper.py` |
| API | ScraperAPI | `scraperapi_provider.py` |
| API | Bright Data | `brightdata_provider.py` |
| API | Oxylabs | `oxylabs_provider.py` |
| API | Apify | `apify_provider.py` |
| API | Zyte | `zyte_provider.py` |
| AI | Firecrawl | `firecrawl_provider.py` |
| AI | Crawl4AI | `crawl4ai_provider.py` |

**Platform Scrapers:**

| Platform | Module |
|----------|--------|
| Amazon | `platforms/amazon.py` |
| Steam | `platforms/steam.py` |
| YouTube | `platforms/youtube.py` |
| Reddit | `platforms/reddit.py` |
| IMDB | `platforms/imdb.py` |
| Twitter/X | `platforms/twitter.py` |
| TikTok | `platforms/tiktok.py` |
| Yelp | `platforms/yelp.py` |
| Trustpilot | `platforms/trustpilot.py` |
| Google Reviews | `platforms/google_reviews.py` |
| App Store | `platforms/appstore.py` |
| Play Store | `platforms/playstore.py` |
| Metacritic | `platforms/metacritic.py` |
| Rotten Tomatoes | `platforms/rottentomatoes.py` |
| LetterBoxD | `platforms/letterboxd.py` |
| Glassdoor | `platforms/glassdoor.py` |
| LinkedIn | `platforms/linkedin.py` |
| Facebook | `platforms/facebook.py` |
| Instagram | `platforms/instagram.py` |
| News Sites | `platforms/news.py` |

**Base Interface:**
```python
class BaseScraperProvider(ABC):
    @abstractmethod
    async def scrape(self, url: str, **kwargs) -> ScrapedContent

    @abstractmethod
    async def scrape_reviews(self, url: str, limit: int = 100) -> List[Review]

    @abstractmethod
    def get_supported_platforms(self) -> List[str]
```

---

### 7. Model Provider Module (`providers/models/`)

**Purpose:** Unified interface for sentiment/emotion models

**Supported Models:**

| Category | Model | Source |
|----------|-------|--------|
| Sentiment | twitter-roberta-base-sentiment | HuggingFace |
| Sentiment | cardiffnlp-sentiment | HuggingFace |
| Sentiment | distilbert-sst2 | HuggingFace |
| Emotion | roberta-base-go_emotions | HuggingFace |
| Emotion | emotion-english-distilroberta | HuggingFace |
| Aspect | deberta-absa | HuggingFace |
| Multi-lingual | xlm-roberta-sentiment | HuggingFace |
| Financial | finbert | HuggingFace |
| Zero-shot | bart-large-mnli | HuggingFace |

**Base Interface:**
```python
class BaseModelProvider(ABC):
    @abstractmethod
    async def predict(self, text: str) -> PredictionResult

    @abstractmethod
    async def predict_batch(self, texts: List[str]) -> List[PredictionResult]

    @abstractmethod
    def get_model_info(self) -> ModelInfo
```

---

## Analysis Modules

### 8. Sentiment Analysis (`analysis/sentiment.py`)

**Features:**
- Quick sentiment (positive/negative/neutral)
- Structured sentiment with confidence scores
- Comparative sentiment analysis
- Temporal sentiment tracking

---

### 9. Emotion Detection (`analysis/emotion.py`)

**Features:**
- Multi-label emotion classification
- Top-K emotion extraction
- Emotion intensity scoring
- Emotion timeline analysis

---

### 10. Aspect-Based Analysis (`analysis/aspect.py`)

**Features:**
- Aspect extraction
- Aspect sentiment mapping
- Category-based grouping
- Aspect summarization

---

### 11. Multi-modal Analysis (`analysis/multimodal.py`)

**Features:**
- Image sentiment analysis
- Audio sentiment analysis
- Video sentiment analysis
- Combined multi-modal scoring

---

## Input/Output Modules

### 12. Input Handlers (`input/`)

| Handler | Supported Formats |
|---------|-------------------|
| Text | str, List[str], file |
| Audio | WAV, MP3, FLAC, OGG |
| Image | PNG, JPG, WEBP, GIF |
| Video | MP4, AVI, MOV, WEBM |

---

### 13. Output Handlers (`output/`)

| Handler | Purpose |
|---------|---------|
| JSON Formatter | Structured JSON output |
| CSV Exporter | Tabular data export |
| HTML Reporter | Rich HTML reports |
| Visualizer | Charts and graphs |
| Webhook Sender | HTTP callbacks |
