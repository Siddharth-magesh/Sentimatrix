# Sentimatrix V2 - Configuration Reference

## Configuration Loading Order

1. Default values (built-in)
2. Config file (`config.yaml`)
3. Environment variables
4. Programmatic overrides (constructor arguments)

Later sources override earlier ones.

---

## Quick Configuration Examples

### Basic Usage (No Config Needed)

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    # Uses all defaults - no configuration required for basic sentiment/emotion
    async with Sentimatrix() as sm:
        result = await sm.analyze_sentiment("Great product!")
        print(result.sentiment)

asyncio.run(main())
```

### With LLM Provider

```python
from sentimatrix import Sentimatrix, LLMConfig

# Pass LLM config directly
async with Sentimatrix(llm_config=LLMConfig(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini"
)) as sm:
    summary = await sm.summarize_reviews(reviews)
```

### With Dict Config

```python
from sentimatrix import Sentimatrix

# Use dictionary configuration
async with Sentimatrix(config={
    "llm": {
        "provider": "groq",
        "api_key": "gsk_..."
    },
    "scraper": {
        "timeout": 60
    }
}) as sm:
    reviews = await sm.scrape_amazon("B08N5WRWNW")
```

### With Full Config Object

```python
from sentimatrix import Sentimatrix, SentimatrixConfig, LLMConfig, ScraperConfig, ModelConfig

config = SentimatrixConfig(
    llm=LLMConfig(provider="openai", api_key="sk-..."),
    scraper=ScraperConfig(timeout=30, max_retries=3),
    models=ModelConfig(device="cuda")
)

async with Sentimatrix(config=config) as sm:
    result = await sm.analyze("Great product!")
```

---

## Full Configuration Schema

```yaml
# Sentimatrix V2 Configuration

# =============================================================================
# LLM Configuration
# =============================================================================
llm:
  # Default provider to use
  provider: "groq"  # openai, anthropic, groq, gemini, mistral, ollama, etc.

  # Fallback chain if primary fails
  fallback_providers:
    - anthropic
    - openai

  # Provider-specific settings
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4o-mini"
      organization: null
      base_url: null
      timeout: 30
      max_retries: 3
      temperature: 0.7
      max_tokens: 1024

    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-3-5-sonnet-20241022"
      max_tokens: 1024
      timeout: 60

    groq:
      api_key: "${GROQ_API_KEY}"
      model: "llama-3.3-70b-versatile"
      timeout: 30

    gemini:
      api_key: "${GOOGLE_API_KEY}"
      model: "gemini-1.5-flash"

    mistral:
      api_key: "${MISTRAL_API_KEY}"
      model: "mistral-small-latest"

    ollama:
      base_url: "http://localhost:11434"
      model: "llama3.1:8b"
      timeout: 120

    together:
      api_key: "${TOGETHER_API_KEY}"
      model: "meta-llama/Llama-3.1-70B-Instruct-Turbo"

    deepseek:
      api_key: "${DEEPSEEK_API_KEY}"
      model: "deepseek-chat"

# =============================================================================
# Scraper Configuration
# =============================================================================
scrapers:
  # Default scraper provider
  provider: "playwright"  # playwright, selenium, requests, httpx

  # Browser settings (for playwright/selenium)
  browser:
    type: "chromium"  # chromium, firefox, webkit
    headless: true
    timeout: 30000
    viewport:
      width: 1920
      height: 1080

  # User agent settings
  user_agent:
    rotation: true
    type: "desktop"  # desktop, mobile, mixed

  # Rate limiting
  rate_limiting:
    enabled: true
    requests_per_second: 1.0
    concurrent_requests: 5
    per_domain: true

  # Retry settings
  retry:
    enabled: true
    max_retries: 3
    backoff_factor: 2.0
    retry_on: [429, 500, 502, 503, 504]

  # Proxy settings
  proxy:
    enabled: false
    provider: null  # brightdata, oxylabs, scraperapi, custom
    rotation: true
    urls: []
    username: null
    password: null

  # API providers
  api_providers:
    scraperapi:
      api_key: "${SCRAPERAPI_KEY}"
      render: true

    brightdata:
      username: "${BRIGHTDATA_USER}"
      password: "${BRIGHTDATA_PASS}"
      zone: "residential"

    apify:
      api_token: "${APIFY_TOKEN}"

# =============================================================================
# Model Configuration
# =============================================================================
models:
  # Sentiment model
  sentiment:
    model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
    device: "auto"  # auto, cpu, cuda, mps
    batch_size: 32
    max_length: 512

  # Emotion model
  emotion:
    model: "SamLowe/roberta-base-go_emotions"
    device: "auto"
    top_k: 5
    threshold: 0.3

  # Aspect-based model
  aspect:
    model: "yangheng/deberta-v3-base-absa-v1.1"
    device: "auto"

  # Translation (for multilingual)
  translation:
    enabled: true
    source_lang: "auto"
    target_lang: "en"

# =============================================================================
# Cache Configuration
# =============================================================================
cache:
  enabled: true
  backend: "memory"  # memory, redis, sqlite

  # TTL settings (seconds)
  ttl:
    default: 3600
    scraping: 86400
    sentiment: 604800
    llm: 3600

  # Backend-specific settings
  memory:
    max_size: 1000

  redis:
    url: "${REDIS_URL}"
    prefix: "sentimatrix:"

  sqlite:
    path: "./cache/sentimatrix.db"

# =============================================================================
# Logging Configuration
# =============================================================================
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "text"  # text, json

  # Output destinations
  handlers:
    console:
      enabled: true
      level: "INFO"

    file:
      enabled: false
      path: "./logs/sentimatrix.log"
      rotation: "daily"
      retention: 7

  # What to log
  log_requests: false
  log_responses: false
  log_timing: true

# =============================================================================
# Output Configuration
# =============================================================================
output:
  # Default export format
  format: "json"  # json, csv, xlsx

  # Visualization settings
  visualization:
    theme: "default"
    width: 800
    height: 600
    save_format: "png"  # png, svg, pdf

  # Include options
  include:
    raw_text: true
    confidence_scores: true
    processing_time: true
    metadata: true

# =============================================================================
# Performance Configuration
# =============================================================================
performance:
  # Batch processing
  batch_size: 32
  max_concurrent: 10

  # Memory management
  gc_after_batch: true
  max_memory_mb: 4096

  # Model loading
  lazy_loading: true
  unload_after_seconds: 300

# =============================================================================
# Debug Configuration
# =============================================================================
debug:
  enabled: false
  profiling: false
  save_html: false
  trace_requests: false
```

---

## Environment Variables

All configuration values can be set via environment variables using the pattern:

`SENTIMATRIX_<SECTION>_<KEY>`

### Examples

```bash
# LLM settings
export SENTIMATRIX_LLM_PROVIDER=openai
export SENTIMATRIX_LLM_PROVIDERS_OPENAI_MODEL=gpt-4o

# Scraper settings
export SENTIMATRIX_SCRAPERS_PROVIDER=playwright
export SENTIMATRIX_SCRAPERS_BROWSER_HEADLESS=true

# Cache settings
export SENTIMATRIX_CACHE_ENABLED=true
export SENTIMATRIX_CACHE_BACKEND=redis

# API keys (standard names also work)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk_...
```

---

## Provider-Specific Configs

### OpenAI

```yaml
openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  organization: "org-..."  # Optional
  base_url: null           # For Azure or proxies
  timeout: 30
  max_retries: 3
```

### Anthropic

```yaml
anthropic:
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-3-5-sonnet-20241022"
  max_tokens: 1024
```

### Groq

```yaml
groq:
  api_key: "${GROQ_API_KEY}"
  model: "llama-3.3-70b-versatile"
```

### Ollama (Local)

```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "llama3.1:8b"
  timeout: 120
  options:
    temperature: 0.7
    num_ctx: 4096
```

---

## Platform-Specific Scraper Configs

### Amazon

```yaml
platforms:
  amazon:
    method: "playwright"  # playwright, scraperapi
    country: "us"
    sort: "recent"
    filter_verified: false
```

### YouTube

```yaml
platforms:
  youtube:
    api_key: "${YOUTUBE_API_KEY}"
    include_replies: true
    max_results: 100
```

### Reddit

```yaml
platforms:
  reddit:
    client_id: "${REDDIT_CLIENT_ID}"
    client_secret: "${REDDIT_SECRET}"
    user_agent: "Sentimatrix/0.2.0"
```

---

## Minimal Configurations

### Development (Local)

```yaml
llm:
  provider: ollama
scrapers:
  provider: playwright
  browser:
    headless: false
cache:
  enabled: false
logging:
  level: DEBUG
```

### Production (Cloud)

```yaml
llm:
  provider: openai
  fallback_providers: [anthropic, groq]
scrapers:
  provider: playwright
  proxy:
    enabled: true
    provider: brightdata
cache:
  enabled: true
  backend: redis
logging:
  level: WARNING
  format: json
```

### Cost-Optimized

```yaml
llm:
  provider: groq
  fallback_providers: [deepseek]
scrapers:
  provider: requests  # No browser overhead
cache:
  enabled: true
  ttl:
    default: 86400  # Cache for 24h
```
