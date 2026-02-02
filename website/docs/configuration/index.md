---
title: Configuration
description: Configure Sentimatrix for your needs
---

# Configuration

Sentimatrix uses Pydantic V2 for centralized configuration management with full validation, type safety, and multiple loading methods.

## Configuration Methods

<div class="grid">

<div class="card">
<h3>:material-file-cog: YAML/JSON Files</h3>
<p>Best for: Production deployments, version control</p>
<p><a href="yaml/">Learn more →</a></p>
</div>

<div class="card">
<h3>:material-cog: Environment Variables</h3>
<p>Best for: Docker, CI/CD, secrets</p>
<p><a href="environment/">Learn more →</a></p>
</div>

<div class="card">
<h3>:material-language-python: Python Objects</h3>
<p>Best for: Dynamic configuration, testing</p>
<p><a href="python/">Learn more →</a></p>
</div>

</div>

## Quick Start

### Minimal Configuration

```python
# No config needed for basic usage
async with Sentimatrix() as sm:
    result = await sm.analyze("Hello world!")
```

### With LLM Provider

=== "Python"

    ```python
    from sentimatrix import Sentimatrix
    from sentimatrix.config import SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
    )

    async with Sentimatrix(config) as sm:
        summary = await sm.summarize_reviews(reviews)
    ```

=== "YAML"

    ```yaml title="sentimatrix.yaml"
    llm:
      provider: groq
      model: llama-3.3-70b-versatile
    ```

    ```python
    # Load from specific file
    config = SentimatrixConfig.from_file("sentimatrix.yaml")
    async with Sentimatrix(config) as sm:
        summary = await sm.summarize_reviews(reviews)
    ```

=== "Environment"

    ```bash
    export GROQ_API_KEY="gsk_..."
    export SENTIMATRIX_LLM__PROVIDER="groq"
    export SENTIMATRIX_LLM__MODEL="llama-3.3-70b-versatile"
    ```

    ```python
    # Automatically loads from environment
    config = SentimatrixConfig.from_env()
    ```

## Configuration Classes

Sentimatrix uses a hierarchy of Pydantic configuration classes:

```
SentimatrixConfig (main)
├── LLMConfig
│   ├── RetryConfig
│   └── RateLimitConfig
├── ScraperConfig
│   ├── ProxyConfig
│   ├── RetryConfig
│   └── RateLimitConfig
├── ModelConfig
├── CacheConfig
├── LogConfig
├── OutputConfig
└── FallbackConfig
```

### LLMConfig

Configure LLM providers for summarization and insights:

```python
class LLMConfig(BaseModel):
    provider: LLMProvider = "openai"      # LLM provider to use
    model: str = "gpt-4o-mini"            # Model name/identifier
    api_key: Optional[str] = None         # API key (can use env var)
    api_base: Optional[str] = None        # Custom API base URL
    organization: Optional[str] = None    # Organization ID
    timeout: int = 30                     # Request timeout (5-300 seconds)
    max_tokens: int = 1024                # Max tokens to generate (1-128000)
    temperature: float = 0.7              # Sampling temperature (0.0-2.0)
    top_p: float = 1.0                    # Top-p sampling (0.0-1.0)
    retry: RetryConfig                    # Retry configuration
    rate_limit: RateLimitConfig           # Rate limit configuration
```

**Supported Providers (21 total):**

| Category | Providers |
|----------|-----------|
| Core | openai, anthropic, gemini |
| Cloud Enterprise | azure_openai, bedrock |
| Fast Inference | groq, cerebras, fireworks, together |
| Router/Gateway | openrouter |
| Specialized | mistral, cohere, deepseek |
| Local Inference | ollama, lmstudio, vllm, llamacpp, textgen, exllamav2 |
| Legacy | huggingface |

```yaml
# YAML example
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  temperature: 0.7
  max_tokens: 4096
  timeout: 30
  retry:
    max_retries: 3
    initial_delay: 1.0
  rate_limit:
    requests_per_second: 1.0
    concurrent_requests: 5
```

### ScraperConfig

Configure web scraping behavior:

```python
class ScraperConfig(BaseModel):
    provider: ScraperProvider = "playwright"   # Scraper provider
    headless: bool = True                      # Run browser headless
    timeout: int = 30                          # Page load timeout (5-120s)
    wait_for_selector: Optional[str] = None    # CSS selector to wait for
    user_agent: Optional[str] = None           # Custom user agent
    viewport_width: int = 1920                 # Browser width (320-3840)
    viewport_height: int = 1080                # Browser height (240-2160)
    proxy: ProxyConfig                         # Proxy configuration
    rate_limit: RateLimitConfig                # Rate limit configuration
    retry: RetryConfig                         # Retry configuration
    screenshots: bool = False                  # Capture screenshots
    screenshot_dir: Optional[str] = None       # Screenshot directory
```

**Supported Scraper Providers:**

| Type | Providers |
|------|-----------|
| Browser | playwright, selenium |
| HTTP | httpx, requests |
| Commercial API | scraperapi, brightdata, oxylabs, apify, zyte, firecrawl |

```yaml
# YAML example
scrapers:
  provider: playwright
  headless: true
  timeout: 30
  viewport_width: 1920
  viewport_height: 1080
  proxy:
    enabled: false
  rate_limit:
    requests_per_second: 2.0
    concurrent_requests: 5
  retry:
    max_retries: 3
    exponential_base: 2.0
```

### ModelConfig

Configure ML models for analysis:

```python
class ModelConfig(BaseModel):
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotion_model: str = "SamLowe/roberta-base-go_emotions"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    batch_size: int = 32                  # Batch size (1-512)
    max_length: int = 512                 # Max sequence length (32-4096)
    use_quantization: bool = False        # Enable model quantization
    cache_models: bool = True             # Cache loaded models
```

```yaml
# YAML example
models:
  sentiment_model: cardiffnlp/twitter-roberta-base-sentiment-latest
  emotion_model: SamLowe/roberta-base-go_emotions
  device: auto
  batch_size: 32
  max_length: 512
  use_quantization: false
  cache_models: true
```

### CacheConfig

Configure caching for performance:

```python
class CacheConfig(BaseModel):
    enabled: bool = True                  # Enable caching
    backend: CacheBackend = "memory"      # memory, redis, sqlite
    ttl: int = 3600                       # TTL in seconds (0-86400)
    max_size: int = 1000                  # Max cache entries (10-100000)
    namespace: str = "sentimatrix"        # Cache key namespace
    redis_url: Optional[str] = None       # Required for redis backend
    sqlite_path: Optional[str] = None     # Required for sqlite backend
    compression: bool = False             # Enable value compression
```

```yaml
# YAML example
cache:
  enabled: true
  backend: memory
  ttl: 3600
  max_size: 1000
  namespace: sentimatrix
  compression: false

# Redis backend
cache:
  enabled: true
  backend: redis
  redis_url: redis://localhost:6379
  ttl: 3600

# SQLite backend
cache:
  enabled: true
  backend: sqlite
  sqlite_path: .cache/sentimatrix.db
```

### LogConfig

Configure logging output:

```python
class LogConfig(BaseModel):
    level: LogLevel = "INFO"              # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: Literal["json", "text"] = "json"
    file_path: Optional[str] = None       # Log file path
    max_file_size_mb: int = 10            # Max file size (1-100 MB)
    backup_count: int = 5                 # Backup files (0-20)
    include_timestamp: bool = True
    include_caller: bool = True
    console_output: bool = True
    colorize: bool = True
```

```yaml
# YAML example
logging:
  level: INFO
  format: json
  file_path: logs/sentimatrix.log
  max_file_size_mb: 10
  backup_count: 5
  console_output: true
  colorize: true
```

### RetryConfig

Configure retry behavior (used by LLM and Scraper):

```python
class RetryConfig(BaseModel):
    max_retries: int = 3                  # Max attempts (0-10)
    initial_delay: float = 1.0            # Initial delay (0.1-60s)
    max_delay: float = 60.0               # Max delay (1-300s)
    exponential_base: float = 2.0         # Backoff base (1-5)
    jitter: bool = True                   # Add random jitter
```

### RateLimitConfig

Configure rate limiting (used by LLM and Scraper):

```python
class RateLimitConfig(BaseModel):
    requests_per_second: float = 1.0      # RPS (0.1-100)
    requests_per_minute: int = 60         # RPM (1-6000)
    concurrent_requests: int = 5          # Max concurrent (1-100)
    backoff_factor: float = 2.0           # Backoff multiplier (1-10)
```

### ProxyConfig

Configure proxy settings for scraping:

```python
class ProxyConfig(BaseModel):
    enabled: bool = False
    provider: Optional[str] = None        # brightdata, oxylabs, custom
    url: Optional[str] = None             # Proxy URL
    username: Optional[str] = None
    password: Optional[str] = None
    rotation: bool = True                 # Enable rotation
    country: Optional[str] = None         # Target country code
```

### FallbackConfig

Configure provider fallback chain:

```python
class FallbackConfig(BaseModel):
    enabled: bool = True
    providers: List[LLMProvider] = ["openai", "anthropic", "groq"]
    max_attempts: int = 3                 # Max attempts (1-10)
```

### OutputConfig

Configure output handling:

```python
class OutputConfig(BaseModel):
    default_format: Literal["json", "csv", "xlsx"] = "json"
    include_metadata: bool = True
    include_raw_data: bool = False
    pretty_print: bool = True
    datetime_format: str = "%Y-%m-%dT%H:%M:%SZ"
```

## Full Configuration Example

```yaml title="sentimatrix.yaml"
# LLM Provider Configuration
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  temperature: 0.7
  max_tokens: 4096
  timeout: 30
  retry:
    max_retries: 3
    initial_delay: 1.0
    max_delay: 60.0
    exponential_base: 2.0
    jitter: true
  rate_limit:
    requests_per_second: 1.0
    requests_per_minute: 60
    concurrent_requests: 5

# Provider Fallback Chain
fallback:
  enabled: true
  providers:
    - openai
    - anthropic
    - groq
  max_attempts: 3

# Web Scraping Configuration
scrapers:
  provider: playwright
  headless: true
  timeout: 30
  viewport_width: 1920
  viewport_height: 1080
  proxy:
    enabled: false
    rotation: true
  rate_limit:
    requests_per_second: 2.0
    concurrent_requests: 5
  retry:
    max_retries: 3
    initial_delay: 1.0
    exponential_base: 2.0

# ML Model Configuration
models:
  sentiment_model: cardiffnlp/twitter-roberta-base-sentiment-latest
  emotion_model: SamLowe/roberta-base-go_emotions
  device: auto
  batch_size: 32
  max_length: 512
  use_quantization: false
  cache_models: true

# Caching Configuration
cache:
  enabled: true
  backend: memory
  ttl: 3600
  max_size: 1000
  namespace: sentimatrix
  compression: false

# Logging Configuration
logging:
  level: INFO
  format: json
  console_output: true
  colorize: true

# Output Configuration
output:
  default_format: json
  include_metadata: true
  pretty_print: true

# Global Settings
debug: false
dry_run: false
```

## Loading Configuration

### From File

Load from YAML or JSON files:

```python
from sentimatrix.config import SentimatrixConfig

# Load from YAML
config = SentimatrixConfig.from_file("config.yaml")

# Load from JSON
config = SentimatrixConfig.from_file("config.json")

# With runtime overrides
config = SentimatrixConfig.from_file(
    "config.yaml",
    debug=True,
    llm={"model": "gpt-4o"}
)
```

### From Environment Variables

Environment variables use `SENTIMATRIX_` prefix with double underscore for nesting:

```python
from sentimatrix.config import SentimatrixConfig

# Automatically reads SENTIMATRIX_* environment variables
config = SentimatrixConfig.from_env()

# With overrides
config = SentimatrixConfig.from_env(debug=True)
```

### Combining Methods

Configuration is merged in order of precedence:

1. Runtime overrides (highest)
2. Environment variables
3. File configuration (lowest)

```python
from sentimatrix.config import SentimatrixConfig, LLMConfig

# File provides base, env vars override, runtime overrides take precedence
config = SentimatrixConfig.from_file("config.yaml", debug=True)

# Create modified copy
new_config = config.with_overrides(
    llm={"provider": "anthropic", "model": "claude-3-5-sonnet"}
)
```

### Saving Configuration

```python
config = SentimatrixConfig(
    llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
)

# Save to YAML
config.save("my-config.yaml")

# Save to JSON
config.save("my-config.json")

# Convert to dict/YAML string
config_dict = config.to_dict()
yaml_string = config.to_yaml()
```

## Environment Variables

All configuration can be set via environment variables with `SENTIMATRIX_` prefix. Nested values use double underscore (`__`) as delimiter:

```bash
# LLM Configuration
export SENTIMATRIX_LLM__PROVIDER="groq"
export SENTIMATRIX_LLM__MODEL="llama-3.3-70b-versatile"
export SENTIMATRIX_LLM__TEMPERATURE="0.7"
export SENTIMATRIX_LLM__MAX_TOKENS="4096"
export SENTIMATRIX_LLM__TIMEOUT="30"

# API Keys (standard provider pattern)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."

# Or via config with env: prefix
export MY_CUSTOM_KEY="sk-..."
# Then in config: api_key: "env:MY_CUSTOM_KEY"

# Scraper Configuration
export SENTIMATRIX_SCRAPERS__PROVIDER="playwright"
export SENTIMATRIX_SCRAPERS__HEADLESS="true"
export SENTIMATRIX_SCRAPERS__TIMEOUT="30"

# Rate Limiting
export SENTIMATRIX_SCRAPERS__RATE_LIMIT__REQUESTS_PER_SECOND="2.0"
export SENTIMATRIX_SCRAPERS__RATE_LIMIT__CONCURRENT_REQUESTS="5"

# Model Configuration
export SENTIMATRIX_MODELS__DEVICE="cuda"
export SENTIMATRIX_MODELS__BATCH_SIZE="64"

# Cache Configuration
export SENTIMATRIX_CACHE__ENABLED="true"
export SENTIMATRIX_CACHE__BACKEND="redis"
export SENTIMATRIX_CACHE__REDIS_URL="redis://localhost:6379"
export SENTIMATRIX_CACHE__TTL="3600"

# Logging
export SENTIMATRIX_LOGGING__LEVEL="DEBUG"
export SENTIMATRIX_LOGGING__FORMAT="json"

# Global
export SENTIMATRIX_DEBUG="true"
export SENTIMATRIX_DRY_RUN="false"
```

## Configuration Validation

Sentimatrix validates all configuration using Pydantic V2:

```python
from sentimatrix.config import SentimatrixConfig, LLMConfig, CacheConfig
from pydantic import ValidationError

# Invalid provider
try:
    config = SentimatrixConfig(
        llm=LLMConfig(provider="invalid_provider")
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# Invalid range
try:
    config = SentimatrixConfig(
        llm=LLMConfig(temperature=3.0)  # Max is 2.0
    )
except ValidationError as e:
    print(f"Temperature out of range: {e}")

# Missing required backend config
try:
    config = SentimatrixConfig(
        cache=CacheConfig(backend="redis")  # Missing redis_url
    )
except ValidationError as e:
    print(f"Redis URL required: {e}")
```

### Validation Rules

| Config | Field | Constraints |
|--------|-------|-------------|
| LLMConfig | timeout | 5-300 seconds |
| LLMConfig | max_tokens | 1-128000 |
| LLMConfig | temperature | 0.0-2.0 |
| LLMConfig | top_p | 0.0-1.0 |
| ScraperConfig | timeout | 5-120 seconds |
| ScraperConfig | viewport_width | 320-3840 |
| ModelConfig | batch_size | 1-512 |
| ModelConfig | max_length | 32-4096 |
| CacheConfig | ttl | 0-86400 seconds |
| CacheConfig | max_size | 10-100000 |
| RetryConfig | max_retries | 0-10 |
| RateLimitConfig | requests_per_second | 0.1-100 |

## Main Configuration Class

The `SentimatrixConfig` class is the main entry point:

```python
class SentimatrixConfig(BaseSettings):
    """
    Main configuration class for Sentimatrix.

    Supports loading from:
    - YAML/JSON files
    - Environment variables (prefixed with SENTIMATRIX_)
    - Runtime overrides
    """

    # Sub-configurations
    llm: LLMConfig                    # LLM provider settings
    scrapers: ScraperConfig           # Web scraping settings
    models: ModelConfig               # ML model settings
    cache: CacheConfig                # Cache backend settings
    logging: LogConfig                # Logging settings
    output: OutputConfig              # Output format settings
    fallback: FallbackConfig          # Provider fallback chain

    # Global settings
    debug: bool = False               # Enable debug mode
    dry_run: bool = False             # No API calls mode

    # Class methods
    @classmethod
    def from_file(cls, path, **overrides) -> "SentimatrixConfig": ...

    @classmethod
    def from_env(cls, **overrides) -> "SentimatrixConfig": ...

    def to_dict(self) -> dict: ...
    def to_yaml(self) -> str: ...
    def save(self, path) -> None: ...
    def with_overrides(self, **overrides) -> "SentimatrixConfig": ...
```

## Convenience Function

```python
from sentimatrix.config import get_config

# From file
config = get_config("config.yaml")

# From environment
config = get_config()

# With overrides
config = get_config("config.yaml", debug=True)
```

## Related

- [LLM Providers](../providers/index.md)
- [Scrapers](../scrapers/index.md)
- [API Reference](../api/index.md)
