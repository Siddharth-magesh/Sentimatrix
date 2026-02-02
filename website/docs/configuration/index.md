---
title: Configuration
description: Configure Sentimatrix for your needs
---

# Configuration

Sentimatrix can be configured using YAML files, environment variables, or Python objects.

## Configuration Methods

<div class="grid">

<div class="card">
<h3>:material-file-cog: YAML Files</h3>
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
    # Auto-loads sentimatrix.yaml from current directory
    async with Sentimatrix() as sm:
        summary = await sm.summarize_reviews(reviews)
    ```

=== "Environment"

    ```bash
    export GROQ_API_KEY="gsk_..."
    export SENTIMATRIX_LLM_PROVIDER="groq"
    export SENTIMATRIX_LLM_MODEL="llama-3.3-70b-versatile"
    ```

## Configuration Sections

### LLM Configuration

Configure LLM providers for summarization and insights:

```yaml
llm:
  provider: groq              # Provider name
  model: llama-3.3-70b        # Model name
  temperature: 0.7            # Creativity (0-2)
  max_tokens: 4096            # Max response length
  timeout: 30                 # Request timeout (seconds)

  # Fallback providers
  fallback:
    - provider: openai
      model: gpt-4o-mini
    - provider: ollama
      model: llama3.2
```

### Scraper Configuration

Configure web scraping behavior:

```yaml
scraper:
  # Rate limiting
  rate_limit:
    requests_per_second: 2
    burst_size: 5
    cooldown_on_429: 60

  # Retry settings
  retry:
    max_retries: 3
    backoff_factor: 2.0
    retryable_errors:
      - timeout
      - connection_error
      - rate_limit

  # Browser settings (Playwright)
  browser:
    headless: true
    timeout: 30000
    stealth_mode: true

  # Commercial API (optional)
  api:
    provider: scraperapi
    # api_key: loaded from environment
```

### Model Configuration

Configure ML models for analysis:

```yaml
model:
  # Sentiment analysis
  sentiment:
    model: cardiffnlp/twitter-roberta-base-sentiment-latest
    device: auto  # auto, cpu, cuda, mps

  # Emotion detection
  emotion:
    model: SamLowe/roberta-base-go_emotions
    taxonomy: goemotion  # ekman, goemotion, plutchik

  # Processing
  batch_size: 32
  max_length: 512
```

### Cache Configuration

Configure caching for performance:

```yaml
cache:
  enabled: true
  backend: memory  # memory, redis, sqlite

  # Memory cache settings
  memory:
    max_size: 10000
    ttl: 3600

  # Redis settings
  redis:
    url: redis://localhost:6379
    prefix: sentimatrix:

  # SQLite settings
  sqlite:
    path: .cache/sentimatrix.db
```

### Logging Configuration

Configure logging output:

```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  format: structured  # structured, simple
  output: console  # console, file, both

  file:
    path: logs/sentimatrix.log
    rotation: 10MB
    retention: 7
```

## Full Configuration Example

```yaml title="sentimatrix.yaml"
# LLM Provider
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  temperature: 0.7
  max_tokens: 4096
  timeout: 30
  fallback:
    - provider: openai
      model: gpt-4o-mini

# Web Scraping
scraper:
  rate_limit:
    requests_per_second: 2
    burst_size: 5
  retry:
    max_retries: 3
    backoff_factor: 2.0
  browser:
    headless: true
    timeout: 30000

# ML Models
model:
  sentiment:
    model: cardiffnlp/twitter-roberta-base-sentiment-latest
    device: auto
  batch_size: 32

# Caching
cache:
  enabled: true
  backend: memory
  memory:
    max_size: 10000
    ttl: 3600

# Logging
logging:
  level: INFO
  format: structured
```

## Loading Configuration

### Auto-Discovery

Sentimatrix automatically loads configuration from:

1. `sentimatrix.yaml` in current directory
2. `config/sentimatrix.yaml`
3. `~/.sentimatrix/config.yaml`

### Explicit Path

```python
from sentimatrix import Sentimatrix

async with Sentimatrix(config_path="/path/to/config.yaml") as sm:
    result = await sm.analyze("Hello")
```

### Combining Methods

Configuration is merged in order of precedence:

1. Python objects (highest)
2. Environment variables
3. YAML files (lowest)

```python
from sentimatrix.config import SentimatrixConfig, LLMConfig

# Override YAML with Python
config = SentimatrixConfig(
    llm=LLMConfig(provider="openai")  # Overrides YAML
)

async with Sentimatrix(config) as sm:
    # Uses OpenAI even if YAML says groq
    pass
```

## Environment Variables

All configuration can be set via environment variables:

```bash
# LLM Configuration
export SENTIMATRIX_LLM_PROVIDER="groq"
export SENTIMATRIX_LLM_MODEL="llama-3.3-70b-versatile"
export SENTIMATRIX_LLM_TEMPERATURE="0.7"

# API Keys (common pattern)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."

# Scraper Configuration
export SENTIMATRIX_SCRAPER_RATE_LIMIT_RPS="2"
export SENTIMATRIX_SCRAPER_BROWSER_HEADLESS="true"

# Cache Configuration
export SENTIMATRIX_CACHE_ENABLED="true"
export SENTIMATRIX_CACHE_BACKEND="redis"
export SENTIMATRIX_CACHE_REDIS_URL="redis://localhost:6379"

# Logging
export SENTIMATRIX_LOGGING_LEVEL="DEBUG"
```

## Configuration Validation

Sentimatrix validates configuration on load:

```python
from sentimatrix.config import SentimatrixConfig, LLMConfig

try:
    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="invalid_provider"
        )
    )
except ValueError as e:
    print(f"Invalid config: {e}")
```

## Section Documentation

- [YAML Configuration](yaml.md) - Full YAML reference
- [Environment Variables](environment.md) - Environment variable patterns
- [Python Objects](python.md) - Programmatic configuration
- [Caching](caching.md) - Cache backends and settings
- [Logging](logging.md) - Logging configuration
