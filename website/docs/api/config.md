---
title: Configuration Classes
description: Configuration API reference
---

# Configuration Classes

Pydantic V2 configuration classes for Sentimatrix.

## SentimatrixConfig

Main configuration class.

```python
class SentimatrixConfig(BaseSettings):
    llm: LLMConfig
    scrapers: ScraperConfig
    models: ModelConfig
    cache: CacheConfig
    logging: LogConfig
    output: OutputConfig
    fallback: FallbackConfig
    debug: bool = False
    dry_run: bool = False

    @classmethod
    def from_file(cls, path, **overrides) -> "SentimatrixConfig"

    @classmethod
    def from_env(cls, **overrides) -> "SentimatrixConfig"

    def to_dict(self) -> dict
    def to_yaml(self) -> str
    def save(self, path) -> None
    def with_overrides(self, **overrides) -> "SentimatrixConfig"
```

## LLMConfig

LLM provider configuration.

```python
class LLMConfig(BaseModel):
    provider: LLMProvider = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7      # 0.0-2.0
    max_tokens: int = 1024        # 1-128000
    timeout: int = 30             # 5-300
    retry: RetryConfig
    rate_limit: RateLimitConfig
```

## ScraperConfig

Web scraping configuration.

```python
class ScraperConfig(BaseModel):
    provider: ScraperProvider = "playwright"
    headless: bool = True
    timeout: int = 30             # 5-120
    viewport_width: int = 1920
    viewport_height: int = 1080
    proxy: ProxyConfig
    retry: RetryConfig
    rate_limit: RateLimitConfig
```

## ModelConfig

ML model configuration.

```python
class ModelConfig(BaseModel):
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotion_model: str = "SamLowe/roberta-base-go_emotions"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    batch_size: int = 32          # 1-512
    max_length: int = 512         # 32-4096
```

## CacheConfig

Cache backend configuration.

```python
class CacheConfig(BaseModel):
    enabled: bool = True
    backend: CacheBackend = "memory"
    ttl: int = 3600               # 0-86400
    max_size: int = 1000
    redis_url: Optional[str] = None
    sqlite_path: Optional[str] = None
```

## Related

- [API Overview](index.md)
- [Configuration Guide](../configuration/index.md)
