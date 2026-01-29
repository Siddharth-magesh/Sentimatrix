"""
Sentimatrix Configuration Module

Provides centralized configuration management using Pydantic V2.
Supports YAML/JSON file loading, environment variables, and runtime overrides.

Example:
    >>> config = SentimatrixConfig.from_file("config.yaml")
    >>> config = SentimatrixConfig.from_env()
    >>> config.llm.provider
    'openai'
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheBackend(str, Enum):
    """Supported cache backends."""

    MEMORY = "memory"
    REDIS = "redis"
    SQLITE = "sqlite"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    # Core Providers
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

    # Cloud Enterprise
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"

    # Fast Inference
    GROQ = "groq"
    CEREBRAS = "cerebras"
    FIREWORKS = "fireworks"
    TOGETHER = "together"

    # Router/Gateway
    OPENROUTER = "openrouter"

    # Specialized
    MISTRAL = "mistral"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"

    # Local Inference
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    TEXTGEN = "textgen"
    EXLLAMAV2 = "exllamav2"

    # Legacy
    HUGGINGFACE = "huggingface"


class ScraperProvider(str, Enum):
    """Supported scraper providers."""

    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    HTTPX = "httpx"
    REQUESTS = "requests"
    SCRAPERAPI = "scraperapi"
    BRIGHTDATA = "brightdata"
    OXYLABS = "oxylabs"
    APIFY = "apify"
    ZYTE = "zyte"
    FIRECRAWL = "firecrawl"


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retry attempts")
    initial_delay: float = Field(
        default=1.0, ge=0.1, le=60.0, description="Initial delay between retries in seconds"
    )
    max_delay: float = Field(
        default=60.0, ge=1.0, le=300.0, description="Maximum delay between retries in seconds"
    )
    exponential_base: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Base for exponential backoff"
    )
    jitter: bool = Field(default=True, description="Add random jitter to delays")


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    model_config = ConfigDict(frozen=True)

    requests_per_second: float = Field(
        default=1.0, ge=0.1, le=100.0, description="Maximum requests per second"
    )
    requests_per_minute: int = Field(
        default=60, ge=1, le=6000, description="Maximum requests per minute"
    )
    concurrent_requests: int = Field(
        default=5, ge=1, le=100, description="Maximum concurrent requests"
    )
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Backoff multiplier on rate limit hit"
    )


class ProxyConfig(BaseModel):
    """Configuration for proxy settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False, description="Enable proxy usage")
    provider: Optional[str] = Field(
        default=None, description="Proxy provider (brightdata, oxylabs, custom)"
    )
    url: Optional[str] = Field(default=None, description="Proxy URL")
    username: Optional[str] = Field(default=None, description="Proxy username")
    password: Optional[str] = Field(default=None, description="Proxy password")
    rotation: bool = Field(default=True, description="Enable proxy rotation")
    country: Optional[str] = Field(default=None, description="Target country code")

    @model_validator(mode="after")
    def validate_proxy_config(self) -> "ProxyConfig":
        """Validate proxy configuration consistency."""
        if self.enabled and not self.url and not self.provider:
            raise ValueError("Either 'url' or 'provider' must be specified when proxy is enabled")
        return self


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""

    model_config = ConfigDict(frozen=True)

    provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider to use")
    model: str = Field(default="gpt-4o-mini", description="Model name/identifier")
    api_key: Optional[str] = Field(default=None, description="API key (can use env var)")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    organization: Optional[str] = Field(default=None, description="Organization ID")
    timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")
    max_tokens: int = Field(default=1024, ge=1, le=128000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limit configuration"
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key_from_env(cls, v: Optional[str]) -> Optional[str]:
        """Resolve API key from environment variable if prefixed with 'env:'."""
        if v and v.startswith("env:"):
            env_var = v[4:]
            return os.environ.get(env_var)
        return v


class ScraperConfig(BaseModel):
    """Configuration for web scraping."""

    model_config = ConfigDict(frozen=True)

    provider: ScraperProvider = Field(
        default=ScraperProvider.PLAYWRIGHT, description="Scraper provider to use"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")
    timeout: int = Field(default=30, ge=5, le=120, description="Page load timeout in seconds")
    wait_for_selector: Optional[str] = Field(
        default=None, description="CSS selector to wait for before scraping"
    )
    user_agent: Optional[str] = Field(default=None, description="Custom user agent string")
    viewport_width: int = Field(default=1920, ge=320, le=3840, description="Browser viewport width")
    viewport_height: int = Field(
        default=1080, ge=240, le=2160, description="Browser viewport height"
    )
    proxy: ProxyConfig = Field(default_factory=ProxyConfig, description="Proxy configuration")
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="Rate limit configuration"
    )
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")
    screenshots: bool = Field(default=False, description="Capture screenshots during scraping")
    screenshot_dir: Optional[str] = Field(
        default=None, description="Directory for screenshots"
    )


class ModelConfig(BaseModel):
    """Configuration for ML models."""

    model_config = ConfigDict(frozen=True)

    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="HuggingFace model for sentiment analysis",
    )
    emotion_model: str = Field(
        default="SamLowe/roberta-base-go_emotions",
        description="HuggingFace model for emotion detection",
    )
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto", description="Device for model inference"
    )
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size for inference")
    max_length: int = Field(
        default=512, ge=32, le=4096, description="Maximum sequence length for tokenization"
    )
    use_quantization: bool = Field(default=False, description="Enable model quantization")
    cache_models: bool = Field(default=True, description="Cache loaded models in memory")


class CacheConfig(BaseModel):
    """Configuration for caching."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Enable caching")
    backend: CacheBackend = Field(default=CacheBackend.MEMORY, description="Cache backend")
    ttl: int = Field(default=3600, ge=0, le=86400, description="Default TTL in seconds (0=no expiry)")
    max_size: int = Field(default=1000, ge=10, le=100000, description="Maximum cache entries")
    namespace: str = Field(default="sentimatrix", description="Cache key namespace")
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    sqlite_path: Optional[str] = Field(default=None, description="SQLite database path")
    compression: bool = Field(default=False, description="Enable cache value compression")

    @model_validator(mode="after")
    def validate_backend_config(self) -> "CacheConfig":
        """Validate backend-specific configuration."""
        if self.backend == CacheBackend.REDIS and not self.redis_url:
            raise ValueError("redis_url is required when using Redis backend")
        if self.backend == CacheBackend.SQLITE and not self.sqlite_path:
            raise ValueError("sqlite_path is required when using SQLite backend")
        return self


class LogConfig(BaseModel):
    """Configuration for logging."""

    model_config = ConfigDict(frozen=True)

    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    format: Literal["json", "text"] = Field(default="json", description="Log format")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size_mb: int = Field(
        default=10, ge=1, le=100, description="Maximum log file size in MB"
    )
    backup_count: int = Field(default=5, ge=0, le=20, description="Number of backup log files")
    include_timestamp: bool = Field(default=True, description="Include timestamp in logs")
    include_caller: bool = Field(default=True, description="Include caller info in logs")
    console_output: bool = Field(default=True, description="Output logs to console")
    colorize: bool = Field(default=True, description="Colorize console output")


class OutputConfig(BaseModel):
    """Configuration for output handling."""

    model_config = ConfigDict(frozen=True)

    default_format: Literal["json", "csv", "xlsx"] = Field(
        default="json", description="Default export format"
    )
    include_metadata: bool = Field(default=True, description="Include metadata in exports")
    include_raw_data: bool = Field(default=False, description="Include raw data in exports")
    pretty_print: bool = Field(default=True, description="Pretty print JSON output")
    datetime_format: str = Field(
        default="%Y-%m-%dT%H:%M:%SZ", description="Datetime format string"
    )


class FallbackConfig(BaseModel):
    """Configuration for provider fallback chain."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Enable fallback chain")
    providers: List[LLMProvider] = Field(
        default_factory=lambda: [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GROQ],
        description="Ordered list of fallback providers",
    )
    max_attempts: int = Field(default=3, ge=1, le=10, description="Maximum fallback attempts")


class SentimatrixConfig(BaseSettings):
    """
    Main configuration class for Sentimatrix.

    Supports loading from:
    - YAML/JSON files
    - Environment variables (prefixed with SENTIMATRIX_)
    - Runtime overrides

    Example:
        >>> config = SentimatrixConfig.from_file("config.yaml")
        >>> config = SentimatrixConfig()  # Uses defaults + env vars
    """

    model_config = SettingsConfigDict(
        env_prefix="SENTIMATRIX_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM provider configuration")
    scrapers: ScraperConfig = Field(
        default_factory=ScraperConfig, description="Scraper configuration"
    )
    models: ModelConfig = Field(default_factory=ModelConfig, description="ML model configuration")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    logging: LogConfig = Field(default_factory=LogConfig, description="Logging configuration")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    fallback: FallbackConfig = Field(
        default_factory=FallbackConfig, description="Fallback configuration"
    )

    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")
    dry_run: bool = Field(default=False, description="Enable dry run mode (no API calls)")

    @classmethod
    def from_file(cls, path: Union[str, Path], **overrides: Any) -> "SentimatrixConfig":
        """
        Load configuration from a YAML or JSON file.

        Args:
            path: Path to configuration file
            **overrides: Additional configuration overrides

        Returns:
            SentimatrixConfig instance

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        path = Path(path)

        if not path.exists():
            from sentimatrix.core.exceptions import ConfigurationError

            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix in (".yaml", ".yml"):
                    data = yaml.safe_load(f) or {}
                elif path.suffix == ".json":
                    import json

                    data = json.load(f)
                else:
                    from sentimatrix.core.exceptions import ConfigurationError

                    raise ConfigurationError(
                        f"Unsupported config file format: {path.suffix}. "
                        "Use .yaml, .yml, or .json"
                    )
        except yaml.YAMLError as e:
            from sentimatrix.core.exceptions import ConfigurationError

            raise ConfigurationError(f"Failed to parse YAML configuration: {e}") from e
        except Exception as e:
            from sentimatrix.core.exceptions import ConfigurationError

            raise ConfigurationError(f"Failed to load configuration file: {e}") from e

        # Merge overrides
        data.update(overrides)

        return cls(**data)

    @classmethod
    def from_env(cls, **overrides: Any) -> "SentimatrixConfig":
        """
        Load configuration from environment variables.

        Environment variables should be prefixed with SENTIMATRIX_.
        Nested values use double underscore as delimiter.

        Example:
            SENTIMATRIX_LLM__PROVIDER=openai
            SENTIMATRIX_LLM__MODEL=gpt-4
            SENTIMATRIX_DEBUG=true

        Args:
            **overrides: Additional configuration overrides

        Returns:
            SentimatrixConfig instance
        """
        return cls(**overrides)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary with serializable values."""
        return self.model_dump(mode='json')

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a file.

        Args:
            path: Output file path (format determined by extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
            elif path.suffix == ".json":
                import json

                json.dump(self.to_dict(), f, indent=2)
            else:
                from sentimatrix.core.exceptions import ConfigurationError

                raise ConfigurationError(
                    f"Unsupported output format: {path.suffix}. Use .yaml, .yml, or .json"
                )

    def with_overrides(self, **overrides: Any) -> "SentimatrixConfig":
        """
        Create a new configuration with the specified overrides.

        Args:
            **overrides: Configuration overrides

        Returns:
            New SentimatrixConfig instance with overrides applied
        """
        data = self.to_dict()
        self._deep_merge(data, overrides)
        return SentimatrixConfig(**data)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep merge updates into base dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                SentimatrixConfig._deep_merge(base[key], value)
            else:
                base[key] = value


# Convenience function for quick access
def get_config(
    config_path: Optional[Union[str, Path]] = None, **overrides: Any
) -> SentimatrixConfig:
    """
    Get configuration instance.

    Loads from file if path provided, otherwise from environment variables.

    Args:
        config_path: Optional path to configuration file
        **overrides: Configuration overrides

    Returns:
        SentimatrixConfig instance
    """
    if config_path:
        return SentimatrixConfig.from_file(config_path, **overrides)
    return SentimatrixConfig.from_env(**overrides)
