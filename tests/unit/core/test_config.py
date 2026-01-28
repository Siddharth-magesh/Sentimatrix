"""
Unit Tests for Configuration Module

Tests the Pydantic configuration system including:
- Default configuration values
- YAML file loading
- Environment variable support
- Configuration validation
- Configuration serialization
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from sentimatrix.core.config import (
    CacheBackend,
    CacheConfig,
    FallbackConfig,
    LLMConfig,
    LLMProvider,
    LogConfig,
    LogLevel,
    ModelConfig,
    OutputConfig,
    ProxyConfig,
    RateLimitConfig,
    RetryConfig,
    ScraperConfig,
    ScraperProvider,
    SentimatrixConfig,
    get_config,
)
from sentimatrix.core.exceptions import ConfigurationError


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_validation_max_retries(self):
        """Test validation of max_retries bounds."""
        with pytest.raises(ValueError):
            RetryConfig(max_retries=-1)
        with pytest.raises(ValueError):
            RetryConfig(max_retries=100)

    def test_frozen_immutability(self):
        """Test that config is immutable (frozen)."""
        config = RetryConfig()
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            config.max_retries = 10


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_values(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        assert config.requests_per_second == 1.0
        assert config.requests_per_minute == 60
        assert config.concurrent_requests == 5
        assert config.backoff_factor == 2.0

    def test_validation_bounds(self):
        """Test validation of rate limit bounds."""
        with pytest.raises(ValueError):
            RateLimitConfig(requests_per_second=0)
        with pytest.raises(ValueError):
            RateLimitConfig(concurrent_requests=0)


class TestProxyConfig:
    """Tests for ProxyConfig."""

    def test_default_disabled(self):
        """Test proxy is disabled by default."""
        config = ProxyConfig()
        assert config.enabled is False
        assert config.url is None
        assert config.provider is None

    def test_enabled_requires_url_or_provider(self):
        """Test validation that enabled proxy needs url or provider."""
        with pytest.raises(ValueError, match="Either 'url' or 'provider'"):
            ProxyConfig(enabled=True)

    def test_enabled_with_url(self):
        """Test enabled proxy with URL."""
        config = ProxyConfig(
            enabled=True,
            url="http://proxy.example.com:8080",
        )
        assert config.enabled is True
        assert config.url == "http://proxy.example.com:8080"

    def test_enabled_with_provider(self):
        """Test enabled proxy with provider."""
        config = ProxyConfig(
            enabled=True,
            provider="brightdata",
        )
        assert config.enabled is True
        assert config.provider == "brightdata"


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4o-mini"
        assert config.timeout == 30
        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 1.0

    def test_custom_provider(self):
        """Test custom provider configuration."""
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-3-sonnet")
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model == "claude-3-sonnet"

    def test_env_api_key_resolution(self, monkeypatch):
        """Test API key resolution from environment variable."""
        monkeypatch.setenv("MY_API_KEY", "secret-key-123")
        config = LLMConfig(api_key="env:MY_API_KEY")
        assert config.api_key == "secret-key-123"

    def test_env_api_key_missing(self, monkeypatch):
        """Test API key when env var is missing."""
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        config = LLMConfig(api_key="env:NONEXISTENT_KEY")
        assert config.api_key is None

    def test_direct_api_key(self):
        """Test direct API key configuration."""
        config = LLMConfig(api_key="direct-key")
        assert config.api_key == "direct-key"

    def test_temperature_bounds(self):
        """Test temperature validation."""
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            LLMConfig(temperature=2.5)


class TestScraperConfig:
    """Tests for ScraperConfig."""

    def test_default_values(self):
        """Test default scraper configuration."""
        config = ScraperConfig()
        assert config.provider == ScraperProvider.PLAYWRIGHT
        assert config.headless is True
        assert config.timeout == 30
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080

    def test_selenium_provider(self):
        """Test Selenium provider configuration."""
        config = ScraperConfig(provider=ScraperProvider.SELENIUM)
        assert config.provider == ScraperProvider.SELENIUM

    def test_viewport_validation(self):
        """Test viewport dimension validation."""
        with pytest.raises(ValueError):
            ScraperConfig(viewport_width=100)  # Below minimum
        with pytest.raises(ValueError):
            ScraperConfig(viewport_height=5000)  # Above maximum


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert "twitter-roberta" in config.sentiment_model
        assert "go_emotions" in config.emotion_model
        assert config.device == "auto"
        assert config.batch_size == 32

    def test_device_options(self):
        """Test device configuration options."""
        for device in ["auto", "cpu", "cuda", "mps"]:
            config = ModelConfig(device=device)
            assert config.device == device


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_values(self):
        """Test default cache configuration."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.backend == CacheBackend.MEMORY
        assert config.ttl == 3600
        assert config.max_size == 1000
        assert config.namespace == "sentimatrix"

    def test_redis_requires_url(self):
        """Test Redis backend requires redis_url."""
        with pytest.raises(ValueError, match="redis_url is required"):
            CacheConfig(backend=CacheBackend.REDIS)

    def test_redis_with_url(self):
        """Test Redis backend with URL."""
        config = CacheConfig(
            backend=CacheBackend.REDIS,
            redis_url="redis://localhost:6379",
        )
        assert config.backend == CacheBackend.REDIS
        assert config.redis_url == "redis://localhost:6379"

    def test_sqlite_requires_path(self):
        """Test SQLite backend requires sqlite_path."""
        with pytest.raises(ValueError, match="sqlite_path is required"):
            CacheConfig(backend=CacheBackend.SQLITE)


class TestLogConfig:
    """Tests for LogConfig."""

    def test_default_values(self):
        """Test default log configuration."""
        config = LogConfig()
        assert config.level == LogLevel.INFO
        assert config.format == "json"
        assert config.console_output is True
        assert config.colorize is True

    def test_log_levels(self):
        """Test all log level options."""
        for level in LogLevel:
            config = LogConfig(level=level)
            assert config.level == level


class TestSentimatrixConfig:
    """Tests for main SentimatrixConfig."""

    def test_default_configuration(self):
        """Test default configuration has all required fields."""
        config = SentimatrixConfig()
        assert config.llm is not None
        assert config.scrapers is not None
        assert config.models is not None
        assert config.cache is not None
        assert config.logging is not None
        assert config.output is not None
        assert config.fallback is not None
        assert config.debug is False

    def test_from_file_yaml(self, sample_config_yaml: Path):
        """Test loading configuration from YAML file."""
        config = SentimatrixConfig.from_file(sample_config_yaml)
        assert config.llm.provider == LLMProvider.OPENAI
        assert config.llm.model == "gpt-4o-mini"
        assert config.llm.temperature == 0.5
        assert config.cache.ttl == 7200
        assert config.cache.max_size == 500

    def test_from_file_not_found(self, temp_dir: Path):
        """Test error when config file not found."""
        with pytest.raises(ConfigurationError, match="not found"):
            SentimatrixConfig.from_file(temp_dir / "nonexistent.yaml")

    def test_from_file_json(self, temp_dir: Path):
        """Test loading configuration from JSON file."""
        config_path = temp_dir / "config.json"
        config_data = {
            "llm": {"provider": "groq", "model": "llama3-70b"},
            "debug": True,
        }
        config_path.write_text(json.dumps(config_data))

        config = SentimatrixConfig.from_file(config_path)
        assert config.llm.provider == LLMProvider.GROQ
        assert config.llm.model == "llama3-70b"
        assert config.debug is True

    def test_from_file_unsupported_format(self, temp_dir: Path):
        """Test error with unsupported file format."""
        config_path = temp_dir / "config.txt"
        config_path.write_text("some content")
        with pytest.raises(ConfigurationError, match="Unsupported config file format"):
            SentimatrixConfig.from_file(config_path)

    def test_from_env(self, env_with_config):
        """Test loading configuration from environment variables."""
        config = SentimatrixConfig.from_env()
        assert config.llm.provider == LLMProvider.ANTHROPIC
        assert config.llm.model == "claude-3-sonnet"
        assert config.debug is True

    def test_to_dict(self, default_config: SentimatrixConfig):
        """Test converting configuration to dictionary."""
        config_dict = default_config.to_dict()
        assert isinstance(config_dict, dict)
        assert "llm" in config_dict
        assert "scrapers" in config_dict
        assert "cache" in config_dict

    def test_to_yaml(self, default_config: SentimatrixConfig):
        """Test converting configuration to YAML string."""
        yaml_str = default_config.to_yaml()
        assert isinstance(yaml_str, str)
        # Should be valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert "llm" in parsed

    def test_save_yaml(self, default_config: SentimatrixConfig, temp_dir: Path):
        """Test saving configuration to YAML file."""
        output_path = temp_dir / "output.yaml"
        default_config.save(output_path)
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            loaded = yaml.safe_load(f)
        assert "llm" in loaded

    def test_save_json(self, default_config: SentimatrixConfig, temp_dir: Path):
        """Test saving configuration to JSON file."""
        output_path = temp_dir / "output.json"
        default_config.save(output_path)
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            loaded = json.load(f)
        assert "llm" in loaded

    def test_with_overrides(self, default_config: SentimatrixConfig):
        """Test creating new config with overrides."""
        new_config = default_config.with_overrides(
            debug=True,
            llm={"temperature": 0.9},
        )
        assert new_config.debug is True
        assert new_config.llm.temperature == 0.9
        # Original should be unchanged
        assert default_config.debug is False

    def test_override_in_from_file(self, sample_config_yaml: Path):
        """Test overrides when loading from file."""
        config = SentimatrixConfig.from_file(
            sample_config_yaml,
            debug=True,
        )
        assert config.debug is True


class TestGetConfig:
    """Tests for get_config convenience function."""

    def test_get_config_default(self, clean_env):
        """Test getting default configuration."""
        config = get_config()
        assert isinstance(config, SentimatrixConfig)

    def test_get_config_from_file(self, sample_config_yaml: Path):
        """Test getting configuration from file."""
        config = get_config(sample_config_yaml)
        assert config.llm.model == "gpt-4o-mini"

    def test_get_config_with_overrides(self, clean_env):
        """Test getting configuration with overrides."""
        config = get_config(debug=True)
        assert config.debug is True
