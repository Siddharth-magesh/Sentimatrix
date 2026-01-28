"""
Pytest Configuration and Shared Fixtures

This module contains shared fixtures and configuration for all tests.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

# Add source directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "sentimatrix"))

from sentimatrix.core.config import (
    CacheConfig,
    LLMConfig,
    LogConfig,
    LogLevel,
    ModelConfig,
    RateLimitConfig,
    RetryConfig,
    ScraperConfig,
    SentimatrixConfig,
)
from sentimatrix.core.cache import CacheManager, MemoryCache
from sentimatrix.core.logger import LogManager, configure_logging


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Configuration fixtures
@pytest.fixture
def default_config() -> SentimatrixConfig:
    """Provide default Sentimatrix configuration."""
    return SentimatrixConfig()


@pytest.fixture
def llm_config() -> LLMConfig:
    """Provide default LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-api-key",
        temperature=0.7,
        max_tokens=1024,
    )


@pytest.fixture
def scraper_config() -> ScraperConfig:
    """Provide default scraper configuration."""
    return ScraperConfig(
        provider="playwright",
        headless=True,
        timeout=30,
    )


@pytest.fixture
def model_config() -> ModelConfig:
    """Provide default model configuration."""
    return ModelConfig(
        sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        emotion_model="SamLowe/roberta-base-go_emotions",
        device="cpu",
    )


@pytest.fixture
def cache_config() -> CacheConfig:
    """Provide default cache configuration."""
    return CacheConfig(
        enabled=True,
        backend="memory",
        ttl=3600,
        max_size=100,
    )


@pytest.fixture
def log_config() -> LogConfig:
    """Provide default log configuration."""
    return LogConfig(
        level=LogLevel.DEBUG,
        format="text",
        console_output=False,  # Disable console output during tests
    )


# Cache fixtures
@pytest.fixture
async def memory_cache() -> MemoryCache:
    """Provide a fresh memory cache instance."""
    cache = MemoryCache(max_size=100, default_ttl=60)
    yield cache


@pytest.fixture
async def cache_manager(cache_config: CacheConfig) -> CacheManager:
    """Provide an initialized cache manager."""
    manager = CacheManager(cache_config)
    await manager.initialize()
    yield manager
    await manager.close()


# Logging fixtures
@pytest.fixture
def configured_logger(log_config: LogConfig):
    """Provide a configured logger instance."""
    manager = LogManager()
    manager.configure(log_config)
    yield manager
    manager.shutdown()


# Temp file fixtures
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that's cleaned up after tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir: Path) -> Path:
    """Provide a temporary config file path."""
    return temp_dir / "config.yaml"


@pytest.fixture
def sample_config_yaml(temp_config_file: Path) -> Path:
    """Create a sample YAML config file."""
    config_content = """
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key
  temperature: 0.5
  max_tokens: 2048

scrapers:
  provider: playwright
  headless: true
  timeout: 60

cache:
  enabled: true
  backend: memory
  ttl: 7200
  max_size: 500

logging:
  level: INFO
  format: json
  console_output: true
"""
    temp_config_file.write_text(config_content)
    return temp_config_file


# Sample data fixtures
@pytest.fixture
def sample_texts() -> list:
    """Provide sample texts for testing."""
    return [
        "This product is absolutely amazing! Best purchase ever.",
        "Terrible experience. Would not recommend to anyone.",
        "It's okay, nothing special but gets the job done.",
        "The quality exceeded my expectations. Very happy!",
        "Worst product I've ever bought. Complete waste of money.",
    ]


@pytest.fixture
def sample_reviews() -> list:
    """Provide sample review data for testing."""
    return [
        {
            "id": "rev_001",
            "text": "Great product, highly recommend!",
            "rating": 5.0,
            "author": "John Doe",
            "platform": "amazon",
        },
        {
            "id": "rev_002",
            "text": "Disappointing quality, broke after a week.",
            "rating": 1.0,
            "author": "Jane Smith",
            "platform": "amazon",
        },
        {
            "id": "rev_003",
            "text": "Decent for the price, nothing special.",
            "rating": 3.0,
            "author": "Bob Wilson",
            "platform": "amazon",
        },
    ]


# Environment fixtures
@pytest.fixture
def clean_env(monkeypatch) -> None:
    """Remove Sentimatrix-related environment variables."""
    for key in list(os.environ.keys()):
        if key.startswith("SENTIMATRIX_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def env_with_config(monkeypatch) -> None:
    """Set up environment variables for config testing."""
    monkeypatch.setenv("SENTIMATRIX_LLM__PROVIDER", "anthropic")
    monkeypatch.setenv("SENTIMATRIX_LLM__MODEL", "claude-3-sonnet")
    monkeypatch.setenv("SENTIMATRIX_DEBUG", "true")


# Markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "live: marks tests that require live API access")
