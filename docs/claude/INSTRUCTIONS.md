# Sentimatrix V2 - Claude Development Instructions

## Project Context

Sentimatrix is a Python library for sentiment analysis and web scraping. You are helping develop version 0.2.0 which is a major update from v0.1.7.

**Key Goals:**
- Modular, maintainable architecture
- Support for 15+ LLM providers
- Support for 20+ platform scrapers
- Async-first design
- 90%+ test coverage
- Comprehensive documentation

---

## Code Style Guidelines

### Python Version
- Target: Python 3.10+
- Use modern Python features (type hints, dataclasses, async/await)

### Formatting
- Formatter: Black (line length 100)
- Linter: Ruff
- Type checker: mypy

### Naming Conventions
```python
# Modules: snake_case
sentiment_analysis.py

# Classes: PascalCase
class SentimentAnalyzer:

# Functions/methods: snake_case
def analyze_sentiment():

# Constants: UPPER_SNAKE_CASE
DEFAULT_MODEL = "..."

# Private: _prefix
def _internal_method():
```

### Docstrings
Use Google style:
```python
def function(arg1: str, arg2: int = 10) -> Result:
    """Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.

    Example:
        >>> result = function("test")
    """
```

### Imports
```python
# Standard library
import os
from typing import List, Optional

# Third-party
import torch
from transformers import AutoModel

# Local
from sentimatrix.core import config
from sentimatrix.providers import get_provider
```

---

## Architecture Guidelines

### Provider Pattern
All providers must inherit from base classes:

```python
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        pass
```

### Configuration
Use Pydantic for all configuration:

```python
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
```

### Async-First
All I/O operations should be async:

```python
# Good
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Avoid
def fetch_data(url: str) -> dict:
    response = requests.get(url)
    return response.json()
```

### Error Handling
Use custom exceptions with context:

```python
class ProviderError(SentimatrixError):
    def __init__(self, provider: str, message: str, cause: Exception = None):
        self.provider = provider
        self.cause = cause
        super().__init__(f"[{provider}] {message}")
```

---

## Implementation Patterns

### LLM Provider Implementation

```python
# providers/llm/example_provider.py
from sentimatrix.providers.llm.base import BaseLLMProvider

class ExampleProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.model = model
        self._client = None  # Lazy initialization

    async def _get_client(self):
        if self._client is None:
            self._client = ExampleClient(api_key=self.api_key)
        return self._client

    async def generate(self, prompt: str, **kwargs) -> str:
        client = await self._get_client()
        try:
            response = await client.complete(
                model=self.model,
                prompt=prompt,
                **kwargs
            )
            return response.text
        except ExampleAPIError as e:
            raise ProviderError("example", str(e), e)
```

### Scraper Implementation

```python
# providers/scrapers/platforms/example.py
from sentimatrix.providers.scrapers.base import BasePlatformScraper

class ExampleScraper(BasePlatformScraper):
    def get_platform_name(self) -> str:
        return "example"

    def validate_url(self, url: str) -> bool:
        return "example.com" in url

    async def scrape_reviews(self, url: str, limit: int = 100) -> List[Review]:
        # Implementation
        pass
```

### Test Implementation

```python
# tests/providers/test_example_provider.py
import pytest
from unittest.mock import AsyncMock, patch

class TestExampleProvider:
    @pytest.fixture
    def provider(self):
        return ExampleProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        with patch.object(provider, '_get_client') as mock:
            mock.return_value.complete = AsyncMock(
                return_value=MockResponse(text="Hello")
            )
            result = await provider.generate("Hi")
            assert result == "Hello"

    @pytest.mark.asyncio
    async def test_generate_error(self, provider):
        with patch.object(provider, '_get_client') as mock:
            mock.return_value.complete = AsyncMock(
                side_effect=ExampleAPIError("Rate limited")
            )
            with pytest.raises(ProviderError):
                await provider.generate("Hi")
```

---

## File Structure Reference

```
sentimatrix/
├── __init__.py              # Public API exports
├── core/
│   ├── __init__.py
│   ├── config.py            # Pydantic config models
│   ├── pipeline.py          # Pipeline orchestration
│   ├── cache.py             # Caching layer
│   ├── logger.py            # Logging setup
│   └── exceptions.py        # Exception hierarchy
├── providers/
│   ├── __init__.py          # Provider registry
│   ├── base.py              # Base provider classes
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py          # BaseLLMProvider
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── ...
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── base.py          # BaseScraperProvider
│   │   ├── playwright_scraper.py
│   │   └── platforms/
│   │       ├── __init__.py
│   │       ├── amazon.py
│   │       └── ...
│   └── models/
│       ├── __init__.py
│       ├── base.py
│       └── sentiment.py
├── analysis/
│   ├── __init__.py
│   ├── sentiment.py
│   ├── emotion.py
│   └── aspect.py
├── output/
│   ├── __init__.py
│   ├── formatters.py
│   ├── exporters.py
│   └── visualizers.py
└── utils/
    ├── __init__.py
    └── helpers.py
```

---

## Common Tasks

### Adding a New LLM Provider

1. Create `providers/llm/newprovider_provider.py`
2. Inherit from `BaseLLMProvider`
3. Implement `generate()` and `generate_stream()`
4. Register in `providers/llm/__init__.py`
5. Add config schema
6. Write tests in `tests/providers/test_newprovider.py`
7. Update documentation

### Adding a New Platform Scraper

1. Create `providers/scrapers/platforms/newplatform.py`
2. Inherit from `BasePlatformScraper`
3. Implement `scrape_reviews()`, `get_platform_name()`, `validate_url()`
4. Register in `providers/scrapers/platforms/__init__.py`
5. Write tests
6. Update documentation

### Adding a New Feature

1. Update config schema if needed
2. Implement in appropriate module
3. Add to main `Sentimatrix` class if public
4. Write unit tests
5. Write integration tests if needed
6. Update API documentation
7. Add usage examples

---

## Quality Checklist

Before completing any implementation:

- [ ] Type hints on all functions
- [ ] Docstrings on public functions
- [ ] Error handling with custom exceptions
- [ ] Unit tests with >90% coverage
- [ ] No hardcoded values (use config)
- [ ] Async where appropriate
- [ ] Logging at appropriate levels
- [ ] No secrets in code
