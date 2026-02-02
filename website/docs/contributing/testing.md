---
title: Testing Guide
description: How to write and run tests
---

# Testing Guide

Sentimatrix has 1100+ tests with 91% coverage.

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=sentimatrix --cov-report=html

# Specific file
pytest tests/test_sentiment.py

# Specific test
pytest tests/test_sentiment.py::test_analyze

# By marker
pytest -m "not slow"
pytest -m "integration"
```

## Test Structure

```
tests/
├── unit/
│   ├── test_config.py
│   ├── test_sentiment.py
│   └── test_emotion.py
├── integration/
│   ├── test_providers.py
│   └── test_scrapers.py
├── e2e/
│   └── test_full_pipeline.py
└── conftest.py          # Fixtures
```

## Writing Tests

### Unit Test Example

```python
import pytest
from sentimatrix.core.sentiment import SentimentAnalyzer

@pytest.mark.asyncio
async def test_analyze_positive():
    analyzer = SentimentAnalyzer()
    result = await analyzer.analyze("Great product!")

    assert result.sentiment == "positive"
    assert result.confidence > 0.8

@pytest.mark.asyncio
async def test_analyze_batch():
    analyzer = SentimentAnalyzer()
    texts = ["Good", "Bad", "Okay"]
    results = await analyzer.analyze_batch(texts)

    assert len(results) == 3
```

### Integration Test Example

```python
import pytest
from sentimatrix import Sentimatrix

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_analysis():
    async with Sentimatrix() as sm:
        result = await sm.analyze("Test text")
        assert result.sentiment in ["positive", "negative", "neutral"]
```

### Mocking External APIs

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_llm_provider():
    with patch("sentimatrix.providers.llm.openai.OpenAIProvider.generate") as mock:
        mock.return_value = LLMResponse(content="Test", ...)

        provider = OpenAIProvider(api_key="test")
        result = await provider.generate("Hello")

        assert result.content == "Test"
        mock.assert_called_once()
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.asyncio` | Async tests |
| `@pytest.mark.slow` | Slow tests |
| `@pytest.mark.integration` | Integration tests |
| `@pytest.mark.e2e` | End-to-end tests |
| `@pytest.mark.skipci` | Skip in CI |

## Fixtures

```python
# conftest.py
import pytest
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig

@pytest.fixture
def config():
    return SentimatrixConfig()

@pytest.fixture
async def sentimatrix(config):
    async with Sentimatrix(config) as sm:
        yield sm
```

## Related

- [Development Setup](setup.md)
- [Adding Providers](providers.md)
- [Adding Scrapers](scrapers.md)
