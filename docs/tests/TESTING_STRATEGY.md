# Sentimatrix V2 - Testing Strategy

## Overview

V2 targets 90%+ code coverage with comprehensive testing across all modules.

---

## Test Categories

### 1. Unit Tests

**Scope:** Individual functions and classes

**Location:** `tests/unit/`

**Framework:** pytest

**Targets:**
- Data models and validators
- Utility functions
- Configuration parsing
- Data transformations
- Individual provider methods (mocked)

**Example:**
```python
# tests/unit/test_models.py
import pytest
from sentimatrix.models import Review, SentimentResult

class TestReview:
    def test_review_creation(self):
        review = Review(
            id="123",
            text="Great product!",
            source="amazon",
            platform="amazon"
        )
        assert review.id == "123"
        assert review.text == "Great product!"

    def test_review_validation_empty_text(self):
        with pytest.raises(ValueError):
            Review(id="123", text="", source="amazon", platform="amazon")

class TestSentimentResult:
    def test_valid_labels(self):
        result = SentimentResult(label="positive", score=0.95)
        assert result.label in ["positive", "negative", "neutral"]

    def test_score_range(self):
        with pytest.raises(ValueError):
            SentimentResult(label="positive", score=1.5)
```

---

### 2. Integration Tests

**Scope:** Component interactions

**Location:** `tests/integration/`

**Targets:**
- Provider integrations (with real APIs, optional)
- Pipeline execution
- Cache operations
- Database operations

**Example:**
```python
# tests/integration/test_pipeline.py
import pytest
from sentimatrix import Sentimatrix

class TestPipeline:
    @pytest.fixture
    def client(self):
        return Sentimatrix(config_path="tests/fixtures/test_config.yaml")

    @pytest.mark.asyncio
    async def test_sentiment_pipeline(self, client):
        result = await client.analyze_sentiment("This product is amazing!")
        assert result.label in ["positive", "negative", "neutral"]
        assert 0 <= result.score <= 1

    @pytest.mark.asyncio
    async def test_batch_processing(self, client):
        texts = ["Great!", "Terrible!", "Okay"]
        results = await client.analyze_sentiment_batch(texts)
        assert len(results) == 3
```

---

### 3. Provider Tests

**Scope:** Individual provider implementations

**Location:** `tests/providers/`

**Approach:**
- Mock tests (default)
- Live tests (optional, with API keys)

**Example:**
```python
# tests/providers/test_openai_provider.py
import pytest
from unittest.mock import AsyncMock, patch
from sentimatrix.providers.llm import OpenAIProvider

class TestOpenAIProvider:
    @pytest.fixture
    def provider(self):
        return OpenAIProvider(api_key="test-key", model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_generate_mocked(self, provider):
        with patch.object(provider.client.chat.completions, 'create') as mock:
            mock.return_value = AsyncMock(
                choices=[AsyncMock(message=AsyncMock(content="Test response"))]
            )
            result = await provider.generate("Test prompt")
            assert result == "Test response"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_generate_live(self):
        """Run with: pytest -m live --live-api-keys"""
        import os
        provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
        result = await provider.generate("Say 'test'")
        assert "test" in result.lower()
```

---

### 4. Scraper Tests

**Scope:** Scraping functionality

**Location:** `tests/scrapers/`

**Approach:**
- Use recorded responses (VCR)
- Mock network calls
- Optional live tests

**Example:**
```python
# tests/scrapers/test_playwright_scraper.py
import pytest
from sentimatrix.providers.scrapers import PlaywrightScraper

class TestPlaywrightScraper:
    @pytest.fixture
    async def scraper(self):
        scraper = PlaywrightScraper()
        await scraper.initialize()
        yield scraper
        await scraper.close()

    @pytest.mark.asyncio
    async def test_scrape_static_page(self, scraper, httpserver):
        httpserver.expect_request("/test").respond_with_data(
            "<html><body><div class='review'>Great!</div></body></html>"
        )
        result = await scraper.scrape(httpserver.url_for("/test"))
        assert "Great!" in result.content

    @pytest.mark.asyncio
    async def test_scrape_with_javascript(self, scraper, httpserver):
        # Test JavaScript rendering
        pass
```

---

### 5. End-to-End Tests

**Scope:** Full workflows

**Location:** `tests/e2e/`

**Targets:**
- Complete analysis pipelines
- API endpoints
- CLI commands

**Example:**
```python
# tests/e2e/test_full_workflow.py
import pytest
from sentimatrix import Sentimatrix

class TestFullWorkflow:
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_url_analysis_workflow(self):
        client = Sentimatrix()

        # Scrape reviews
        reviews = await client.scrape_reviews(
            "https://example.com/product",
            limit=10
        )
        assert len(reviews) > 0

        # Analyze sentiment
        results = await client.analyze_sentiment_batch(
            [r.text for r in reviews]
        )
        assert len(results) == len(reviews)

        # Generate summary
        summary = await client.summarize_reviews(reviews, results)
        assert len(summary) > 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_comparison_workflow(self):
        client = Sentimatrix()

        comparison = await client.compare_products(
            ["https://example.com/product1", "https://example.com/product2"]
        )
        assert "winner" in comparison
```

---

### 6. Performance Tests

**Scope:** Performance benchmarks

**Location:** `tests/performance/`

**Targets:**
- Inference latency
- Throughput
- Memory usage
- Scalability

**Example:**
```python
# tests/performance/test_inference_performance.py
import pytest
import time
from sentimatrix import Sentimatrix

class TestInferencePerformance:
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_inference_latency(self):
        client = Sentimatrix()

        start = time.perf_counter()
        await client.analyze_sentiment("Test text")
        latency = time.perf_counter() - start

        assert latency < 0.1  # 100ms target

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_throughput(self):
        client = Sentimatrix()
        texts = ["Test text"] * 100

        start = time.perf_counter()
        await client.analyze_sentiment_batch(texts)
        duration = time.perf_counter() - start

        throughput = len(texts) / duration
        assert throughput > 50  # 50 texts/second target
```

---

## Test Configuration

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    live: Tests requiring live API keys
    slow: Slow running tests

addopts =
    --strict-markers
    -v
    --tb=short
```

### conftest.py

```python
# tests/conftest.py
import pytest
import asyncio
from sentimatrix import Sentimatrix

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    return {
        "llm": {"provider": "mock"},
        "scrapers": {"provider": "mock"},
        "cache": {"backend": "memory"}
    }

@pytest.fixture
def mock_client(test_config):
    return Sentimatrix(config=test_config)

@pytest.fixture
def sample_reviews():
    return [
        {"text": "Amazing product!", "rating": 5},
        {"text": "Terrible quality", "rating": 1},
        {"text": "It's okay", "rating": 3}
    ]
```

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=sentimatrix --cov-report=html

# Parallel execution
pytest -n auto

# Specific markers
pytest -m "unit and not slow"

# Live API tests
pytest -m live --live-api-keys

# Performance tests
pytest -m performance --benchmark-only
```

---

## Coverage Requirements

| Module | Target Coverage |
|--------|-----------------|
| core/ | 95% |
| providers/ | 90% |
| analysis/ | 95% |
| utils/ | 90% |
| Overall | 90% |

---

## CI Integration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests
        run: pytest --cov=sentimatrix --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## Test Data Management

### Fixtures Location
```
tests/
├── fixtures/
│   ├── configs/
│   │   └── test_config.yaml
│   ├── html/
│   │   ├── amazon_reviews.html
│   │   └── steam_reviews.html
│   ├── responses/
│   │   ├── openai_response.json
│   │   └── groq_response.json
│   └── reviews/
│       └── sample_reviews.json
```

### VCR Cassettes

Use VCR to record and replay HTTP interactions:

```python
@pytest.mark.vcr
async def test_api_call():
    # First run: records actual API response
    # Subsequent runs: replays recorded response
    pass
```
