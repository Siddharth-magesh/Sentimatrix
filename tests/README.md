# Sentimatrix Test Suite

## Structure

```
tests/
├── conftest.py           # Shared fixtures and pytest configuration
├── unit/                 # Unit tests
│   ├── core/             # Tests for core modules
│   │   ├── test_config.py       # Configuration tests
│   │   ├── test_exceptions.py   # Exception tests
│   │   ├── test_logger.py       # Logger tests
│   │   └── test_cache.py        # Cache tests
│   └── providers/        # Tests for provider modules
│       └── test_base.py         # Base provider tests
├── integration/          # Integration tests (to be implemented)
├── fixtures/             # Test fixtures and sample data
│   └── sample_data.py           # Sample test data
└── results/              # Test results (gitignored)
```

## Running Tests

### Using pytest directly

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/core/test_config.py

# Run specific test class
pytest tests/unit/core/test_config.py::TestLLMConfig

# Run specific test
pytest tests/unit/core/test_config.py::TestLLMConfig::test_default_values
```

### Using the test runner script

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run with coverage report
python run_tests.py --coverage

# Run with verbose output
python run_tests.py --verbose

# Skip slow tests
python run_tests.py --skip-slow

# Skip tests requiring live API access
python run_tests.py --skip-live
```

## Test Markers

Tests can be marked with:
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.live` - Tests requiring live API access

Example:
```python
@pytest.mark.slow
async def test_large_batch_processing():
    ...

@pytest.mark.live
async def test_openai_api():
    ...
```

## Coverage

To run tests with coverage:

```bash
pytest --cov=sentimatrix --cov-report=html
```

Coverage report will be generated in `tests/results/coverage_html/`.

## Writing Tests

### Test Organization

- Group related tests in classes
- Use descriptive test names: `test_what_should_happen`
- Use fixtures for common setup

### Example Test

```python
import pytest
from sentimatrix.core.config import LLMConfig

class TestLLMConfig:
    def test_default_values(self):
        """Test default LLM configuration values."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.temperature == 0.7

    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operations."""
        result = await some_async_function()
        assert result is not None
```

### Using Fixtures

```python
@pytest.fixture
def sample_config():
    """Provide a sample configuration."""
    return LLMConfig(provider="openai", model="gpt-4")

def test_with_fixture(sample_config):
    assert sample_config.model == "gpt-4"
```

## Test Results

Test results are saved to `tests/results/`:
- `test_results_YYYYMMDD_HHMMSS.xml` - JUnit XML format
- `test_summary_YYYYMMDD_HHMMSS.md` - Markdown summary
- `coverage_html/` - HTML coverage report
