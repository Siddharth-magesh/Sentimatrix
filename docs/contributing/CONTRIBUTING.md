# Contributing to Sentimatrix

Thank you for your interest in contributing to Sentimatrix V2.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Virtual environment tool (venv, conda)

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Sentimatrix.git
cd Sentimatrix

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest

# Run specific test suite
pytest tests/unit/providers/llm/
pytest tests/unit/providers/scrapers/
```

---

## Project Structure

```
sentimatrix/
├── core/               # Core infrastructure
│   ├── config.py       # Configuration (Pydantic v2)
│   ├── logger.py       # Structured logging
│   ├── exceptions.py   # Exception hierarchy (50+ types)
│   ├── cache.py        # Memory & Redis caching
│   └── pipeline.py     # Pipeline orchestration
├── providers/
│   ├── base.py         # Provider interfaces
│   ├── llm/            # 19 LLM providers
│   │   ├── openai_provider.py
│   │   ├── groq_provider.py
│   │   ├── anthropic_provider.py
│   │   └── ... (16 more)
│   ├── scrapers/
│   │   ├── platforms/  # Platform scrapers (8)
│   │   └── commercial/ # Commercial APIs (7)
│   └── models/         # HuggingFace models
├── analysis/
│   ├── sentiment.py    # Sentiment analysis
│   ├── emotion.py      # Emotion detection
│   └── multimodal.py   # Audio/image/video
├── input/              # Input handlers
├── output/             # Exporters, formatters, visualizers
├── cli.py              # CLI interface
└── main.py             # Main Sentimatrix class
```

---

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Use the bug report template
3. Include:
   - Python version
   - Sentimatrix version (`pip show sentimatrix`)
   - Minimal reproduction code
   - Expected vs actual behavior
   - Full error traceback

### Suggesting Features

1. Check existing feature requests
2. Use the feature request template
3. Describe the use case
4. Explain why existing solutions are inadequate

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes following our style guide
4. Write/update tests (minimum 90% coverage)
5. Run the test suite: `pytest`
6. Run linting: `ruff check .`
7. Run formatting: `black .`
8. Run type checking: `mypy sentimatrix/`
9. Commit with descriptive messages
10. Push and create a Pull Request

---

## Adding New Providers

### Adding an LLM Provider

1. Create provider file in `sentimatrix/providers/llm/`:

```python
"""
NewProvider LLM Provider

Implements the BaseLLMProvider interface for NewProvider's API.
"""
from sentimatrix.providers.base import BaseLLMProvider, LLMResponse
from sentimatrix.core.config import LLMConfig

class NewProvider(BaseLLMProvider):
    """NewProvider implementation."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        super().__init__(config)
        self._client = None

    async def initialize(self) -> None:
        """Initialize the client."""
        # Setup client
        self._initialized = True

    async def close(self) -> None:
        """Close the client."""
        self._initialized = False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        self._ensure_initialized()
        # Implementation
        return LLMResponse(...)

    async def generate_stream(self, prompt: str, **kwargs):
        """Stream completion."""
        self._ensure_initialized()
        # Implementation
        yield "chunk"
```

2. Register in `__init__.py`:

```python
from .new_provider import NewProvider

__all__ = [..., "NewProvider"]
```

3. Add tests in `tests/unit/providers/llm/test_new_provider.py`
4. Update documentation in `docs/providers/`

### Adding a Platform Scraper

1. Create scraper in `sentimatrix/providers/scrapers/platforms/`:

```python
from sentimatrix.providers.scrapers.platforms.base import BasePlatformScraper
from sentimatrix.providers.base import Review

class NewPlatformScraper(BasePlatformScraper):
    """Scraper for NewPlatform."""

    PLATFORM = "newplatform"

    async def scrape_reviews(
        self,
        identifier: str,
        limit: int = 100,
        **kwargs,
    ) -> List[Review]:
        """Scrape reviews."""
        # Implementation
        return reviews

    @staticmethod
    def validate_id(identifier: str) -> bool:
        """Validate platform ID format."""
        return bool(identifier)

    @staticmethod
    def extract_id(url: str) -> Optional[str]:
        """Extract ID from URL."""
        # Implementation
        return None
```

2. Register in `__init__.py`
3. Add tests
4. Update scraper documentation

---

## Code Style

### Formatter and Linter

- **Formatter**: Black (line length 100)
- **Linter**: Ruff
- **Type checker**: mypy (strict mode)
- **Docstrings**: Google style

### Example Docstring

```python
async def analyze_sentiment(
    self,
    text: str,
    threshold: float = 0.5,
) -> SentimentResult:
    """
    Analyze sentiment of text.

    Args:
        text: Text to analyze.
        threshold: Confidence threshold (0-1).

    Returns:
        SentimentResult with sentiment label and confidence.

    Raises:
        ValidationError: If text is empty.
        ProviderError: If analysis fails.

    Example:
        >>> result = await analyzer.analyze_sentiment("Great product!")
        >>> print(result.sentiment)  # "positive"
    """
```

### Type Hints

All public functions must have complete type hints:

```python
from typing import List, Optional, Dict, Any

async def process_reviews(
    reviews: List[Review],
    options: Optional[Dict[str, Any]] = None,
) -> ProcessingResult:
    ...
```

---

## Testing Requirements

### Coverage Requirements

- Minimum 90% coverage for new code
- Unit tests for all public functions
- Integration tests for complex features
- Mock external dependencies

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestNewFeature:
    """Tests for new feature."""

    @pytest.fixture
    def sample_data(self):
        """Sample test data."""
        return {"key": "value"}

    @pytest.mark.asyncio
    async def test_success_case(self, sample_data):
        """Test successful operation."""
        result = await some_function(sample_data)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValidationError):
            await some_function(invalid_data)

    @pytest.mark.asyncio
    async def test_with_mock(self):
        """Test with mocked dependency."""
        with patch("module.external_call") as mock:
            mock.return_value = AsyncMock(return_value="mocked")
            result = await function_using_external()
            assert result == "expected"
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=sentimatrix --cov-report=html

# Specific module
pytest tests/unit/providers/llm/

# Verbose
pytest -v

# Only fast tests
pytest -m "not slow"

# Parallel execution
pytest -n auto
```

---

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guide
- [ ] All tests pass (`pytest`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy sentimatrix/`)
- [ ] New code has tests (90%+ coverage)
- [ ] Documentation updated if needed
- [ ] No merge conflicts

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address review comments
4. Squash commits before merge

---

## Commit Messages

Format: `type(scope): description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tool changes

Examples:
```
feat(llm): add Mistral provider
fix(scrapers): handle Amazon CAPTCHA
docs(api): update provider documentation
test(sentiment): add batch analysis tests
refactor(cache): improve TTL handling
```

---

## Getting Help

- Check existing documentation
- Search existing issues
- Ask in discussions
- Tag maintainers in complex PRs

---

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md
- Release notes
- README acknowledgments

Thank you for contributing to Sentimatrix.
