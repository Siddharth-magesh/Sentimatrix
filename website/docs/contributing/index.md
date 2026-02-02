---
title: Contributing
description: Contribute to Sentimatrix development
---

# Contributing

Thank you for your interest in contributing to Sentimatrix! This guide will help you get started.

## Ways to Contribute

<div class="grid">

<div class="card">
<h3>:material-bug: Report Bugs</h3>
<p>Found a bug? Open an issue with reproduction steps.</p>
</div>

<div class="card">
<h3>:material-lightbulb: Suggest Features</h3>
<p>Have an idea? We'd love to hear it!</p>
</div>

<div class="card">
<h3>:material-code-tags: Submit Code</h3>
<p>Fix bugs, add features, improve performance.</p>
</div>

<div class="card">
<h3>:material-file-document: Improve Docs</h3>
<p>Help us improve documentation.</p>
</div>

</div>

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Make (optional but recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/sentimatrix/sentimatrix.git
cd sentimatrix

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Install Playwright for scraper tests
playwright install chromium
```

### Verify Setup

```bash
# Run tests
pytest tests/

# Run linting
ruff check sentimatrix tests
black --check sentimatrix tests
mypy sentimatrix
```

## Project Structure

```
sentimatrix/
├── sentimatrix/           # Main package
│   ├── __init__.py
│   ├── core/              # Core functionality
│   ├── providers/         # LLM providers
│   ├── scrapers/          # Web scrapers
│   ├── models/            # Data models
│   ├── config/            # Configuration
│   └── utils/             # Utilities
├── tests/                 # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/                  # Documentation
├── website/               # Documentation website
└── examples/              # Example scripts
```

## Adding a New LLM Provider

1. Create provider file in `sentimatrix/providers/`:

```python
# sentimatrix/providers/my_provider.py
from .base import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    """My custom LLM provider."""

    name = "myprovider"

    async def initialize(self) -> None:
        """Initialize provider resources."""
        self.client = MyClient(api_key=self.config.api_key)

    async def complete(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """Generate completion."""
        response = await self.client.complete(
            model=self.config.model,
            prompt=prompt,
            **kwargs,
        )
        return response.text

    async def close(self) -> None:
        """Clean up resources."""
        await self.client.close()
```

2. Register in `sentimatrix/providers/__init__.py`:

```python
from .my_provider import MyProvider

PROVIDERS = {
    # ... existing providers
    "myprovider": MyProvider,
}
```

3. Add tests in `tests/unit/providers/test_my_provider.py`

4. Add documentation in `website/docs/providers/myprovider.md`

## Adding a New Platform Scraper

1. Create scraper file in `sentimatrix/scrapers/platforms/`:

```python
# sentimatrix/scrapers/platforms/my_platform.py
from ..base import BasePlatformScraper
from ...models import Review

class MyPlatformScraper(BasePlatformScraper):
    """Scraper for MyPlatform."""

    name = "myplatform"
    requires_browser = False

    async def scrape_reviews(
        self,
        url: str,
        max_reviews: int = 50,
        **kwargs,
    ) -> list[Review]:
        """Scrape reviews from MyPlatform."""
        reviews = []

        # Implementation here

        return reviews
```

2. Register in `sentimatrix/scrapers/__init__.py`

3. Add tests

4. Add documentation

## Code Style

We use the following tools:

- **Ruff** - Linting
- **Black** - Code formatting
- **isort** - Import sorting
- **mypy** - Type checking

### Pre-commit Hooks

Pre-commit hooks run automatically on commit:

```bash
# Manual run
pre-commit run --all-files
```

### Type Hints

All public APIs should have type hints:

```python
async def analyze(
    self,
    text: str,
    *,
    mode: str = "quick",
) -> SentimentResult:
    """Analyze sentiment of text."""
    ...
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/unit/test_analyzer.py

# With coverage
pytest tests/ --cov=sentimatrix --cov-report=html

# Skip slow tests
pytest tests/ -m "not slow"

# Run in parallel
pytest tests/ -n auto
```

### Writing Tests

```python
# tests/unit/providers/test_my_provider.py
import pytest
from sentimatrix.providers import MyProvider

@pytest.fixture
def provider():
    return MyProvider(config=MockConfig())

@pytest.mark.asyncio
async def test_complete(provider):
    """Test completion works."""
    result = await provider.complete("Hello")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_handles_error(provider):
    """Test error handling."""
    with pytest.raises(ProviderError):
        await provider.complete("")
```

## Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "Add my feature"
   ```

3. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```

4. **Fill out PR template**
   - Description of changes
   - Related issues
   - Testing done
   - Checklist

5. **Address review feedback**

6. **Merge after approval**

## Documentation

### Building Docs

```bash
cd website
pip install -r requirements.txt
mkdocs serve
```

Visit `http://localhost:8000` to preview.

### Writing Docs

- Use Markdown with MkDocs Material extensions
- Include code examples
- Add to navigation in `mkdocs.yml`

## Getting Help

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas
- **Discord**: Real-time chat (coming soon)

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. Please be respectful and inclusive.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
