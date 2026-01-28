# Sentimatrix V2 - Development Workflow

## Environment Setup

### Prerequisites

- Python 3.10+
- Git
- Docker (optional)
- CUDA-capable GPU (optional)

### Initial Setup

```bash
# Clone repository
git clone https://github.com/Siddharth-magesh/Sentimatrix.git
cd Sentimatrix

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Install Playwright browsers (for scraping)
playwright install chromium
```

### IDE Setup

**VS Code Extensions:**
- Python
- Pylance
- Black Formatter
- Ruff
- GitLens

**settings.json:**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

---

## Code Standards

### Style Guide

- **Formatter:** Black (line length 100)
- **Linter:** Ruff
- **Type Checker:** mypy
- **Docstrings:** Google style

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `sentiment_analysis.py` |
| Classes | PascalCase | `SentimentAnalyzer` |
| Functions | snake_case | `analyze_sentiment()` |
| Constants | UPPER_SNAKE | `DEFAULT_MODEL` |
| Private | _prefix | `_internal_method()` |

### Import Order

```python
# Standard library
import os
import json
from typing import List, Dict

# Third-party
import torch
from transformers import AutoModel

# Local
from sentimatrix.core import config
from sentimatrix.providers import get_provider
```

---

## Git Workflow

### Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/description` | `feature/add-tiktok-scraper` |
| Bug fix | `fix/description` | `fix/sentiment-score-range` |
| Docs | `docs/description` | `docs/update-api-reference` |
| Refactor | `refactor/description` | `refactor/provider-interface` |

### Commit Messages

Format: `type(scope): description`

```
feat(scrapers): add TikTok scraper support
fix(sentiment): correct score normalization
docs(api): update provider documentation
refactor(core): simplify pipeline execution
test(providers): add OpenAI provider tests
```

### Pull Request Process

1. Create feature branch from `main`
2. Make changes with atomic commits
3. Run tests locally: `pytest`
4. Run linting: `ruff check .`
5. Run formatting: `black .`
6. Create PR with description
7. Address review comments
8. Squash and merge

---

## Adding New Features

### Adding a New LLM Provider

1. Create provider file:
```
providers/llm/new_provider.py
```

2. Implement interface:
```python
from sentimatrix.providers.llm.base import BaseLLMProvider

class NewProvider(BaseLLMProvider):
    async def generate(self, prompt: str, **kwargs) -> str:
        ...

    async def generate_stream(self, prompt: str, **kwargs):
        ...
```

3. Register provider:
```python
# providers/llm/__init__.py
from .new_provider import NewProvider

PROVIDERS = {
    ...
    "new_provider": NewProvider,
}
```

4. Add configuration:
```yaml
# In config schema
new_provider:
  api_key: str
  model: str
```

5. Write tests:
```python
# tests/providers/test_new_provider.py
```

6. Update documentation:
```markdown
# docs/providers/NEW_PROVIDER.md
```

### Adding a New Scraper

1. Create scraper file:
```
providers/scrapers/platforms/new_platform.py
```

2. Implement interface:
```python
from sentimatrix.providers.scrapers.base import BasePlatformScraper

class NewPlatformScraper(BasePlatformScraper):
    async def scrape_reviews(self, url: str, limit: int) -> List[Review]:
        ...

    def get_platform_name(self) -> str:
        return "new_platform"
```

3. Register scraper
4. Write tests
5. Update documentation

---

## Testing During Development

### Quick Test Run

```bash
# Run specific test file
pytest tests/unit/test_sentiment.py -v

# Run specific test
pytest tests/unit/test_sentiment.py::test_positive_sentiment -v

# Run with print output
pytest -s

# Run only fast tests
pytest -m "not slow"
```

### Coverage Check

```bash
# Run with coverage
pytest --cov=sentimatrix --cov-report=term-missing

# Generate HTML report
pytest --cov=sentimatrix --cov-report=html
open htmlcov/index.html
```

### Type Checking

```bash
mypy sentimatrix/
```

---

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Debug Configuration

```yaml
debug:
  enabled: true
  log_requests: true
  log_responses: true
  save_html: true
```

### Using Debugger

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use VS Code debugger with launch.json
```

---

## Documentation

### Building Docs

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build docs
cd docs
make html

# Serve locally
python -m http.server -d _build/html
```

### Docstring Format

```python
def analyze_sentiment(
    text: str,
    model: str = "default",
    return_scores: bool = False
) -> SentimentResult:
    """Analyze sentiment of input text.

    Args:
        text: The text to analyze.
        model: Model name to use for analysis.
        return_scores: Whether to return all class scores.

    Returns:
        SentimentResult containing label and confidence score.

    Raises:
        ValueError: If text is empty.
        ModelNotFoundError: If specified model is not available.

    Example:
        >>> result = analyze_sentiment("Great product!")
        >>> print(result.label)
        positive
    """
```

---

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch: `release/v0.2.0`
4. Run full test suite
5. Build package: `python -m build`
6. Test on TestPyPI
7. Merge to main
8. Tag release: `git tag v0.2.0`
9. Publish to PyPI
10. Create GitHub release
