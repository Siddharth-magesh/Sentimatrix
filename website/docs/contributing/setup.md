---
title: Development Setup
description: Set up your development environment
---

# Development Setup

Get started contributing to Sentimatrix.

## Prerequisites

- Python 3.10+
- Git
- GPU (optional, for faster testing)

## Clone Repository

```bash
git clone https://github.com/Siddharth-magesh/Sentimatrix.git
cd Sentimatrix
```

## Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Install Playwright (for scraper tests)

```bash
playwright install chromium
```

## Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=sentimatrix --cov-report=html

# Specific test file
pytest tests/test_sentiment.py

# Fast tests only
pytest -m "not slow"
```

## Code Quality

```bash
# Format code
black sentimatrix tests
isort sentimatrix tests

# Lint
ruff check sentimatrix tests

# Type check
mypy sentimatrix
```

## Build Documentation

```bash
cd website
pip install -r requirements.txt
mkdocs serve
```

## Related

- [Contributing Overview](index.md)
- [Testing](testing.md)
- [Adding Providers](providers.md)
