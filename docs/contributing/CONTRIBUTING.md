# Contributing to Sentimatrix

Thank you for your interest in contributing to Sentimatrix.

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

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

---

## How to Contribute

### Reporting Bugs

1. Check existing issues first
2. Use the bug report template
3. Include:
   - Python version
   - Sentimatrix version
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
4. Write/update tests
5. Run the test suite: `pytest`
6. Run linting: `ruff check .`
7. Run formatting: `black .`
8. Commit with descriptive messages
9. Push and create a Pull Request

---

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guide
- [ ] All tests pass
- [ ] New code has tests
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

## Development Guidelines

### Code Style

- Formatter: Black (line length 100)
- Linter: Ruff
- Type checker: mypy
- Docstrings: Google style

### Testing Requirements

- Minimum 90% coverage for new code
- Unit tests for all public functions
- Integration tests for complex features
- Mock external dependencies

### Documentation

- Docstrings for all public APIs
- Update relevant .md files
- Add examples for new features

### Commit Messages

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
feat(scrapers): add TikTok scraper
fix(sentiment): correct score normalization
docs(api): update provider documentation
```

---

## Architecture Overview

```
sentimatrix/
├── core/           # Core infrastructure
├── providers/      # Provider implementations
│   ├── llm/        # LLM providers
│   ├── scrapers/   # Scraper providers
│   └── models/     # ML model providers
├── analysis/       # Analysis modules
├── output/         # Export and visualization
└── utils/          # Utilities
```

### Adding New Providers

1. Create provider file in appropriate directory
2. Inherit from base provider class
3. Implement required interface methods
4. Register in `__init__.py`
5. Add configuration schema
6. Write comprehensive tests
7. Update documentation

---

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=sentimatrix

# Specific test file
pytest tests/unit/test_sentiment.py

# Verbose output
pytest -v

# Run only fast tests
pytest -m "not slow"
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
