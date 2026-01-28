# Sentimatrix V2

Advanced sentiment analysis toolkit with multi-provider LLM support and web scraping capabilities.

## Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic, Groq, Google Gemini, Ollama, and more
- **Web Scraping**: Playwright, Selenium, and API-based scrapers for 20+ platforms
- **Sentiment Analysis**: Quick sentiment, emotion detection, aspect-based analysis
- **Flexible Configuration**: YAML/JSON files, environment variables, runtime overrides
- **Async-First Design**: Built for high-performance async operations
- **Type Safety**: Full type hints and Pydantic validation

## Installation

```bash
# Basic installation
pip install sentimatrix

# With LLM providers
pip install sentimatrix[llm]

# With scraping support
pip install sentimatrix[scraping]

# With ML models
pip install sentimatrix[models]

# Full installation
pip install sentimatrix[all]
```

## Quick Start

```python
from sentimatrix import Sentimatrix

# Create instance
sm = Sentimatrix()

# Quick sentiment analysis
result = await sm.analyze_sentiment("This product is amazing!")
print(result.label)  # "positive"
print(result.score)  # 0.95

# Emotion detection
emotions = await sm.detect_emotions("I'm so excited about this!")
print(emotions.dominant_emotion)  # "joy"

# Scrape and analyze
analysis = await sm.analyze_url("https://amazon.com/product/B0123")
print(analysis.summary)
```

## Configuration

### YAML Configuration

```yaml
# config.yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7

scrapers:
  provider: playwright
  headless: true
  timeout: 30

cache:
  enabled: true
  backend: memory
  ttl: 3600
```

```python
from sentimatrix import SentimatrixConfig

config = SentimatrixConfig.from_file("config.yaml")
sm = Sentimatrix(config)
```

### Environment Variables

```bash
export SENTIMATRIX_LLM__PROVIDER=openai
export SENTIMATRIX_LLM__MODEL=gpt-4
export SENTIMATRIX_DEBUG=true
```

## Project Structure

```
sentimatrix/
├── core/
│   ├── config.py      # Configuration management
│   ├── logger.py      # Structured logging
│   ├── exceptions.py  # Exception hierarchy
│   └── cache.py       # Caching system
├── providers/
│   ├── base.py        # Provider interfaces
│   ├── llm/           # LLM providers
│   ├── scrapers/      # Scraper providers
│   └── models/        # ML model providers
├── analysis/          # Analysis modules
├── input/             # Input handlers
└── output/            # Output formatters
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/sentimatrix/sentimatrix.git
cd sentimatrix

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=sentimatrix
```

### Running Tests

```bash
# All tests
python run_tests.py

# Unit tests only
python run_tests.py --unit

# With coverage
python run_tests.py --coverage
```

## Documentation

- [Architecture Overview](docs/architecture/OVERVIEW.md)
- [Configuration Guide](docs/usage/CONFIGURATION.md)
- [API Reference](docs/api/REFERENCE.md)
- [Provider Guide](docs/providers/OVERVIEW.md)

## Roadmap

See [ROADMAP.md](docs/tasks/ROADMAP.md) for development roadmap.

## Contributing

See [CONTRIBUTING.md](docs/contributing/CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see LICENSE file for details.

## Version

Current version: 0.2.0-dev (Stage 1 Complete)
