---
title: YAML Configuration
description: Configure Sentimatrix using YAML files
---

# YAML Configuration

YAML files provide a clean, readable way to configure Sentimatrix.

## Basic Structure

```yaml title="sentimatrix.yaml"
# LLM Provider
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  temperature: 0.7

# Scraping
scrapers:
  provider: playwright
  headless: true

# ML Models
models:
  sentiment_model: cardiffnlp/twitter-roberta-base-sentiment-latest
  device: auto

# Caching
cache:
  enabled: true
  backend: memory
```

## Loading YAML

```python
from sentimatrix.config import SentimatrixConfig

# From specific file
config = SentimatrixConfig.from_file("config.yaml")

# With overrides
config = SentimatrixConfig.from_file("config.yaml", debug=True)
```

## File Locations

Sentimatrix searches for config files in order:

1. Explicit path provided
2. `sentimatrix.yaml` in current directory
3. `config/sentimatrix.yaml`
4. `~/.sentimatrix/config.yaml`

## JSON Support

JSON files also supported:

```json title="sentimatrix.json"
{
  "llm": {
    "provider": "groq",
    "model": "llama-3.3-70b-versatile"
  }
}
```

## Related

- [Configuration Overview](index.md)
- [Environment Variables](environment.md)
- [Python Objects](python.md)
