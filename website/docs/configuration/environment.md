---
title: Environment Variables
description: Configure Sentimatrix via environment variables
---

# Environment Variables

Environment variables are ideal for Docker, CI/CD, and secrets management.

## Prefix Convention

All Sentimatrix variables use `SENTIMATRIX_` prefix with double underscore (`__`) for nesting.

## Common Variables

```bash
# LLM Configuration
export SENTIMATRIX_LLM__PROVIDER="groq"
export SENTIMATRIX_LLM__MODEL="llama-3.3-70b-versatile"
export SENTIMATRIX_LLM__TEMPERATURE="0.7"

# API Keys (standard pattern)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."

# Scraper Configuration
export SENTIMATRIX_SCRAPERS__PROVIDER="playwright"
export SENTIMATRIX_SCRAPERS__HEADLESS="true"

# Cache
export SENTIMATRIX_CACHE__ENABLED="true"
export SENTIMATRIX_CACHE__BACKEND="redis"
export SENTIMATRIX_CACHE__REDIS_URL="redis://localhost:6379"

# Debug
export SENTIMATRIX_DEBUG="true"
```

## Loading from Environment

```python
from sentimatrix.config import SentimatrixConfig

# Automatically loads SENTIMATRIX_* variables
config = SentimatrixConfig.from_env()
```

## Dynamic API Key Resolution

Use `env:` prefix to reference environment variables:

```yaml
llm:
  api_key: "env:MY_CUSTOM_API_KEY"
```

## Docker Example

```dockerfile
ENV SENTIMATRIX_LLM__PROVIDER=groq
ENV SENTIMATRIX_LLM__MODEL=llama-3.3-70b-versatile
ENV GROQ_API_KEY=${GROQ_API_KEY}
```

## Related

- [Configuration Overview](index.md)
- [YAML Configuration](yaml.md)
- [Python Objects](python.md)
