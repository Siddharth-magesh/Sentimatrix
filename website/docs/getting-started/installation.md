---
title: Installation
description: Install Sentimatrix and configure your environment
---

# Installation

This guide covers installing Sentimatrix and configuring it for your use case.

## Basic Installation

Install Sentimatrix using pip:

```bash
pip install sentimatrix
```

This installs the core library with minimal dependencies, suitable for basic sentiment analysis using transformer models.

## Installation with Extras

Sentimatrix offers optional dependencies for extended functionality:

=== "LLM Providers"

    Install support for cloud LLM providers:

    ```bash
    pip install sentimatrix[llm]
    ```

    Includes: `openai`, `anthropic`, `google-generativeai`, `groq`, `mistralai`, `together`, `cohere`

=== "Web Scraping"

    Install web scraping capabilities:

    ```bash
    pip install sentimatrix[scraping]
    ```

    Includes: `playwright`, `selenium`, `beautifulsoup4`, `lxml`

    After installation, set up Playwright browsers:

    ```bash
    playwright install chromium
    ```

=== "ML Models"

    Install local transformer models:

    ```bash
    pip install sentimatrix[models]
    ```

    Includes: `transformers`, `torch`, `tokenizers`

=== "Caching"

    Install caching backends:

    ```bash
    pip install sentimatrix[cache]
    ```

    Includes: `redis`, `aiosqlite`

=== "Visualization"

    Install visualization tools:

    ```bash
    pip install sentimatrix[visualization]
    ```

    Includes: `matplotlib`, `plotly`, `pandas`

=== "Development"

    Install development dependencies:

    ```bash
    pip install sentimatrix[dev]
    ```

    Includes: `pytest`, `pytest-asyncio`, `ruff`, `black`, `mypy`

=== "Everything"

    Install all optional dependencies:

    ```bash
    pip install sentimatrix[all]
    ```

## Alternative Package Managers

=== "Poetry"

    ```bash
    # Basic installation
    poetry add sentimatrix

    # With extras
    poetry add sentimatrix[llm,scraping]
    ```

=== "uv"

    ```bash
    # Basic installation
    uv add sentimatrix

    # With extras
    uv add sentimatrix[llm,scraping]
    ```

=== "pdm"

    ```bash
    pdm add sentimatrix[all]
    ```

=== "From Source"

    ```bash
    git clone https://github.com/sentimatrix/sentimatrix.git
    cd sentimatrix
    pip install -e ".[all]"
    ```

## Setting Up Playwright

If you're using web scraping features, install Playwright browsers:

```bash
# Install Chromium (recommended)
playwright install chromium

# Or install all browsers
playwright install

# Install system dependencies (Linux)
playwright install-deps
```

!!! tip "Headless Mode"
    By default, Sentimatrix runs Playwright in headless mode. No display is required.

## Environment Configuration

### API Keys

Set up API keys for LLM providers:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."

# Groq
export GROQ_API_KEY="gsk_..."

# Other providers
export MISTRAL_API_KEY="..."
export COHERE_API_KEY="..."
export TOGETHER_API_KEY="..."
export FIREWORKS_API_KEY="..."
export OPENROUTER_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

### Using .env Files

Create a `.env` file in your project root:

```ini title=".env"
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Commercial Scraping APIs (optional)
SCRAPERAPI_KEY=...
APIFY_TOKEN=...
BRIGHTDATA_USERNAME=...
BRIGHTDATA_PASSWORD=...

# Local Providers
OLLAMA_HOST=http://localhost:11434
LMSTUDIO_HOST=http://localhost:1234
```

Sentimatrix automatically loads `.env` files using `python-dotenv`.

## Verifying Installation

Verify your installation is working:

```python
import sentimatrix

# Check version
print(f"Sentimatrix version: {sentimatrix.__version__}")

# Quick test
import asyncio
from sentimatrix import Sentimatrix

async def test():
    async with Sentimatrix() as sm:
        result = await sm.analyze("Hello, world!")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence:.2%}")

asyncio.run(test())
```

Expected output:

```
Sentimatrix version: 0.2.0
Sentiment: neutral
Confidence: 87.34%
```

## Troubleshooting

### Common Issues

??? question "ModuleNotFoundError: No module named 'sentimatrix'"

    Ensure you've activated your virtual environment:

    ```bash
    # Create and activate virtual environment
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # or
    .\venv\Scripts\activate  # Windows

    # Install sentimatrix
    pip install sentimatrix
    ```

??? question "playwright._impl._errors.Error: Executable doesn't exist"

    Install Playwright browsers:

    ```bash
    playwright install chromium
    playwright install-deps  # Linux only
    ```

??? question "ImportError: cannot import name 'Sentimatrix'"

    You may have a naming conflict. Check for local files named `sentimatrix.py`:

    ```bash
    # Find conflicting files
    find . -name "sentimatrix.py" -type f
    ```

??? question "SSL Certificate errors"

    Update your certificates:

    ```bash
    pip install --upgrade certifi
    ```

### Platform-Specific Notes

=== "Linux"

    Install system dependencies for Playwright:

    ```bash
    sudo apt-get update
    sudo apt-get install -y libgbm1 libasound2
    playwright install-deps
    ```

=== "macOS"

    No additional dependencies required. If using Homebrew Python:

    ```bash
    brew install python@3.11
    pip3 install sentimatrix
    ```

=== "Windows"

    Use PowerShell with administrator privileges for Playwright:

    ```powershell
    playwright install chromium
    ```

## Next Steps

- **[Quick Start](quickstart.md)** - Run your first sentiment analysis
- **[Configuration](../configuration/)** - Configure Sentimatrix for your needs
- **[LLM Providers](../providers/)** - Set up LLM providers
