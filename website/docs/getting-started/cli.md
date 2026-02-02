---
title: CLI Usage
description: Use Sentimatrix from the command line
---

# Command Line Interface

Sentimatrix includes a powerful CLI for quick analysis without writing code.

## Installation

The CLI is installed automatically with Sentimatrix:

```bash
pip install sentimatrix
```

Verify installation:

```bash
sentimatrix --version
```

## Basic Commands

### Analyze Text

Analyze sentiment of text directly:

```bash
# Single text
sentimatrix analyze "This product is amazing!"

# Output
# Sentiment: positive
# Confidence: 96.7%
# Scores: positive=0.967, negative=0.021, neutral=0.012
```

### Analyze from File

Analyze text from a file (one text per line):

```bash
sentimatrix analyze --file reviews.txt
```

With JSON output:

```bash
sentimatrix analyze --file reviews.txt --format json > results.json
```

### Analyze with Emotions

Include emotion detection:

```bash
sentimatrix analyze "I'm so excited about this!" --emotions

# Output
# Sentiment: positive (97.2%)
# Emotions:
#   Primary: joy (89.3%)
#   Secondary: surprise (45.2%), anticipation (32.1%)
```

## Scraping Commands

### Scrape Reviews

Scrape reviews from supported platforms:

```bash
# Steam reviews
sentimatrix scrape https://store.steampowered.com/app/1245620 \
  --platform steam \
  --max-reviews 50

# Amazon reviews (requires browser)
sentimatrix scrape https://www.amazon.com/dp/B0BSHF7WHW \
  --platform amazon \
  --max-reviews 30 \
  --browser
```

### Scrape and Analyze

Combine scraping with analysis:

```bash
sentimatrix scrape-analyze https://store.steampowered.com/app/1245620 \
  --platform steam \
  --max-reviews 100 \
  --output analysis.json
```

### List Supported Platforms

```bash
sentimatrix platforms

# Output
# Platform Scrapers:
#   amazon     - Amazon product reviews
#   steam      - Steam game reviews
#   youtube    - YouTube comments
#   reddit     - Reddit posts and comments
#   imdb       - IMDB movie reviews
#   yelp       - Yelp business reviews
#   trustpilot - Trustpilot reviews
#   google     - Google Reviews
```

## Configuration

### Using Environment Variables

```bash
export GROQ_API_KEY="gsk_..."
export SENTIMATRIX_LLM_PROVIDER="groq"
export SENTIMATRIX_LLM_MODEL="llama-3.3-70b-versatile"

sentimatrix analyze "Test text" --llm
```

### Using Config File

Create `sentimatrix.yaml`:

```yaml
llm:
  provider: groq
  model: llama-3.3-70b-versatile

scraper:
  rate_limit:
    requests_per_second: 2
```

Use it:

```bash
sentimatrix analyze "Test" --config sentimatrix.yaml
```

### Specify Config Path

```bash
sentimatrix --config /path/to/config.yaml analyze "Test text"
```

## Output Formats

### Text (Default)

```bash
sentimatrix analyze "Great product!"

# Sentiment: positive
# Confidence: 96.7%
```

### JSON

```bash
sentimatrix analyze "Great product!" --format json

# {"sentiment": "positive", "confidence": 0.967, "scores": {...}}
```

### CSV

```bash
sentimatrix analyze --file reviews.txt --format csv > results.csv
```

### Table

```bash
sentimatrix analyze --file reviews.txt --format table

# ┌────────────────────────────────┬───────────┬────────────┐
# │ Text                           │ Sentiment │ Confidence │
# ├────────────────────────────────┼───────────┼────────────┤
# │ Great product!                 │ positive  │ 96.7%      │
# │ Terrible experience            │ negative  │ 94.2%      │
# │ It's okay                      │ neutral   │ 78.3%      │
# └────────────────────────────────┴───────────┴────────────┘
```

## LLM Commands

### Summarize Reviews

```bash
sentimatrix summarize --file reviews.txt --llm groq

# Summary:
# The product receives generally positive feedback with customers
# praising its build quality and value. Common complaints include
# shipping delays and limited color options.
```

### Generate Insights

```bash
sentimatrix insights --file reviews.txt --llm groq

# Pros:
#   + Excellent build quality
#   + Great value for money
#   + Fast shipping (usually)
#
# Cons:
#   - Limited color options
#   - Occasional quality control issues
#
# Recommendation: Buy
```

## Batch Processing

### Process Directory

```bash
sentimatrix batch /path/to/reviews/ \
  --pattern "*.txt" \
  --output results/ \
  --format json
```

### Parallel Processing

```bash
sentimatrix batch /path/to/reviews/ \
  --workers 4 \
  --output results/
```

## Utility Commands

### Check Configuration

```bash
sentimatrix config check

# Configuration:
#   LLM Provider: groq
#   LLM Model: llama-3.3-70b-versatile
#   Rate Limit: 2 req/s
#   Cache: memory
# Status: Valid
```

### List Providers

```bash
sentimatrix providers

# Cloud Providers:
#   openai     - OpenAI (GPT-4o, GPT-4o-mini)
#   anthropic  - Anthropic (Claude 3.5 Sonnet)
#   google     - Google (Gemini 2.0, 1.5)
#   groq       - Groq (LLaMA 3.3, Mixtral)
#   mistral    - Mistral (7B, 8x7B, Large)
#   cohere     - Cohere (Command R+)
#
# Inference Providers:
#   together   - Together AI (200+ models)
#   fireworks  - Fireworks AI
#   openrouter - OpenRouter
#   cerebras   - Cerebras
#   deepseek   - DeepSeek
#
# Local Providers:
#   ollama     - Ollama (localhost:11434)
#   lmstudio   - LM Studio (localhost:1234)
#   vllm       - vLLM
#   llamacpp   - llama.cpp
```

### Test Provider

```bash
sentimatrix test-provider groq

# Testing Groq...
# Model: llama-3.3-70b-versatile
# Status: Connected
# Latency: 142ms
# Test: PASSED
```

## Command Reference

| Command | Description |
|---------|-------------|
| `analyze` | Analyze sentiment of text |
| `scrape` | Scrape reviews from a platform |
| `scrape-analyze` | Scrape and analyze in one step |
| `summarize` | Generate LLM summary |
| `insights` | Generate pros/cons insights |
| `batch` | Process multiple files |
| `platforms` | List supported platforms |
| `providers` | List LLM providers |
| `config` | Configuration management |
| `test-provider` | Test LLM provider connection |

## Global Options

| Option | Description |
|--------|-------------|
| `--config FILE` | Path to configuration file |
| `--format FORMAT` | Output format (text, json, csv, table) |
| `--verbose` | Enable verbose output |
| `--quiet` | Suppress all output except results |
| `--version` | Show version and exit |
| `--help` | Show help message |

## Examples

### E-commerce Analysis

```bash
# Scrape and analyze Amazon product
sentimatrix scrape-analyze "https://www.amazon.com/dp/B0BSHF7WHW" \
  --platform amazon \
  --max-reviews 100 \
  --emotions \
  --llm groq \
  --output product_analysis.json
```

### Gaming Review Dashboard

```bash
# Analyze multiple Steam games
for game_id in 1245620 1174180 292030; do
  sentimatrix scrape-analyze "https://store.steampowered.com/app/$game_id" \
    --platform steam \
    --max-reviews 50 \
    --output "game_${game_id}.json"
done
```

### Social Media Monitoring

```bash
# Analyze Reddit discussion
sentimatrix scrape "https://reddit.com/r/technology/comments/abc123" \
  --platform reddit \
  --max-comments 200 \
  | sentimatrix analyze --stdin --emotions
```

## Next Steps

- **[Configuration](../configuration/)** - Configure the CLI
- **[Examples](../examples/)** - More CLI examples
- **[API Reference](../api/)** - Python API documentation
