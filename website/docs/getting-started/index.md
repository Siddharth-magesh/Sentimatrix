---
title: Getting Started
description: Start using Sentimatrix for sentiment analysis in minutes
---

# Getting Started

Welcome to Sentimatrix! This section will help you get up and running with sentiment analysis in just a few minutes.

## Overview

Sentimatrix provides a comprehensive toolkit for:

- **Sentiment Analysis** - Classify text as positive, negative, or neutral
- **Emotion Detection** - Identify emotions like joy, anger, sadness, and more
- **Review Scraping** - Collect reviews from Amazon, Steam, YouTube, Reddit, and other platforms
- **LLM Integration** - Enhance analysis with GPT-4, Claude, Gemini, and 16 other providers

## Quick Navigation

<div class="grid">

<div class="card">
<h3>:material-download: Installation</h3>
<p>Install Sentimatrix and its dependencies using pip, poetry, or uv.</p>
<p><a href="installation/">Read more →</a></p>
</div>

<div class="card">
<h3>:material-rocket-launch: Quick Start</h3>
<p>Get started with basic sentiment analysis in under 5 minutes.</p>
<p><a href="quickstart/">Read more →</a></p>
</div>

<div class="card">
<h3>:material-test-tube: First Analysis</h3>
<p>Build your first complete analysis pipeline with web scraping.</p>
<p><a href="first-analysis/">Read more →</a></p>
</div>

<div class="card">
<h3>:material-console: CLI Usage</h3>
<p>Use Sentimatrix from the command line for quick analysis.</p>
<p><a href="cli/">Read more →</a></p>
</div>

</div>

## Prerequisites

Before installing Sentimatrix, ensure you have:

- **Python 3.10+** - Sentimatrix requires Python 3.10 or later
- **pip** - Python package installer (comes with Python)
- **Git** - For cloning the repository (optional)

### Optional Dependencies

Depending on your use case, you may need:

| Feature | Requirement |
|---------|-------------|
| Web Scraping | Playwright browsers (`playwright install`) |
| LLM Providers | API keys for your chosen provider |
| Local LLMs | Ollama, LM Studio, or vLLM running locally |
| Caching | Redis server (optional, for distributed caching) |

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 500 MB | 2+ GB |
| Python | 3.10 | 3.11+ |

!!! note "GPU Support"
    GPU acceleration is optional but recommended for:

    - Local transformer models
    - Local LLMs (Ollama, vLLM)
    - Batch processing large datasets

## Next Steps

1. **[Install Sentimatrix](installation/)** - Get the library installed
2. **[Quick Start Guide](quickstart/)** - Run your first analysis
3. **[First Analysis](first-analysis/)** - Build a complete pipeline

---

Need help? Check our [troubleshooting guide](../guides/troubleshooting/) or [open an issue](https://github.com/sentimatrix/sentimatrix/issues).
