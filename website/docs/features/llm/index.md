---
title: LLM Features
description: Summarization, insights, and reasoning with 21 LLM providers
---

# LLM Features

Sentimatrix integrates with 21 LLM providers to provide advanced text analysis capabilities beyond basic sentiment detection.

## Overview

<div class="grid">

<div class="card">
<h3>:material-text-box: Summarization</h3>
<p>Generate concise summaries of reviews and feedback.</p>
</div>

<div class="card">
<h3>:material-lightbulb: Insights</h3>
<p>Extract pros, cons, themes, and recommendations.</p>
</div>

<div class="card">
<h3>:material-compare: Comparison</h3>
<p>Compare products or services based on reviews.</p>
</div>

<div class="card">
<h3>:material-brain: Reasoning</h3>
<p>Chain-of-thought analysis for complex understanding.</p>
</div>

</div>

## Quick Setup

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",  # Free tier available
        model="llama-3.3-70b-versatile"
    )
)

async with Sentimatrix(config) as sm:
    # Now LLM features are available
    summary = await sm.summarize_reviews(reviews)
```

## Review Summarization

Generate natural language summaries of review collections:

```python
async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="amazon")

    summary = await sm.summarize_reviews(reviews)
    print(summary)
```

Output:

```
Customers generally praise this product for its build quality and
value for money. The most common compliments focus on durability
and ease of use. However, some users report issues with the
included instructions and occasional shipping delays. Overall,
the product receives positive feedback with a recommendation
to buy for most use cases.
```

### Summarization Styles

```python
# Professional (default)
summary = await sm.summarize_reviews(reviews, style="professional")

# Casual
summary = await sm.summarize_reviews(reviews, style="casual")

# Bullet points
summary = await sm.summarize_reviews(reviews, style="bullet_points")

# Executive brief
summary = await sm.summarize_reviews(reviews, style="executive")
```

## Insight Generation

Extract structured insights from reviews:

```python
async with Sentimatrix(config) as sm:
    insights = await sm.generate_insights(reviews)

    print("PROS:")
    for pro in insights.pros:
        print(f"  + {pro}")

    print("\nCONS:")
    for con in insights.cons:
        print(f"  - {con}")

    print(f"\nTHEMES: {insights.themes}")
    print(f"RECOMMENDATION: {insights.recommendation}")
```

Output:

```
PROS:
  + Excellent build quality
  + Great value for the price
  + Easy to set up and use
  + Responsive customer support

CONS:
  - Instructions could be clearer
  - Shipping sometimes delayed
  - Limited color options

THEMES: ['quality', 'value', 'usability', 'support']
RECOMMENDATION: Recommended for most users seeking good value
```

### InsightsResult Structure

```python
@dataclass
class InsightsResult:
    summary: str           # Brief overview
    key_points: list[str]  # Main takeaways
    pros: list[str]        # Positive aspects
    cons: list[str]        # Negative aspects
    themes: list[str]      # Common topics
    recommendations: list[str]  # Suggestions
    raw_response: str      # Full LLM response
```

## Product Comparison

Compare products based on their reviews:

```python
async with Sentimatrix(config) as sm:
    # Scrape reviews for both products
    product_a_reviews = await sm.scrape_reviews(url_a, platform="amazon")
    product_b_reviews = await sm.scrape_reviews(url_b, platform="amazon")

    comparison = await sm.compare_products(
        product_a_reviews,
        product_b_reviews,
        product_a_name="Product A",
        product_b_name="Product B"
    )

    print(comparison.comparison_summary)
    print(f"Winner: {comparison.winner}")
```

### ComparisonResult Structure

```python
@dataclass
class ComparisonResult:
    product_a_id: str
    product_b_id: str
    analysis_a: ReviewAnalysisResult
    analysis_b: ReviewAnalysisResult
    comparison_summary: str
    winner: Optional[str]
```

## Streaming Responses

Get real-time streaming for long summaries:

```python
async with Sentimatrix(config) as sm:
    async for chunk in sm.stream_summary(reviews):
        print(chunk, end="", flush=True)
```

## Provider Fallback

Configure fallback providers for reliability:

```python
from sentimatrix.config import FallbackConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile"
    ),
    fallback=FallbackConfig(
        enabled=True,
        providers=[
            {"provider": "together", "model": "meta-llama/Llama-3-70b"},
            {"provider": "openai", "model": "gpt-4o-mini"},
        ],
        max_attempts=3,
    )
)
```

## Custom Prompts

Use custom prompts for specialized analysis:

```python
async with Sentimatrix(config) as sm:
    result = await sm.llm_analyze(
        reviews,
        prompt="""Analyze these product reviews and identify:
        1. Quality issues mentioned
        2. Comparison with competitors
        3. Suggestions for improvement

        Format as JSON.
        """,
        output_format="json"
    )
```

## Model Selection by Task

| Task | Recommended Provider | Model |
|------|---------------------|-------|
| Quick summaries | Groq | llama-3.3-70b |
| Detailed insights | OpenAI | gpt-4o |
| Complex reasoning | Anthropic | claude-3.5-sonnet |
| Cost-effective | DeepSeek | deepseek-chat |
| Privacy (local) | Ollama | llama3.2 |

## Performance Tips

1. **Batch Reviews for Context**
    ```python
    # Send multiple reviews in one request
    summary = await sm.summarize_reviews(reviews[:50])
    ```

2. **Use Streaming for UX**
    ```python
    # Better user experience for long responses
    async for chunk in sm.stream_summary(reviews):
        display(chunk)
    ```

3. **Configure Timeouts**
    ```python
    LLMConfig(timeout=60)  # For long summaries
    ```

4. **Cache Results**
    ```python
    CacheConfig(enabled=True, ttl=3600)
    ```

## Provider Manager

Sentimatrix includes a sophisticated provider manager:

- **Health Monitoring**: Track provider availability
- **Rate Limit Handling**: Automatic backoff on limits
- **Load Balancing**: Distribute requests
- **Lazy Loading**: Initialize only when needed

```python
# The manager handles all this automatically
async with Sentimatrix(config) as sm:
    # Automatically uses healthy providers
    # Falls back on failures
    # Respects rate limits
    summary = await sm.summarize_reviews(reviews)
```

## Error Handling

```python
from sentimatrix.exceptions import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
    TokenLimitExceededError,
)

async with Sentimatrix(config) as sm:
    try:
        summary = await sm.summarize_reviews(reviews)
    except RateLimitError as e:
        print(f"Rate limited, retry after {e.retry_after}s")
    except TokenLimitExceededError:
        # Too many reviews for context window
        summary = await sm.summarize_reviews(reviews[:25])
    except ProviderError as e:
        print(f"Provider error: {e}")
```

## Related

- [LLM Providers](../../providers/index.md)
- [Provider Selection Guide](../../providers/selection-guide.md)
- [Configuration](../../configuration/index.md)
