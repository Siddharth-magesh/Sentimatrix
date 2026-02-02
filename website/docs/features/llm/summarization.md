---
title: Review Summarization
description: LLM-powered review summaries
---

# Review Summarization

Generate natural language summaries of review collections using LLM.

## Usage

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

## Styles

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

## Length Control

```python
# Short summary
summary = await sm.summarize_reviews(reviews, max_length=200)

# Detailed summary
summary = await sm.summarize_reviews(reviews, max_length=1000)
```

## Streaming

```python
async for chunk in sm.stream_summary(reviews):
    print(chunk, end="", flush=True)
```

## Related

- [LLM Features](index.md)
- [Insights](insights.md)
- [Provider Selection](../../providers/selection-guide.md)
