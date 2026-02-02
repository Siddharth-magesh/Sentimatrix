---
title: Insight Generation
description: Extract structured insights from reviews
---

# Insight Generation

Extract structured insights including pros, cons, themes, and recommendations.

## Usage

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

## InsightsResult Structure

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

## Use Cases

- Product development feedback
- Competitive analysis
- Customer service improvement
- Marketing insights

## Related

- [LLM Features](index.md)
- [Summarization](summarization.md)
- [Reasoning](reasoning.md)
