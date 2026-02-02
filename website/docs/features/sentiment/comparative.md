---
title: Comparative Sentiment
description: Compare sentiment across products or time
---

# Comparative Sentiment

Compare sentiment distribution across products, services, or competitors.

## Usage

```python
products = {
    "Product A": [
        "Great quality product",
        "Love this item",
        "Some issues with durability"
    ],
    "Product B": [
        "Okay product",
        "Not worth the price",
        "Returned it"
    ]
}

comparison = await sm.compare_sentiment(products)

print(comparison)
# {
#   'Product A': {'positive': 0.67, 'negative': 0.33, 'avg_score': 0.45},
#   'Product B': {'positive': 0.00, 'negative': 0.67, 'avg_score': -0.52},
#   'winner': 'Product A'
# }
```

## Full Product Comparison

```python
# Scrape both products
reviews_a = await sm.scrape_reviews(url_a, platform="amazon")
reviews_b = await sm.scrape_reviews(url_b, platform="amazon")

# Compare
comparison = await sm.compare_products(
    reviews_a,
    reviews_b,
    product_a_name="Our Product",
    product_b_name="Competitor"
)

print(comparison.comparison_summary)
print(f"Winner: {comparison.winner}")
```

## Metrics Compared

- Positive/negative ratio
- Average polarity score
- Sentiment distribution
- Key themes per product

## Related

- [Sentiment Overview](index.md)
- [Temporal Analysis](temporal.md)
