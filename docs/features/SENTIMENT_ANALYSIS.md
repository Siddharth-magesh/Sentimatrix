# Sentimatrix V2 - Sentiment Analysis Features

## Overview

Sentiment analysis is the core capability of Sentimatrix. V2 provides multiple approaches to sentiment classification with varying levels of detail and performance characteristics.

---

## 1. Quick Sentiment Analysis

**Purpose:** Fast classification for high-volume processing

**Output:** Single label with confidence score

**Labels:** `positive`, `negative`, `neutral`

**Use Cases:**
- Real-time stream processing
- Batch classification of large datasets
- Quick filtering and categorization

**Performance Target:** < 50ms per text (CPU), < 10ms (GPU)

**Default Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`

---

## 2. Structured Sentiment Analysis

**Purpose:** Detailed sentiment with full metadata

**Output:**
```python
{
    "text": "Original input text",
    "label": "positive",
    "score": 0.92,
    "confidence": 0.89,
    "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "processing_time_ms": 45.2
}
```

**Use Cases:**
- Detailed analysis reports
- Quality-sensitive applications
- Audit and compliance

---

## 3. Fine-Grained Sentiment

**Purpose:** 5-class sentiment classification

**Labels:** `very_negative`, `negative`, `neutral`, `positive`, `very_positive`

**Use Cases:**
- Rating prediction
- Detailed sentiment gradients
- Review scoring

**Model Options:**
- `nlptown/bert-base-multilingual-uncased-sentiment` (1-5 stars)
- `cardiffnlp/twitter-roberta-base-sentiment` (3-class)

---

## 4. Aspect-Based Sentiment Analysis (ABSA)

**Purpose:** Sentiment per product/service aspect

**Aspects Detected:**
- Product quality
- Price/value
- Customer service
- Shipping/delivery
- Durability
- User experience
- Performance
- Design/aesthetics

**Output:**
```python
{
    "aspects": [
        {"aspect": "battery", "sentiment": "negative", "score": 0.85},
        {"aspect": "camera", "sentiment": "positive", "score": 0.92},
        {"aspect": "price", "sentiment": "neutral", "score": 0.65}
    ],
    "overall": {"label": "positive", "score": 0.72}
}
```

**Model Options:**
- `yangheng/deberta-v3-base-absa-v1.1`
- Custom fine-tuned models
- LLM-based extraction

---

## 5. Comparative Sentiment Analysis

**Purpose:** Compare sentiment across multiple products/sources

**Features:**
- Side-by-side comparison
- Winner determination
- Aspect-level comparison
- Statistical significance testing

**Output:**
```python
{
    "products": [
        {"name": "Product A", "sentiment_score": 0.78, "review_count": 150},
        {"name": "Product B", "sentiment_score": 0.65, "review_count": 200}
    ],
    "winner": "Product A",
    "comparison_summary": "Product A outperforms..."
}
```

---

## 6. Temporal Sentiment Analysis

**Purpose:** Track sentiment changes over time

**Features:**
- Time-series sentiment data
- Trend detection
- Anomaly detection
- Moving averages

**Use Cases:**
- Product launch monitoring
- Crisis detection
- Long-term brand tracking

---

## 7. Domain-Specific Sentiment

**Supported Domains:**

| Domain | Model | Optimized For |
|--------|-------|---------------|
| Financial | FinBERT | Stock news, earnings calls |
| Social Media | Twitter-RoBERTa | Tweets, short-form |
| Product Reviews | Review-specific models | E-commerce reviews |
| Healthcare | BioBERT variants | Medical text |

---

## 8. Multi-lingual Sentiment

**Supported Languages:** 100+ languages

**Approach:**
1. Language detection
2. Translation to English (optional)
3. Native multi-lingual model inference

**Models:**
- `xlm-roberta-base` (100+ languages)
- `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- Language-specific models

---

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| High volume | DistilBERT | Speed |
| Accuracy critical | RoBERTa-large | Accuracy |
| Multi-lingual | XLM-RoBERTa | Language coverage |
| Social media | Twitter-RoBERTa | Domain fit |
| Financial | FinBERT | Domain fit |
| Resource limited | TinyBERT | Size |

---

## Configuration Options

```yaml
sentiment:
  model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  device: "auto"  # auto, cpu, cuda, mps
  batch_size: 32
  max_length: 512
  return_all_scores: false
  confidence_threshold: 0.5
```

---

## Integration with LLMs

Sentiment results can be enhanced with LLM-generated insights:

1. **Summary Generation** - Natural language summary of sentiment distribution
2. **Reason Extraction** - Why users feel positive/negative
3. **Recommendation Generation** - Actionable suggestions
4. **Comparative Narrative** - Story-form comparison

---

## Performance Benchmarks

| Model | Accuracy (SST-2) | Speed (CPU) | Speed (GPU) |
|-------|------------------|-------------|-------------|
| DistilBERT | 91.3% | 15ms | 3ms |
| BERT-base | 92.7% | 45ms | 8ms |
| RoBERTa-base | 94.8% | 50ms | 9ms |
| RoBERTa-large | 96.4% | 120ms | 20ms |
