---
title: Sentiment Models
description: Sentiment analysis model reference
---

# Sentiment Models

Available models for sentiment analysis.

## General Models (7)

| Model | Classes | Languages |
|-------|---------|-----------|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | 3 | English |
| `cardiffnlp/twitter-roberta-base-sentiment` | 3 | English |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 5 | 100+ |
| `distilbert-base-uncased-finetuned-sst-2-english` | 2 | English |
| `finiteautomata/bertweet-base-sentiment-analysis` | 3 | English |
| `siebert/sentiment-roberta-large-english` | 2 | English |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | 3 | 100+ |

## Domain-Specific Models (4)

| Model | Domain |
|-------|--------|
| `ProsusAI/finbert` | Financial |
| `yiyanghkust/finbert-tone` | Financial |
| `nlpaueb/legal-bert-base-uncased` | Legal |
| `allenai/scibert_scivocab_uncased` | Scientific |

## Configuration

```python
ModelConfig(
    sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device="auto",
    batch_size=32
)
```

## Domain Selection

```python
result = await sm.analyze(
    "Stock prices fell sharply",
    domain="financial"  # Uses FinBERT
)
```

## Performance

| Model | CPU Latency | GPU Latency |
|-------|-------------|-------------|
| DistilBERT | 25ms | 5ms |
| RoBERTa Base | 45ms | 8ms |
| RoBERTa Large | 120ms | 25ms |

## Related

- [API Overview](index.md)
- [Emotion Models](emotion-models.md)
- [Sentiment Analysis](../features/sentiment/index.md)
