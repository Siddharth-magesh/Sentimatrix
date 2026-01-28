# Sentimatrix V2 - ML Model Tracking

## Status Legend

| Status | Meaning |
|--------|---------|
| Planned | Not started |
| In Progress | Currently being implemented |
| Implemented | Code complete |
| Tested | Unit tests passing |
| Working | Integration tested and verified |
| Stable | Production ready |

---

## Sentiment Analysis Models

| Model | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|----------|--------|-------------|--------|---------|-------|
| cardiffnlp/twitter-roberta-base-sentiment-latest | HuggingFace | P0 | **Working** | [x] | [x] | [x] | Default, 3-class (neg/neu/pos) |
| cardiffnlp/twitter-roberta-base-sentiment | HuggingFace | P1 | Planned | [ ] | [ ] | [ ] | Twitter optimized |
| nlptown/bert-base-multilingual-uncased-sentiment | HuggingFace | P1 | **Working** | [x] | [x] | [x] | 5-star rating, supported |
| distilbert-base-uncased-finetuned-sst-2-english | HuggingFace | P1 | Planned | [ ] | [ ] | [ ] | Fast, binary |
| finiteautomata/bertweet-base-sentiment-analysis | HuggingFace | P2 | **Working** | [x] | [x] | [x] | Twitter, supported |
| siebert/sentiment-roberta-large-english | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | High accuracy |
| lxyuan/distilbert-base-multilingual-cased-sentiments-student | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | Multilingual |

---

## Emotion Detection Models

| Model | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|----------|--------|-------------|--------|---------|-------|
| SamLowe/roberta-base-go_emotions | HuggingFace | P0 | **Working** | [x] | [x] | [x] | Default, 28 classes, top-k, multi-label |
| bhadresh-savani/distilbert-base-uncased-emotion | HuggingFace | P1 | **Working** | [x] | [x] | [x] | 6 basic emotions, supported |
| j-hartmann/emotion-english-distilroberta-base | HuggingFace | P1 | **Working** | [x] | [x] | [x] | 7 emotions, supported |
| cardiffnlp/twitter-roberta-base-emotion | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | Twitter, 4 classes |
| mrm8488/t5-base-finetuned-emotion | HuggingFace | P3 | Planned | [ ] | [ ] | [ ] | T5-based |

---

## Aspect-Based Sentiment Models

| Model | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|----------|--------|-------------|--------|---------|-------|
| yangheng/deberta-v3-base-absa-v1.1 | HuggingFace | P1 | Planned | [ ] | [ ] | [ ] | ABSA |
| yangheng/deberta-v3-large-absa-v1.1 | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | Large ABSA |
| kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | InstructABSA |

---

## Multilingual Models

| Model | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|----------|--------|-------------|--------|---------|-------|
| cardiffnlp/twitter-xlm-roberta-base-sentiment | HuggingFace | P1 | Planned | [ ] | [ ] | [ ] | 100+ languages |
| lxyuan/distilbert-base-multilingual-cased-sentiments-student | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | Distilled multilingual |
| nlptown/bert-base-multilingual-uncased-sentiment | HuggingFace | P1 | Planned | [ ] | [ ] | [ ] | Multilingual 5-star |

---

## Domain-Specific Models

| Model | Domain | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|--------|----------|--------|-------------|--------|---------|-------|
| ProsusAI/finbert | Financial | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | Finance sentiment |
| yiyanghkust/finbert-tone | Financial | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | Finance tone |
| nlpaueb/legal-bert-base-uncased | Legal | HuggingFace | P3 | Planned | [ ] | [ ] | [ ] | Legal text |
| allenai/scibert_scivocab_uncased | Scientific | HuggingFace | P3 | Planned | [ ] | [ ] | [ ] | Scientific text |

---

## Zero-Shot Classification

| Model | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|----------|--------|-------------|--------|---------|-------|
| facebook/bart-large-mnli | HuggingFace | P1 | Planned | [ ] | [ ] | [ ] | Zero-shot NLI |
| MoritzLaworers/DeBERTa-v3-base-mnli-fever-anli | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | DeBERTa NLI |
| cross-encoder/nli-deberta-v3-base | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | Cross-encoder |

---

## Embedding Models

| Model | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|----------|--------|-------------|--------|---------|-------|
| sentence-transformers/all-MiniLM-L6-v2 | HuggingFace | P1 | Planned | [ ] | [ ] | [ ] | Fast embeddings |
| sentence-transformers/all-mpnet-base-v2 | HuggingFace | P1 | Planned | [ ] | [ ] | [ ] | Quality embeddings |
| BAAI/bge-small-en-v1.5 | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | BGE embeddings |
| intfloat/e5-large-v2 | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | E5 embeddings |

---

## Speech-to-Text Models

| Model | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|----------|--------|-------------|--------|---------|-------|
| openai/whisper-base | OpenAI/HF | P1 | Planned | [ ] | [ ] | [ ] | Default STT |
| openai/whisper-small | OpenAI/HF | P1 | Planned | [ ] | [ ] | [ ] | Better quality |
| openai/whisper-medium | OpenAI/HF | P2 | Planned | [ ] | [ ] | [ ] | High quality |
| openai/whisper-large-v3 | OpenAI/HF | P2 | Planned | [ ] | [ ] | [ ] | Best quality |

---

## Vision Models

| Model | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|----------|--------|-------------|--------|---------|-------|
| llava | Ollama | P1 | Planned | [ ] | [ ] | [ ] | Image understanding |
| Salesforce/blip-image-captioning-base | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | Image captioning |
| Salesforce/blip2-opt-2.7b | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | BLIP-2 |

---

## Translation Models

| Model | Source | Priority | Status | Implemented | Tested | Working | Notes |
|-------|--------|----------|--------|-------------|--------|---------|-------|
| deep_translator | Library | P1 | Planned | [ ] | [ ] | [ ] | Google Translate |
| Helsinki-NLP/opus-mt-* | HuggingFace | P2 | Planned | [ ] | [ ] | [ ] | Offline translation |
| facebook/mbart-large-50-many-to-many-mmt | HuggingFace | P3 | Planned | [ ] | [ ] | [ ] | 50 languages |

---

## Implementation Summary

| Category | Total | P0 | P1 | P2 | P3 | Implemented | Working |
|----------|-------|----|----|----|----|-------------|---------|
| Sentiment | 7 | 1 | 3 | 3 | 0 | 3 | 3 |
| Emotion | 5 | 1 | 2 | 1 | 1 | 3 | 3 |
| ABSA | 3 | 0 | 1 | 2 | 0 | 0 | 0 |
| Multilingual | 3 | 0 | 2 | 1 | 0 | 0 | 0 |
| Domain-Specific | 4 | 0 | 0 | 2 | 2 | 0 | 0 |
| Zero-Shot | 3 | 0 | 1 | 2 | 0 | 0 | 0 |
| Embeddings | 4 | 0 | 2 | 2 | 0 | 0 | 0 |
| Speech-to-Text | 4 | 0 | 2 | 2 | 0 | 0 | 0 |
| Vision | 3 | 0 | 1 | 2 | 0 | 0 | 0 |
| Translation | 3 | 0 | 1 | 1 | 1 | 0 | 0 |
| **Total** | **39** | **2** | **15** | **18** | **4** | **6** | **6** |

### Model Provider Infrastructure

| Component | Status | Implemented | Tested | Working | Notes |
|-----------|--------|-------------|--------|---------|-------|
| HuggingFaceModelProvider | **Working** | [x] | [x] | [x] | Base provider for all HF models |
| SentimentModelProvider | **Working** | [x] | [x] | [x] | Specialized sentiment provider |
| EmotionModelProvider | **Working** | [x] | [x] | [x] | Top-k, multi-label, Ekman mapping |
| Device Auto-Detection | **Working** | [x] | [x] | [x] | CPU/CUDA/MPS automatic |
| Model Caching | **Working** | [x] | [x] | [x] | Global cache to avoid reloading |
| Batch Processing | **Working** | [x] | [x] | [x] | Efficient multi-text inference |

---

## Implementation Order

### Phase 1 (P0) - Core ✅ COMPLETE
1. ✅ cardiffnlp/twitter-roberta-base-sentiment-latest (default sentiment)
2. ✅ SamLowe/roberta-base-go_emotions (default emotion)

### Phase 2 (P1) - Important (Partially Complete)
1. ✅ Additional sentiment models (nlptown, bertweet)
2. ✅ Additional emotion models (hartmann, savani)
3. Multilingual support
4. Zero-shot classification
5. Embeddings (for similarity)
6. Whisper (audio)
7. LLaVA (vision)
8. Translation

### Phase 3 (P2+) - Extended
- Domain-specific models
- Additional multilingual
- Advanced vision/audio

**Phase 1 Progress:** 2/2 complete (100%)
**Phase 2 Progress:** 2/8 complete (25%)

---

## Model Performance Benchmarks

| Model | Dataset | Accuracy | F1 | Speed (CPU) | Speed (GPU) |
|-------|---------|----------|-----|-------------|-------------|
| twitter-roberta-sentiment | SST-2 | 94.8% | 0.94 | 50ms | 9ms |
| distilbert-sst-2 | SST-2 | 91.3% | 0.91 | 15ms | 3ms |
| go_emotions | GoEmotions | 51% | 0.48 | 55ms | 10ms |
| (To be filled during testing) | | | | | |

---

## Notes

- Update benchmarks as models are tested
- Track model versions and updates
- Document any fine-tuning performed
- Note hardware requirements for each model
