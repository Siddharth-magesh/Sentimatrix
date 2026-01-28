# Sentimatrix V2 - Performance Optimization

## Overview

This document covers performance optimization strategies for Sentimatrix V2 across different deployment scenarios.

---

## 1. Model Inference Optimization

### Batch Processing

Process multiple texts in batches rather than one-by-one.

```python
# Configuration
analysis:
  batch_size: 32
  max_batch_wait_ms: 100  # Wait up to 100ms to fill batch
```

**Performance Impact:**
| Batch Size | Throughput Improvement |
|------------|------------------------|
| 1 | Baseline |
| 8 | 3-4x |
| 32 | 8-10x |
| 64 | 10-12x |

### Model Quantization

Reduce model precision for faster inference.

| Precision | Memory | Speed | Quality |
|-----------|--------|-------|---------|
| FP32 | 100% | 1x | Best |
| FP16 | 50% | 1.5x | Near-best |
| INT8 | 25% | 2x | Good |
| INT4 | 12.5% | 3x | Acceptable |

**Configuration:**
```yaml
models:
  quantization: "int8"  # fp32, fp16, int8, int4
  use_flash_attention: true
```

### GPU Optimization

```yaml
inference:
  device: "cuda"
  gpu_memory_fraction: 0.9
  use_cuda_graphs: true
  compile_model: true  # torch.compile
```

### CPU Optimization

```yaml
inference:
  device: "cpu"
  num_threads: 8
  use_mkl: true  # Intel MKL
  use_onnx: true  # ONNX Runtime
```

---

## 2. Scraping Optimization

### Connection Pooling

```yaml
scrapers:
  connection_pool:
    max_connections: 100
    max_keepalive: 20
    keepalive_timeout: 30
```

### Async Concurrency

```yaml
scrapers:
  concurrency:
    max_concurrent_requests: 10
    max_concurrent_domains: 5
    delay_between_requests: 0.5
```

### Caching

```yaml
cache:
  enabled: true
  backend: "redis"  # memory, redis, sqlite
  ttl: 3600
  max_size: 10000

  # Cache keys
  cache_html: true
  cache_reviews: true
  cache_sentiment: true
```

### Browser Reuse

```yaml
playwright:
  reuse_context: true
  context_limit: 5
  page_pool_size: 10
```

---

## 3. Memory Optimization

### Streaming Processing

Process large datasets without loading everything into memory.

```python
# Instead of
results = analyze_all(reviews)  # Loads all in memory

# Use streaming
async for result in analyze_stream(reviews):
    process(result)
```

### Garbage Collection

```yaml
memory:
  gc_after_batch: true
  gc_threshold: 1000  # Force GC after N items
  clear_cache_on_gc: true
```

### Model Loading

```yaml
models:
  lazy_loading: true  # Load models on first use
  unload_after: 300   # Unload after 5 min of inactivity
  max_loaded_models: 3
```

---

## 4. LLM API Optimization

### Request Batching

```yaml
llm:
  batch_requests: true
  max_batch_size: 10
  batch_timeout_ms: 500
```

### Response Caching

```yaml
llm:
  cache:
    enabled: true
    ttl: 86400  # 24 hours
    hash_prompts: true
```

### Token Optimization

- Use concise prompts
- Set appropriate max_tokens
- Use stop sequences

```yaml
llm:
  max_tokens: 500  # Don't request more than needed
  stop_sequences: ["\n\n", "END"]
```

### Provider Selection

```yaml
llm:
  # Use cheaper provider for simple tasks
  routing:
    summarization: "groq"  # Fast, cheap
    analysis: "gpt-4o-mini"  # Good, moderate
    reasoning: "claude-3-5-sonnet"  # Best quality
```

---

## 5. Database/Storage Optimization

### Connection Pooling

```yaml
database:
  pool_size: 10
  max_overflow: 20
  pool_recycle: 3600
```

### Indexing

```yaml
database:
  indexes:
    - reviews.created_at
    - reviews.sentiment_label
    - analysis.url
```

### Write Batching

```yaml
database:
  batch_writes: true
  batch_size: 100
  flush_interval: 5
```

---

## 6. Network Optimization

### DNS Caching

```yaml
network:
  dns_cache: true
  dns_cache_ttl: 300
```

### Compression

```yaml
network:
  compression: true
  accept_encoding: "gzip, br"
```

### Keep-Alive

```yaml
network:
  keep_alive: true
  keep_alive_timeout: 30
```

---

## 7. Deployment Optimization

### Worker Configuration

```yaml
deployment:
  workers: 4  # Number of worker processes
  threads_per_worker: 2
  max_requests_per_worker: 1000
```

### Load Balancing

```yaml
deployment:
  load_balancer: "round_robin"
  health_check_interval: 10
```

### Auto-Scaling

```yaml
deployment:
  auto_scale:
    enabled: true
    min_replicas: 1
    max_replicas: 10
    target_cpu: 70
    target_memory: 80
```

---

## 8. Benchmarking

### Built-in Benchmarks

```bash
sentimatrix benchmark --type sentiment --samples 1000
sentimatrix benchmark --type scraping --urls urls.txt
sentimatrix benchmark --type llm --provider groq
```

### Metrics to Track

| Metric | Target |
|--------|--------|
| Sentiment inference | < 50ms/text |
| Batch inference (32) | < 500ms |
| Page scrape | < 5s |
| LLM response | < 2s |
| Memory usage | < 4GB |

---

## 9. Profiling

### Enable Profiling

```yaml
debug:
  profiling: true
  profile_output: "./profiles/"
  trace_requests: true
```

### Profile Analysis

```bash
sentimatrix profile analyze ./profiles/
```

---

## 10. Best Practices Summary

1. **Use batching** - Always batch when possible
2. **Cache aggressively** - Cache scrapes, models, LLM responses
3. **Right-size models** - Use smallest model that meets quality needs
4. **Async everything** - Use async I/O for network operations
5. **Pool connections** - Reuse HTTP and database connections
6. **Stream large data** - Don't load everything into memory
7. **Profile first** - Measure before optimizing
8. **Monitor continuously** - Track metrics in production
