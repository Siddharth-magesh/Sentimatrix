# Sentimatrix V2 - Advanced Inference

## Overview

This document covers advanced inference techniques, optimization strategies, and specialized model usage patterns.

---

## 1. Reasoning Models

### OpenAI o1 Series

**Description:** Models optimized for complex reasoning tasks.

**Models:**
| Model | Context | Thinking | Best For |
|-------|---------|----------|----------|
| o1-preview | 128K | Extended | Complex analysis |
| o1-mini | 128K | Standard | Fast reasoning |

**Usage:**
```python
result = await sm.analyze(
    text,
    model="o1-preview",
    reasoning_mode=True
)

# Access reasoning trace
print(result.reasoning_trace)
```

**Configuration:**
```yaml
reasoning:
  model: "o1-preview"
  max_reasoning_tokens: 5000
  show_reasoning: true
```

**Best Practices:**
- Use for complex, multi-step analysis
- Higher latency, higher accuracy
- Don't use for simple classification
- Cost is higher due to reasoning tokens

### DeepSeek R1

**Description:** Open-source reasoning model.

**Usage:**
```python
result = await sm.analyze(
    text,
    provider="deepseek",
    model="deepseek-reasoner",
    reasoning_mode=True
)
```

**Advantages:**
- Much cheaper than o1
- Good reasoning quality
- Longer context support

---

## 2. Function Calling / Tool Use

### Basic Function Calling

```python
tools = [
    {
        "name": "get_sentiment",
        "description": "Get sentiment of text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "detailed": {"type": "boolean"}
            },
            "required": ["text"]
        }
    }
]

response = await llm.generate(
    prompt="Analyze the sentiment of: Great product!",
    tools=tools,
    tool_choice="auto"
)

if response.tool_calls:
    for call in response.tool_calls:
        result = await execute_tool(call.name, call.arguments)
```

### Parallel Function Calling

```python
# Multiple tools can be called in parallel
response = await llm.generate(
    prompt="Compare sentiment of these two products",
    tools=tools,
    parallel_tool_calls=True
)

# Execute all tool calls concurrently
results = await asyncio.gather(*[
    execute_tool(call.name, call.arguments)
    for call in response.tool_calls
])
```

### Forced Tool Use

```python
# Force specific tool
response = await llm.generate(
    prompt=prompt,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_sentiment"}}
)

# Force any tool (no text response)
response = await llm.generate(
    prompt=prompt,
    tools=tools,
    tool_choice="required"
)
```

---

## 3. Structured Output

### JSON Mode

```python
response = await llm.generate(
    prompt="Analyze sentiment and return JSON",
    response_format={"type": "json_object"}
)
result = json.loads(response)
```

### JSON Schema Enforcement

```python
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "aspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sentiment": {"type": "string"}
                }
            }
        }
    },
    "required": ["sentiment", "confidence"]
}

response = await llm.generate(
    prompt=prompt,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_analysis",
            "schema": schema,
            "strict": True
        }
    }
)
```

### Pydantic Integration

```python
from pydantic import BaseModel
from instructor import patch

class SentimentOutput(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str

# Using instructor library for structured outputs
client = patch(openai_client)
result = await client.chat.completions.create(
    model="gpt-4o",
    response_model=SentimentOutput,
    messages=[{"role": "user", "content": prompt}]
)
```

---

## 4. Context Window Management

### Long Context Handling

```python
class ContextManager:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def fit_context(self, texts: List[str], query: str) -> str:
        """Fit as much context as possible within limit"""
        query_tokens = count_tokens(query)
        available = self.max_tokens - query_tokens - 500  # Buffer

        selected = []
        current_tokens = 0

        for text in texts:
            text_tokens = count_tokens(text)
            if current_tokens + text_tokens <= available:
                selected.append(text)
                current_tokens += text_tokens
            else:
                break

        return "\n\n".join(selected)
```

### Chunking Strategies

```python
class ChunkingStrategy:
    @staticmethod
    def fixed_size(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fixed-size chunks with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    @staticmethod
    def semantic(text: str, model: str) -> List[str]:
        """Semantic chunking based on meaning"""
        # Use embeddings to find natural break points
        pass

    @staticmethod
    def sentence(text: str) -> List[str]:
        """Chunk by sentences"""
        return sent_tokenize(text)
```

### Map-Reduce for Large Inputs

```python
async def analyze_large_document(text: str, chunk_size: int = 4000) -> str:
    # Split into chunks
    chunks = ChunkingStrategy.fixed_size(text, chunk_size, overlap=200)

    # Map: Analyze each chunk
    chunk_results = await asyncio.gather(*[
        analyze_chunk(chunk)
        for chunk in chunks
    ])

    # Reduce: Combine results
    combined = await reduce_results(chunk_results)

    return combined
```

---

## 5. Multi-Model Strategies

### Model Ensemble

```python
class ModelEnsemble:
    def __init__(self, models: List[str], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)

    async def predict(self, text: str) -> SentimentResult:
        # Get predictions from all models
        predictions = await asyncio.gather(*[
            get_prediction(model, text)
            for model in self.models
        ])

        # Weighted voting
        scores = {"positive": 0, "negative": 0, "neutral": 0}
        for pred, weight in zip(predictions, self.weights):
            scores[pred.label] += weight * pred.confidence

        best_label = max(scores, key=scores.get)
        return SentimentResult(
            label=best_label,
            confidence=scores[best_label] / sum(self.weights)
        )
```

### Cascade Model Selection

```python
class CascadeSelector:
    """Use simpler models first, escalate if uncertain"""

    def __init__(self):
        self.models = [
            ("fast", "gpt-4o-mini", 0.9),    # Fast, cheap, high threshold
            ("balanced", "gpt-4o", 0.8),      # Balanced
            ("accurate", "o1-mini", 0.0)      # Most accurate, no threshold
        ]

    async def predict(self, text: str) -> SentimentResult:
        for name, model, threshold in self.models:
            result = await analyze_with_model(model, text)

            if result.confidence >= threshold:
                return result

        return result  # Return last result
```

### Router-Based Selection

```python
class ModelRouter:
    """Route to appropriate model based on task"""

    def __init__(self):
        self.rules = [
            (lambda t: len(t) < 100, "gpt-4o-mini"),      # Short text
            (lambda t: len(t) > 10000, "gemini-1.5-pro"), # Long text
            (lambda t: "financial" in t.lower(), "finbert"), # Domain
            (lambda t: True, "gpt-4o")                     # Default
        ]

    def select_model(self, text: str) -> str:
        for condition, model in self.rules:
            if condition(text):
                return model
        return "gpt-4o"
```

---

## 6. Caching and Optimization

### Semantic Caching

```python
class SemanticCache:
    def __init__(self, embedding_model: str, threshold: float = 0.95):
        self.cache = {}
        self.embeddings = []
        self.threshold = threshold

    async def get_or_compute(self, text: str, compute_fn: Callable) -> Any:
        # Check for semantically similar cached result
        text_embedding = await embed(text)

        for i, cached_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(text_embedding, cached_embedding)
            if similarity >= self.threshold:
                return list(self.cache.values())[i]

        # Compute and cache
        result = await compute_fn(text)
        self.cache[text] = result
        self.embeddings.append(text_embedding)
        return result
```

### Batch Optimization

```python
class BatchOptimizer:
    def __init__(self, batch_size: int = 32, max_wait_ms: int = 100):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()
        self.results = {}

    async def add(self, text: str) -> str:
        """Add to batch and wait for result"""
        future = asyncio.Future()
        await self.queue.put((text, future))
        return await future

    async def process_batches(self):
        """Background task to process batches"""
        while True:
            batch = []
            deadline = time.time() + self.max_wait_ms / 1000

            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=deadline - time.time()
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if batch:
                texts = [item[0] for item in batch]
                results = await batch_analyze(texts)

                for (text, future), result in zip(batch, results):
                    future.set_result(result)
```

---

## 7. Specialized Inference Patterns

### Confidence Calibration

```python
class ConfidenceCalibrator:
    """Calibrate model confidence scores"""

    def __init__(self, calibration_data: List[Tuple[float, bool]]):
        # Train isotonic regression on validation data
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        confidences = [c for c, _ in calibration_data]
        correct = [1 if c else 0 for _, c in calibration_data]
        self.calibrator.fit(confidences, correct)

    def calibrate(self, confidence: float) -> float:
        return self.calibrator.predict([confidence])[0]
```

### Uncertainty Quantification

```python
class UncertaintyEstimator:
    async def estimate(self, text: str, n_samples: int = 5) -> UncertaintyResult:
        # Multiple forward passes with temperature
        predictions = await asyncio.gather(*[
            analyze_with_temperature(text, temperature=0.8)
            for _ in range(n_samples)
        ])

        # Calculate uncertainty metrics
        labels = [p.label for p in predictions]
        scores = [p.score for p in predictions]

        return UncertaintyResult(
            mean_confidence=np.mean(scores),
            std_confidence=np.std(scores),
            entropy=self._calculate_entropy(labels),
            agreement=self._calculate_agreement(labels)
        )
```

### Active Learning Integration

```python
class ActiveLearner:
    """Select most informative samples for labeling"""

    def __init__(self, model: str, uncertainty_threshold: float = 0.3):
        self.model = model
        self.threshold = uncertainty_threshold
        self.labeled_data = []

    async def select_for_labeling(self, unlabeled: List[str], n: int) -> List[str]:
        # Get predictions with uncertainty
        results = await asyncio.gather(*[
            self._predict_with_uncertainty(text)
            for text in unlabeled
        ])

        # Sort by uncertainty (highest first)
        sorted_items = sorted(
            zip(unlabeled, results),
            key=lambda x: x[1].entropy,
            reverse=True
        )

        return [text for text, _ in sorted_items[:n]]
```

---

## 8. Performance Monitoring

### Inference Metrics

```python
class InferenceMetrics:
    def __init__(self):
        self.latencies = []
        self.token_counts = []
        self.costs = []

    def record(self, latency_ms: float, tokens: int, cost: float):
        self.latencies.append(latency_ms)
        self.token_counts.append(tokens)
        self.costs.append(cost)

    def summary(self) -> dict:
        return {
            "avg_latency_ms": np.mean(self.latencies),
            "p95_latency_ms": np.percentile(self.latencies, 95),
            "total_tokens": sum(self.token_counts),
            "total_cost_usd": sum(self.costs),
            "requests": len(self.latencies)
        }
```

### Quality Monitoring

```python
class QualityMonitor:
    def __init__(self, reference_model: str):
        self.reference_model = reference_model
        self.agreements = []

    async def check_quality(self, text: str, result: SentimentResult) -> bool:
        # Compare with reference model
        reference = await analyze_with_model(self.reference_model, text)
        agrees = result.label == reference.label

        self.agreements.append(agrees)
        return agrees

    def agreement_rate(self) -> float:
        return sum(self.agreements) / len(self.agreements) if self.agreements else 0
```

---

## Configuration

```yaml
inference:
  # Reasoning models
  reasoning:
    enabled: true
    default_model: "o1-mini"
    max_reasoning_tokens: 5000

  # Function calling
  tools:
    parallel_calls: true
    max_tools_per_call: 5
    timeout_per_tool: 30

  # Structured output
  structured:
    json_mode: true
    schema_validation: strict

  # Context management
  context:
    max_tokens: 128000
    chunking_strategy: "semantic"
    overlap_tokens: 200

  # Multi-model
  ensemble:
    enabled: false
    models: ["gpt-4o", "claude-3-5-sonnet"]
    aggregation: "weighted_vote"

  cascade:
    enabled: true
    confidence_threshold: 0.85

  # Caching
  cache:
    semantic_cache: true
    similarity_threshold: 0.95
    max_cache_size: 10000

  # Batching
  batch:
    enabled: true
    max_size: 32
    max_wait_ms: 100

  # Monitoring
  monitoring:
    track_latency: true
    track_costs: true
    quality_checks: true
```
