# Sentimatrix V2 - LLM Models Overview

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM INTERACTION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │
│  │    INPUT      │  │   REASONING   │  │    OUTPUT     │                   │
│  │   HANDLERS    │  │    ENGINE     │  │   HANDLERS    │                   │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘                   │
│          │                  │                  │                            │
│          ▼                  ▼                  ▼                            │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    PROMPT ORCHESTRATOR                           │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │       │
│  │  │ Template │ │ Chain of │ │  ReAct   │ │  Agent   │           │       │
│  │  │ Manager  │ │ Thought  │ │  Engine  │ │ Executor │           │       │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    PROVIDER ABSTRACTION                          │       │
│  │  OpenAI │ Anthropic │ Groq │ Gemini │ Ollama │ ...              │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. Dynamic Model Selection

Sentimatrix V2 dynamically selects the appropriate model and reasoning strategy based on:

- Task complexity
- Input type and length
- Required output format
- Cost constraints
- Latency requirements
- Accuracy requirements

```python
# Automatic model selection
result = await sm.analyze(
    input_data,
    strategy="auto",  # Automatically choose best approach
    constraints={
        "max_latency_ms": 2000,
        "max_cost_usd": 0.01,
        "min_accuracy": 0.9
    }
)
```

### 2. Reasoning Strategies

| Strategy | Use Case | Complexity | Accuracy |
|----------|----------|------------|----------|
| Direct | Simple tasks | Low | Medium |
| Chain of Thought | Complex reasoning | Medium | High |
| ReAct | Multi-step with tools | High | Very High |
| Tree of Thoughts | Exploration tasks | Very High | Very High |
| Self-Consistency | Critical decisions | High | Very High |

### 3. Input Modalities

| Modality | Supported Formats | Processing |
|----------|-------------------|------------|
| Text | String, List[String] | Direct |
| Structured | JSON, Dict | Serialization |
| Audio | WAV, MP3, FLAC | Transcription first |
| Image | PNG, JPG, WEBP | Vision model or captioning |
| Video | MP4, AVI | Frame extraction + audio |
| URL | HTTP/HTTPS | Scraping first |
| File | CSV, JSON, TXT | Parsing first |

### 4. Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| Text | Free-form response | Summaries |
| JSON | Structured data | API responses |
| Structured | Pydantic models | Type-safe outputs |
| Streaming | Chunk-by-chunk | Real-time UI |
| Tool Calls | Function invocations | Agents |

---

## Model Capabilities Matrix

| Capability | OpenAI | Anthropic | Groq | Gemini | Ollama |
|------------|--------|-----------|------|--------|--------|
| Text Generation | Yes | Yes | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes | Yes | Yes |
| Function Calling | Yes | Yes | Yes | Yes | Partial |
| JSON Mode | Yes | Yes | Yes | Yes | Yes |
| Vision | Yes | Yes | No | Yes | Yes* |
| Long Context | 128K | 200K | 128K | 2M | Model-dependent |
| Reasoning (o1) | Yes | No | No | No | No |

*Requires vision-capable model like LLaVA

---

## Task-to-Strategy Mapping

| Task | Recommended Strategy | Fallback |
|------|---------------------|----------|
| Simple sentiment | Direct | - |
| Review summarization | Chain of Thought | Direct |
| Aspect extraction | ReAct with tools | CoT |
| Product comparison | ReAct + Self-Consistency | CoT |
| Root cause analysis | Tree of Thoughts | ReAct |
| Multi-source aggregation | Agentic workflow | ReAct |

---

## Configuration

```yaml
models:
  # Default reasoning strategy
  default_strategy: "auto"

  # Strategy configurations
  strategies:
    direct:
      temperature: 0.3
      max_tokens: 1024

    chain_of_thought:
      temperature: 0.5
      max_tokens: 2048
      thinking_tokens: 1024

    react:
      temperature: 0.7
      max_iterations: 10
      max_tokens: 4096
      tools_enabled: true

    tree_of_thoughts:
      branches: 3
      depth: 3
      evaluation_model: "same"

    self_consistency:
      samples: 5
      temperature: 0.8
      aggregation: "majority_vote"

  # Dynamic selection rules
  auto_selection:
    simple_tasks:
      max_input_tokens: 500
      strategy: "direct"
    complex_tasks:
      min_input_tokens: 500
      strategy: "chain_of_thought"
    tool_tasks:
      requires_tools: true
      strategy: "react"
```

---

## Integration Points

### With Sentiment Analysis
```python
# Direct sentiment
result = await sm.analyze_sentiment(text, strategy="direct")

# CoT sentiment with reasoning
result = await sm.analyze_sentiment(
    text,
    strategy="chain_of_thought",
    return_reasoning=True
)
```

### With Scraping
```python
# Agentic scraping with dynamic adaptation
result = await sm.scrape_and_analyze(
    url,
    strategy="react",
    tools=["scraper", "sentiment", "summarizer"]
)
```

### With Multi-Modal
```python
# Vision + reasoning
result = await sm.analyze_image(
    image_path,
    strategy="chain_of_thought",
    prompt="Analyze the sentiment expressed in this image"
)
```

---

## Performance Considerations

| Strategy | Latency | Cost | Accuracy |
|----------|---------|------|----------|
| Direct | Low | Low | Medium |
| CoT | Medium | Medium | High |
| ReAct | High | High | Very High |
| ToT | Very High | Very High | Very High |
| Self-Consistency | High | High (5x) | Very High |

---

## Related Documentation

- [Prompting Strategies](./PROMPTING_STRATEGIES.md)
- [Agentic Patterns](./AGENTIC_PATTERNS.md)
- [Input/Output Formats](./INPUT_OUTPUT.md)
- [Advanced Inference](./ADVANCED_INFERENCE.md)
- [Reasoning Models](./REASONING_MODELS.md)
