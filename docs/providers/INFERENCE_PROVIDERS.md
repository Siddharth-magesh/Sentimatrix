# Sentimatrix V2 - Specialized Inference Providers

## High-Speed Inference Providers

These providers specialize in extremely fast inference using custom hardware.

---

## 1. Groq

**Website:** https://groq.com

**Technology:** LPU (Language Processing Unit) - custom ASIC

**Speed:** ~750 tokens/second (Llama 3.1 70B)

**Models:**
| Model | Context | Input $/1M | Output $/1M |
|-------|---------|------------|-------------|
| llama-3.3-70b-versatile | 128K | $0.59 | $0.79 |
| llama-3.1-8b-instant | 128K | $0.05 | $0.08 |
| mixtral-8x7b-32768 | 32K | $0.24 | $0.24 |
| gemma2-9b-it | 8K | $0.20 | $0.20 |

**Free Tier:** Yes (rate limited)

**Features:**
- Fastest cloud inference
- Function calling
- JSON mode
- Streaming

**Implementation:**
```python
# Module: providers/llm/groq_provider.py
from groq import AsyncGroq

class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024)
        )
        return response.choices[0].message.content

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

**Configuration:**
```yaml
groq:
  api_key: "${GROQ_API_KEY}"
  model: "llama-3.3-70b-versatile"
  timeout: 30
```

**Rate Limits:**
- Free: 30 requests/min, 6,000 tokens/min
- Paid: Higher limits

---

## 2. Cerebras

**Website:** https://cerebras.ai

**Technology:** Wafer-Scale Engine (WSE-3)

**Speed:** ~1,800 tokens/second (Llama 3.1 70B)

**Models:**
| Model | Speed | Notes |
|-------|-------|-------|
| llama3.1-8b | 2,100 tok/s | Fastest small |
| llama3.1-70b | 1,800 tok/s | Fastest large |

**Features:**
- Fastest inference available
- OpenAI-compatible API
- Streaming

**Implementation:**
```python
# Module: providers/llm/cerebras_provider.py
from openai import AsyncOpenAI

class CerebrasProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "llama3.1-70b"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1"
        )
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

**Configuration:**
```yaml
cerebras:
  api_key: "${CEREBRAS_API_KEY}"
  model: "llama3.1-70b"
```

---

## 3. SambaNova

**Website:** https://sambanova.ai

**Technology:** Reconfigurable Dataflow Units (RDU)

**Speed:** ~580 tokens/second (Llama 3.1 405B)

**Models:**
| Model | Notes |
|-------|-------|
| llama-3.1-405b | Only provider with 405B |
| llama-3.1-70b | Fast |
| llama-3.1-8b | Fastest |

**Features:**
- Only cloud provider with Llama 405B
- OpenAI-compatible API

**Implementation:**
```python
# Module: providers/llm/sambanova_provider.py
from openai import AsyncOpenAI

class SambaNovaProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "Meta-Llama-3.1-405B-Instruct"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.sambanova.ai/v1"
        )
        self.model = model
```

---

## Open-Source Model Platforms

### 4. Together AI

**Website:** https://together.ai

**Models:** 200+ open-source models

**Notable Models:**
| Model | Context | Input $/1M | Output $/1M |
|-------|---------|------------|-------------|
| Llama-3.1-70B | 128K | $0.88 | $0.88 |
| Llama-3.1-8B | 128K | $0.18 | $0.18 |
| Qwen2.5-72B | 128K | $1.20 | $1.20 |
| DeepSeek-V3 | 128K | $0.50 | $0.50 |

**Features:**
- Wide model selection
- Fine-tuning
- Embeddings
- Function calling
- $25 free credits

**Implementation:**
```python
# Module: providers/llm/together_provider.py
from together import AsyncTogether

class TogetherProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.1-70B-Instruct-Turbo"):
        self.client = AsyncTogether(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

---

### 5. Fireworks AI

**Website:** https://fireworks.ai

**Models:** 100+ models with FireAttention optimization

**Pricing:**
| Model | Input $/1M | Output $/1M |
|-------|------------|-------------|
| Llama-3.1-70B | $0.90 | $0.90 |
| Llama-3.1-8B | $0.20 | $0.20 |
| Mixtral-8x22B | $1.20 | $1.20 |

**Features:**
- Fast inference (FireAttention)
- Function calling
- JSON mode
- Fine-tuning
- Serverless & dedicated

**Implementation:**
```python
# Module: providers/llm/fireworks_provider.py
from openai import AsyncOpenAI

class FireworksProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )
        self.model = model
```

---

### 6. Replicate

**Website:** https://replicate.com

**Pricing:** Per-second billing (varies by model)

**Features:**
- Run any model
- Custom model deployment
- Serverless
- No minimum

**Implementation:**
```python
# Module: providers/llm/replicate_provider.py
import replicate

class ReplicateProvider(BaseLLMProvider):
    def __init__(self, api_token: str):
        self.client = replicate.Client(api_token=api_token)

    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        output = await replicate.async_run(
            model,
            input={"prompt": prompt}
        )
        return "".join(output)
```

---

### 7. DeepSeek

**Website:** https://deepseek.com

**Models:**
| Model | Context | Input $/1M | Output $/1M |
|-------|---------|------------|-------------|
| deepseek-chat (V3) | 64K | $0.07 | $0.27 |
| deepseek-reasoner (R1) | 64K | $0.55 | $2.19 |

**Features:**
- Extremely cost-effective
- Strong reasoning (R1)
- MoE architecture
- Function calling

**Implementation:**
```python
# Module: providers/llm/deepseek_provider.py
from openai import AsyncOpenAI

class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model
```

**Configuration:**
```yaml
deepseek:
  api_key: "${DEEPSEEK_API_KEY}"
  model: "deepseek-chat"  # or deepseek-reasoner
```

---

## Speed Comparison

| Provider | Model | Tokens/sec | Notes |
|----------|-------|------------|-------|
| Cerebras | Llama 70B | 1,800 | Fastest |
| Groq | Llama 70B | 750 | Very fast |
| SambaNova | Llama 70B | 580 | Only 405B |
| Fireworks | Llama 70B | 200 | Good balance |
| Together | Llama 70B | 150 | Wide selection |
| OpenAI | GPT-4o | 100 | Baseline |

---

## Selection Guide

| Priority | Provider |
|----------|----------|
| Fastest inference | Cerebras |
| Best free tier | Groq |
| Largest model | SambaNova (405B) |
| Most models | Together AI |
| Cheapest | DeepSeek |
| Best reasoning | DeepSeek R1 |
