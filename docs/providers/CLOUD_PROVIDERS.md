# Sentimatrix V2 - Cloud LLM Providers

## 1. OpenAI

**Website:** https://openai.com

**Models:**
| Model | Context | Input $/1M | Output $/1M | Notes |
|-------|---------|------------|-------------|-------|
| gpt-4o | 128K | $2.50 | $10.00 | Flagship |
| gpt-4o-mini | 128K | $0.15 | $0.60 | Cost-effective |
| gpt-4-turbo | 128K | $10.00 | $30.00 | Legacy |
| o1-preview | 128K | $15.00 | $60.00 | Reasoning |
| o1-mini | 128K | $3.00 | $12.00 | Reasoning (faster) |

**Features:**
- Function calling
- JSON mode
- Vision (gpt-4o, gpt-4-turbo)
- Streaming
- Fine-tuning
- Batch API

**Implementation:**
```python
# Module: providers/llm/openai_provider.py
from openai import AsyncOpenAI

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
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
openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  organization: null
  base_url: null  # For Azure or proxies
  timeout: 30
  max_retries: 3
```

---

## 2. Anthropic (Claude)

**Website:** https://anthropic.com

**Models:**
| Model | Context | Input $/1M | Output $/1M | Notes |
|-------|---------|------------|-------------|-------|
| claude-3-5-sonnet | 200K | $3.00 | $15.00 | Best balance |
| claude-3-opus | 200K | $15.00 | $75.00 | Most capable |
| claude-3-haiku | 200K | $0.25 | $1.25 | Fastest |
| claude-3-5-haiku | 200K | $1.00 | $5.00 | Fast + capable |

**Features:**
- Tool use (function calling)
- Vision
- Long context (200K)
- Streaming
- System prompts
- Extended thinking (beta)

**Implementation:**
```python
# Module: providers/llm/anthropic_provider.py
from anthropic import AsyncAnthropic

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
            system=kwargs.get("system_prompt")
        )
        return response.content[0].text
```

**Configuration:**
```yaml
anthropic:
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-3-5-sonnet-20241022"
  max_tokens: 1024
  timeout: 60
```

---

## 3. Google (Gemini)

**Website:** https://ai.google.dev

**Models:**
| Model | Context | Input $/1M | Output $/1M | Notes |
|-------|---------|------------|-------------|-------|
| gemini-2.0-flash | 1M | $0.10 | $0.40 | Latest |
| gemini-1.5-pro | 2M | $1.25 | $5.00 | Long context |
| gemini-1.5-flash | 1M | $0.075 | $0.30 | Fast |
| gemini-1.5-flash-8b | 1M | $0.0375 | $0.15 | Cheapest |

**Features:**
- Very long context (up to 2M)
- Multi-modal (text, image, audio, video)
- Function calling
- Code execution
- Grounding with Google Search

**Implementation:**
```python
# Module: providers/llm/gemini_provider.py
import google.generativeai as genai

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.model.generate_content_async(prompt)
        return response.text
```

**Configuration:**
```yaml
gemini:
  api_key: "${GOOGLE_API_KEY}"
  model: "gemini-1.5-flash"
  safety_settings: "default"
```

---

## 4. AWS Bedrock

**Website:** https://aws.amazon.com/bedrock

**Available Models:**
- Anthropic Claude (all versions)
- Meta Llama 3
- Mistral
- Amazon Titan
- Cohere Command
- AI21 Jurassic

**Features:**
- Enterprise security (VPC, IAM)
- No data retention
- Unified billing
- Knowledge bases
- Agents

**Implementation:**
```python
# Module: providers/llm/bedrock_provider.py
import boto3

class BedrockProvider(BaseLLMProvider):
    def __init__(self, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        self.client = boto3.client("bedrock-runtime")
        self.model_id = model_id

    async def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1024)
            })
        )
        return json.loads(response["body"].read())["content"][0]["text"]
```

**Configuration:**
```yaml
bedrock:
  region: "us-east-1"
  model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
  # AWS credentials via environment or IAM role
```

---

## 5. Azure OpenAI

**Website:** https://azure.microsoft.com/en-us/products/ai-services/openai-service

**Features:**
- Enterprise compliance
- Regional deployment
- Private endpoints
- Content filtering
- Same models as OpenAI

**Implementation:**
```python
# Module: providers/llm/azure_openai_provider.py
from openai import AsyncAzureOpenAI

class AzureOpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, endpoint: str, deployment: str):
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint
        )
        self.deployment = deployment

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

**Configuration:**
```yaml
azure_openai:
  api_key: "${AZURE_OPENAI_KEY}"
  endpoint: "${AZURE_OPENAI_ENDPOINT}"
  deployment: "gpt-4o"
  api_version: "2024-02-01"
```

---

## 6. Mistral AI

**Website:** https://mistral.ai

**Models:**
| Model | Context | Input $/1M | Output $/1M |
|-------|---------|------------|-------------|
| mistral-large | 128K | $2.00 | $6.00 |
| mistral-small | 128K | $0.20 | $0.60 |
| codestral | 32K | $0.20 | $0.60 |
| mistral-embed | 8K | $0.10 | - |

**Features:**
- Function calling
- JSON mode
- Code generation (Codestral)
- Embeddings
- EU-based

**Implementation:**
```python
# Module: providers/llm/mistral_provider.py
from mistralai import Mistral

class MistralProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "mistral-small-latest"):
        self.client = Mistral(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.complete_async(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

---

## 7. Cohere

**Website:** https://cohere.com

**Models:**
| Model | Use Case | Notes |
|-------|----------|-------|
| command-r-plus | RAG | Best for retrieval |
| command-r | Chat | Fast |
| embed-v3 | Embeddings | Multilingual |
| rerank-v3 | Reranking | Search |

**Features:**
- RAG-optimized
- Reranking
- Multi-lingual embeddings
- Web connectors

**Implementation:**
```python
# Module: providers/llm/cohere_provider.py
import cohere

class CohereProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "command-r"):
        self.client = cohere.AsyncClient(api_key)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat(
            model=self.model,
            message=prompt
        )
        return response.text

    async def embed(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings
```
