---
title: Adding Providers
description: How to add new LLM providers
---

# Adding Providers

Guide to adding new LLM providers to Sentimatrix.

## Provider Structure

```
sentimatrix/providers/llm/
├── base.py              # Base class
├── openai_provider.py   # Example provider
└── your_provider.py     # New provider
```

## Implement Base Class

```python
from sentimatrix.providers.llm.base import BaseLLMProvider
from sentimatrix.models import LLMResponse

class YourProvider(BaseLLMProvider):
    """Your LLM provider implementation."""

    @property
    def name(self) -> str:
        return "your_provider"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Implement text generation
        response = await self._call_api(prompt, **kwargs)
        return LLMResponse(
            content=response.text,
            model=self.model,
            provider=self.name,
            usage=TokenUsage(...),
            finish_reason=response.finish_reason,
            response_time_ms=elapsed
        )

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        # Implement chat completion
        pass

    async def stream(self, prompt: str, **kwargs):
        # Implement streaming
        async for chunk in self._stream_api(prompt, **kwargs):
            yield chunk

    async def health_check(self) -> bool:
        # Check provider availability
        try:
            await self._call_api("test", max_tokens=1)
            return True
        except Exception:
            return False
```

## Register Provider

Add to `LLMProvider` enum in `config.py`:

```python
class LLMProvider(str, Enum):
    YOUR_PROVIDER = "your_provider"
```

Register in provider manager:

```python
# In provider_manager.py
self._register_provider("your_provider", YourProvider)
```

## Add Tests

```python
# tests/providers/test_your_provider.py
import pytest
from sentimatrix.providers.llm.your_provider import YourProvider

@pytest.mark.asyncio
async def test_generate():
    provider = YourProvider(api_key="test")
    response = await provider.generate("Hello")
    assert response.content
    assert response.model

@pytest.mark.asyncio
async def test_health_check():
    provider = YourProvider(api_key="test")
    assert await provider.health_check()
```

## Add Documentation

Create `docs/providers/your-provider.md` with:

- Quick start
- Configuration
- Available models
- Pricing
- Examples

## Related

- [Base Provider API](../api/base-provider.md)
- [Adding Scrapers](scrapers.md)
- [Testing](testing.md)
