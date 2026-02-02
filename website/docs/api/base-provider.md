---
title: Base Provider
description: LLM provider base class reference
---

# Base Provider

Abstract base class for LLM providers.

## Class Definition

```python
class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Generate text completion."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion."""

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream text completion."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider availability."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether provider supports streaming."""
```

## Implementing a Provider

```python
class MyProvider(BaseLLMProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Implementation
        pass

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        # Implementation
        pass

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        # Implementation
        pass

    async def health_check(self) -> bool:
        # Implementation
        return True
```

## Related

- [API Overview](index.md)
- [Provider Manager](provider-manager.md)
- [Providers Guide](../providers/index.md)
