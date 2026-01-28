"""
LLM Provider Manager

Manages multiple LLM providers with support for:
- Automatic fallback on errors
- Load balancing
- Provider health checking
- Rate limit handling
- Unified interface for all providers

This is the main entry point for LLM operations in Sentimatrix.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Type, Union

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    LLMProviderError,
    ProviderNotFoundError,
    RateLimitError,
)
from sentimatrix.core.logger import get_logger
from sentimatrix.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderRegistry,
    ProviderType,
    TokenUsage,
)

logger = get_logger(__name__)


class FallbackStrategy(str, Enum):
    """Strategy for provider fallback."""

    SEQUENTIAL = "sequential"  # Try providers in order
    ROUND_ROBIN = "round_robin"  # Distribute load
    FASTEST = "fastest"  # Use fastest responding provider
    CHEAPEST = "cheapest"  # Use cheapest provider first


@dataclass
class ProviderHealth:
    """Health status of a provider."""

    provider_name: str
    is_healthy: bool = True
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    avg_response_time_ms: float = 0.0
    rate_limited_until: Optional[float] = None

    def record_success(self, response_time_ms: float) -> None:
        """Record a successful request."""
        self.is_healthy = True
        self.consecutive_failures = 0
        self.total_requests += 1

        # Update average response time
        if self.avg_response_time_ms == 0:
            self.avg_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            self.avg_response_time_ms = (
                0.9 * self.avg_response_time_ms + 0.1 * response_time_ms
            )

    def record_failure(self, error: str, is_rate_limit: bool = False) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_error = error
        self.last_error_time = time.time()

        # Mark unhealthy after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.is_healthy = False

        # Handle rate limiting
        if is_rate_limit:
            self.rate_limited_until = time.time() + 60  # 60 second cooldown

    def is_available(self) -> bool:
        """Check if provider is currently available."""
        if not self.is_healthy:
            return False

        if self.rate_limited_until:
            if time.time() < self.rate_limited_until:
                return False
            # Reset rate limit
            self.rate_limited_until = None

        return True


@dataclass
class ProviderConfig:
    """Configuration for a single provider in the chain."""

    name: str
    config: LLMConfig
    priority: int = 0  # Lower = higher priority
    weight: float = 1.0  # For load balancing
    max_retries: int = 2
    enabled: bool = True


@dataclass
class LLMManagerConfig:
    """Configuration for the LLM Manager."""

    providers: List[ProviderConfig] = field(default_factory=list)
    fallback_strategy: FallbackStrategy = FallbackStrategy.SEQUENTIAL
    enable_health_checks: bool = True
    health_check_interval: int = 60  # seconds
    max_total_retries: int = 5


class LLMProviderManager:
    """
    Manager for multiple LLM providers with fallback support.

    Features:
    - Automatic fallback on provider errors
    - Rate limit handling with backoff
    - Provider health tracking
    - Load balancing (round-robin, weighted)
    - Unified interface for all providers

    Example:
        >>> manager = LLMProviderManager()
        >>> manager.add_provider("openai", LLMConfig(
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     model="gpt-4o-mini"
        ... ))
        >>> manager.add_provider("groq", LLMConfig(
        ...     provider="groq",
        ...     api_key="gsk_...",
        ...     model="llama-3.3-70b-versatile"
        ... ))
        >>> async with manager:
        ...     response = await manager.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(
        self,
        config: Optional[LLMManagerConfig] = None,
    ) -> None:
        """
        Initialize the LLM Manager.

        Args:
            config: Manager configuration
        """
        self._config = config or LLMManagerConfig()
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._provider_configs: Dict[str, ProviderConfig] = {}
        self._health: Dict[str, ProviderHealth] = {}
        self._round_robin_index: int = 0
        self._initialized: bool = False
        self._registry = ProviderRegistry()

    def add_provider(
        self,
        name: str,
        config: LLMConfig,
        priority: int = 0,
        weight: float = 1.0,
        max_retries: int = 2,
    ) -> None:
        """
        Add a provider to the manager.

        Args:
            name: Unique name for this provider instance
            config: LLM configuration
            priority: Priority (lower = tried first)
            weight: Weight for load balancing
            max_retries: Max retries for this provider
        """
        provider_config = ProviderConfig(
            name=name,
            config=config,
            priority=priority,
            weight=weight,
            max_retries=max_retries,
        )
        self._provider_configs[name] = provider_config
        self._health[name] = ProviderHealth(provider_name=name)

    def remove_provider(self, name: str) -> None:
        """Remove a provider from the manager."""
        if name in self._providers:
            del self._providers[name]
        if name in self._provider_configs:
            del self._provider_configs[name]
        if name in self._health:
            del self._health[name]

    async def initialize(self) -> None:
        """Initialize all providers."""
        if self._initialized:
            return

        for name, pc in self._provider_configs.items():
            if not pc.enabled:
                continue

            try:
                # Get provider class from registry
                provider_type = pc.config.provider.value if hasattr(pc.config.provider, 'value') else str(pc.config.provider)
                provider = self._registry.get(provider_type, ProviderType.LLM, pc.config)
                await provider.initialize()
                self._providers[name] = provider
                logger.info(f"Initialized provider: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize provider {name}: {e}")
                self._health[name].record_failure(str(e))

        self._initialized = True

    async def close(self) -> None:
        """Close all providers."""
        for name, provider in self._providers.items():
            try:
                await provider.close()
                logger.debug(f"Closed provider: {name}")
            except Exception as e:
                logger.warning(f"Error closing provider {name}: {e}")

        self._providers.clear()
        self._initialized = False

    async def __aenter__(self) -> "LLMProviderManager":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _get_ordered_providers(self) -> List[str]:
        """Get providers in order based on strategy."""
        available = [
            name for name, health in self._health.items()
            if health.is_available() and name in self._providers
        ]

        if not available:
            # If no healthy providers, try all
            available = list(self._providers.keys())

        strategy = self._config.fallback_strategy

        if strategy == FallbackStrategy.SEQUENTIAL:
            # Sort by priority
            return sorted(
                available,
                key=lambda n: self._provider_configs.get(n, ProviderConfig(name=n, config=LLMConfig())).priority
            )

        elif strategy == FallbackStrategy.ROUND_ROBIN:
            # Rotate through providers
            if available:
                self._round_robin_index = (self._round_robin_index + 1) % len(available)
                return available[self._round_robin_index:] + available[:self._round_robin_index]
            return available

        elif strategy == FallbackStrategy.FASTEST:
            # Sort by average response time
            return sorted(
                available,
                key=lambda n: self._health.get(n, ProviderHealth(provider_name=n)).avg_response_time_ms
            )

        elif strategy == FallbackStrategy.CHEAPEST:
            # Priority-based (assumes lower priority = cheaper)
            return sorted(
                available,
                key=lambda n: self._provider_configs.get(n, ProviderConfig(name=n, config=LLMConfig())).priority
            )

        return available

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion using available providers.

        Automatically falls back to other providers on failure.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            provider_name: Specific provider to use (skips fallback)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content

        Raises:
            LLMProviderError: If all providers fail
        """
        if not self._initialized:
            await self.initialize()

        # If specific provider requested
        if provider_name:
            if provider_name not in self._providers:
                raise ProviderNotFoundError(provider_name)
            return await self._generate_with_provider(
                provider_name, prompt, system_prompt, temperature, max_tokens, stop, **kwargs
            )

        # Try providers in order
        providers = self._get_ordered_providers()
        last_error: Optional[Exception] = None
        total_retries = 0

        for provider_name in providers:
            if total_retries >= self._config.max_total_retries:
                break

            pc = self._provider_configs.get(provider_name)
            max_retries = pc.max_retries if pc else 2

            for retry in range(max_retries):
                total_retries += 1
                try:
                    response = await self._generate_with_provider(
                        provider_name, prompt, system_prompt, temperature, max_tokens, stop, **kwargs
                    )

                    # Record success
                    self._health[provider_name].record_success(response.response_time_ms)
                    return response

                except RateLimitError as e:
                    logger.warning(f"Rate limit for {provider_name}: {e}")
                    self._health[provider_name].record_failure(str(e), is_rate_limit=True)
                    last_error = e
                    break  # Don't retry rate limits on same provider

                except LLMProviderError as e:
                    logger.warning(f"Provider {provider_name} error (attempt {retry + 1}): {e}")
                    self._health[provider_name].record_failure(str(e))
                    last_error = e

                    if retry < max_retries - 1:
                        # Exponential backoff
                        await asyncio.sleep(0.5 * (2 ** retry))

                except Exception as e:
                    logger.error(f"Unexpected error from {provider_name}: {e}")
                    self._health[provider_name].record_failure(str(e))
                    last_error = e
                    break  # Don't retry unexpected errors

        # All providers failed
        raise LLMProviderError(
            f"All providers failed. Last error: {last_error}",
            provider="manager",
            original_error=last_error if isinstance(last_error, Exception) else None,
        )

    async def _generate_with_provider(
        self,
        provider_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate with a specific provider."""
        provider = self._providers[provider_name]
        return await provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion using available providers.

        Note: Streaming doesn't support automatic fallback mid-stream.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            provider_name: Specific provider to use
            **kwargs: Additional parameters

        Yields:
            Text chunks as they're generated
        """
        if not self._initialized:
            await self.initialize()

        # Select provider
        if provider_name:
            if provider_name not in self._providers:
                raise ProviderNotFoundError(provider_name)
        else:
            providers = self._get_ordered_providers()
            if not providers:
                raise LLMProviderError("No providers available", provider="manager")
            provider_name = providers[0]

        provider = self._providers[provider_name]

        try:
            async for chunk in provider.generate_stream(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            ):
                yield chunk

        except Exception as e:
            self._health[provider_name].record_failure(str(e))
            raise

    async def generate_with_functions(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate with function calling support.

        Args:
            prompt: User message
            functions: List of function definitions
            system_prompt: Optional system message
            function_call: How to handle function calls
            provider_name: Specific provider to use
            **kwargs: Additional parameters

        Returns:
            LLMResponse with potential tool_calls
        """
        if not self._initialized:
            await self.initialize()

        # Filter to providers that support function calling
        if provider_name:
            if provider_name not in self._providers:
                raise ProviderNotFoundError(provider_name)
            providers = [provider_name]
        else:
            providers = [
                name for name in self._get_ordered_providers()
                if self._providers[name].supports_function_calling
            ]

        if not providers:
            raise LLMProviderError(
                "No providers available with function calling support",
                provider="manager",
            )

        last_error: Optional[Exception] = None

        for pname in providers:
            try:
                response = await self._providers[pname].generate_with_functions(
                    prompt=prompt,
                    functions=functions,
                    system_prompt=system_prompt,
                    function_call=function_call,
                    **kwargs,
                )
                self._health[pname].record_success(response.response_time_ms)
                return response

            except Exception as e:
                logger.warning(f"Function calling failed for {pname}: {e}")
                self._health[pname].record_failure(str(e))
                last_error = e

        raise LLMProviderError(
            f"All providers failed for function calling. Last error: {last_error}",
            provider="manager",
            original_error=last_error if isinstance(last_error, Exception) else None,
        )

    async def embed(
        self,
        text: Union[str, List[str]],
        provider_name: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings.

        Args:
            text: Single text or list of texts
            provider_name: Specific provider to use

        Returns:
            Embedding vector(s)
        """
        if not self._initialized:
            await self.initialize()

        # Filter to providers that support embeddings
        if provider_name:
            if provider_name not in self._providers:
                raise ProviderNotFoundError(provider_name)
            providers = [provider_name]
        else:
            providers = [
                name for name in self._get_ordered_providers()
                if self._providers[name].info.capabilities.embeddings
            ]

        if not providers:
            raise LLMProviderError(
                "No providers available with embedding support",
                provider="manager",
            )

        for pname in providers:
            try:
                return await self._providers[pname].embed(text)
            except Exception as e:
                logger.warning(f"Embedding failed for {pname}: {e}")
                self._health[pname].record_failure(str(e))

        raise LLMProviderError(
            "All providers failed for embeddings",
            provider="manager",
        )

    def get_health_status(self) -> Dict[str, ProviderHealth]:
        """Get health status of all providers."""
        return dict(self._health)

    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        """Get a specific provider instance."""
        return self._providers.get(name)

    def list_providers(self) -> List[str]:
        """List all configured provider names."""
        return list(self._provider_configs.keys())

    def list_available_providers(self) -> List[str]:
        """List currently available (healthy) providers."""
        return [
            name for name, health in self._health.items()
            if health.is_available() and name in self._providers
        ]


# Convenience function to create a manager from dict config
def create_manager_from_config(config: Dict[str, Any]) -> LLMProviderManager:
    """
    Create an LLM Manager from a configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured LLMProviderManager

    Example:
        >>> config = {
        ...     "providers": [
        ...         {"name": "openai", "api_key": "sk-...", "model": "gpt-4o-mini"},
        ...         {"name": "groq", "api_key": "gsk_...", "model": "llama-3.3-70b"},
        ...     ],
        ...     "fallback_strategy": "sequential"
        ... }
        >>> manager = create_manager_from_config(config)
    """
    manager = LLMProviderManager()

    providers = config.get("providers", [])
    for i, p in enumerate(providers):
        name = p.get("name", p.get("provider", f"provider_{i}"))
        provider_type = p.get("provider", name)

        llm_config = LLMConfig(
            provider=provider_type,
            model=p.get("model", ""),
            api_key=p.get("api_key"),
            api_base=p.get("api_base"),
            timeout=p.get("timeout", 30),
            max_tokens=p.get("max_tokens", 1024),
            temperature=p.get("temperature", 0.7),
        )

        manager.add_provider(
            name=name,
            config=llm_config,
            priority=p.get("priority", i),
            weight=p.get("weight", 1.0),
            max_retries=p.get("max_retries", 2),
        )

    # Set fallback strategy
    strategy = config.get("fallback_strategy", "sequential")
    manager._config.fallback_strategy = FallbackStrategy(strategy)

    return manager
