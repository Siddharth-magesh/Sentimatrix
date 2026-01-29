"""
Amazon Bedrock LLM Provider

Implements the BaseLLMProvider interface for Amazon Bedrock.
Provides access to multiple foundation models including Claude, Llama, Titan, and more.

Official API Documentation: https://docs.aws.amazon.com/bedrock/
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    AuthenticationError,
    ContentFilteredError,
    InvalidModelError,
    InvalidResponseError,
    RateLimitError,
    TokenLimitExceededError,
    ProviderError,
)
from sentimatrix.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderCapabilities,
    ProviderInfo,
    ProviderType,
    TokenUsage,
    register_provider,
)

# Lazy imports
_boto3 = None
_botocore = None


def _get_boto3():
    """Lazy import of boto3 module."""
    global _boto3
    if _boto3 is None:
        try:
            import boto3
            _boto3 = boto3
        except ImportError:
            raise ImportError(
                "boto3 package is required for Amazon Bedrock provider. "
                "Install it with: pip install boto3"
            )
    return _boto3


def _get_botocore():
    """Lazy import of botocore module."""
    global _botocore
    if _botocore is None:
        try:
            import botocore
            _botocore = botocore
        except ImportError:
            pass
    return _botocore


# Default model
DEFAULT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Anthropic Claude
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "context": 200000, "vision": True, "streaming": True
    },
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "context": 200000, "vision": True, "streaming": True
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "context": 200000, "vision": True, "streaming": True
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "context": 200000, "vision": True, "streaming": True
    },
    # Meta Llama
    "meta.llama3-2-90b-instruct-v1:0": {
        "context": 128000, "vision": True, "streaming": True
    },
    "meta.llama3-2-11b-instruct-v1:0": {
        "context": 128000, "vision": True, "streaming": True
    },
    "meta.llama3-1-405b-instruct-v1:0": {
        "context": 128000, "vision": False, "streaming": True
    },
    "meta.llama3-1-70b-instruct-v1:0": {
        "context": 128000, "vision": False, "streaming": True
    },
    "meta.llama3-1-8b-instruct-v1:0": {
        "context": 128000, "vision": False, "streaming": True
    },
    # Amazon Titan
    "amazon.titan-text-premier-v1:0": {
        "context": 32000, "vision": False, "streaming": True
    },
    "amazon.titan-text-express-v1": {
        "context": 8000, "vision": False, "streaming": True
    },
    "amazon.titan-text-lite-v1": {
        "context": 4000, "vision": False, "streaming": True
    },
    # Mistral
    "mistral.mistral-large-2407-v1:0": {
        "context": 128000, "vision": False, "streaming": True
    },
    "mistral.mixtral-8x7b-instruct-v0:1": {
        "context": 32000, "vision": False, "streaming": True
    },
    # Cohere
    "cohere.command-r-plus-v1:0": {
        "context": 128000, "vision": False, "streaming": True
    },
    "cohere.command-r-v1:0": {
        "context": 128000, "vision": False, "streaming": True
    },
}


class BedrockProvider(BaseLLMProvider):
    """
    Amazon Bedrock LLM Provider.

    Provides access to multiple foundation models through AWS Bedrock:
    - Anthropic Claude (3.5 Sonnet, 3 Sonnet, Haiku, Opus)
    - Meta Llama (3.2, 3.1)
    - Amazon Titan
    - Mistral
    - Cohere Command

    Supports:
    - Chat completions
    - Streaming responses
    - Vision (model-dependent)
    - Text embeddings (Titan, Cohere)

    Example:
        >>> config = LLMConfig(
        ...     provider="bedrock",
        ...     model="anthropic.claude-3-sonnet-20240229-v1:0",
        ... )
        >>> async with BedrockProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Bedrock provider.

        Args:
            config: LLM configuration with optional AWS settings.
                AWS credentials can be provided via:
                - config.api_key (access key) and config.api_secret (secret key)
                - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
                - AWS credentials file
                - IAM role (when running on AWS)
        """
        super().__init__(config)
        self._client: Any = None
        self._runtime_client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._region = getattr(config, 'region', None) or os.environ.get(
            "AWS_REGION", "us-east-1"
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model_config = MODEL_CONFIGS.get(self._model, {})
        context_limit = model_config.get("context", 128000)

        return ProviderInfo(
            name="bedrock",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="Amazon Bedrock - Multi-model foundation model service",
            capabilities=ProviderCapabilities(
                streaming=model_config.get("streaming", True),
                function_calling=False,  # Bedrock has limited function calling
                vision=model_config.get("vision", False),
                json_mode=False,
                embeddings=True,
                max_context_tokens=context_limit,
                max_output_tokens=min(4096, context_limit),
            ),
            supported_models=list(MODEL_CONFIGS.keys()),
            website="https://aws.amazon.com/bedrock/",
            documentation="https://docs.aws.amazon.com/bedrock/",
        )

    async def initialize(self) -> None:
        """Initialize the Bedrock client."""
        if self._initialized:
            return

        boto3 = _get_boto3()

        # Build session kwargs
        session_kwargs: Dict[str, Any] = {
            "region_name": self._region,
        }

        # Check for explicit credentials
        if self._config and self._config.api_key:
            session_kwargs["aws_access_key_id"] = self._config.api_key
            if hasattr(self._config, 'api_secret'):
                session_kwargs["aws_secret_access_key"] = self._config.api_secret

        try:
            session = boto3.Session(**session_kwargs)
            self._runtime_client = session.client("bedrock-runtime")
            self._client = session.client("bedrock")
            self._initialized = True
        except Exception as e:
            raise AuthenticationError(
                "bedrock",
                f"Failed to initialize Bedrock client: {e}"
            ) from e

    async def close(self) -> None:
        """Close the Bedrock client."""
        self._runtime_client = None
        self._client = None
        self._initialized = False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion using Amazon Bedrock.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Build request based on model provider
        if "anthropic" in self._model:
            body, content_extractor = self._build_claude_request(
                prompt, system_prompt, temperature, max_tokens, stop
            )
        elif "meta.llama" in self._model:
            body, content_extractor = self._build_llama_request(
                prompt, system_prompt, temperature, max_tokens, stop
            )
        elif "amazon.titan" in self._model:
            body, content_extractor = self._build_titan_request(
                prompt, system_prompt, temperature, max_tokens, stop
            )
        elif "mistral" in self._model:
            body, content_extractor = self._build_mistral_request(
                prompt, system_prompt, temperature, max_tokens, stop
            )
        elif "cohere" in self._model:
            body, content_extractor = self._build_cohere_request(
                prompt, system_prompt, temperature, max_tokens, stop
            )
        else:
            raise InvalidModelError(self._model, "bedrock")

        try:
            import asyncio
            loop = asyncio.get_event_loop()

            response = await loop.run_in_executor(
                None,
                lambda: self._runtime_client.invoke_model(
                    modelId=self._model,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            response_body = json.loads(response["body"].read())
            content, usage = content_extractor(response_body)

            return LLMResponse(
                content=content,
                model=self._model,
                provider="bedrock",
                usage=usage,
                finish_reason="stop",
                response_time_ms=elapsed_ms,
                raw_response=response_body,
            )

        except Exception as e:
            self._handle_error(e)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion using Amazon Bedrock."""
        self._ensure_initialized()

        # Build request based on model provider
        if "anthropic" in self._model:
            body, _ = self._build_claude_request(
                prompt, system_prompt, temperature, max_tokens, stop
            )
        elif "meta.llama" in self._model:
            body, _ = self._build_llama_request(
                prompt, system_prompt, temperature, max_tokens, stop
            )
        else:
            # Fall back to non-streaming for unsupported models
            response = await self.generate(
                prompt, system_prompt, temperature, max_tokens, stop, **kwargs
            )
            yield response.content
            return

        try:
            import asyncio
            loop = asyncio.get_event_loop()

            response = await loop.run_in_executor(
                None,
                lambda: self._runtime_client.invoke_model_with_response_stream(
                    modelId=self._model,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
            )

            stream = response.get("body")
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_data = json.loads(chunk.get("bytes").decode())
                        if "anthropic" in self._model:
                            if chunk_data.get("type") == "content_block_delta":
                                delta = chunk_data.get("delta", {})
                                if "text" in delta:
                                    yield delta["text"]
                        elif "meta.llama" in self._model:
                            if "generation" in chunk_data:
                                yield chunk_data["generation"]

        except Exception as e:
            self._handle_error(e)

    def _build_claude_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop: Optional[List[str]],
    ) -> tuple:
        """Build request body for Claude models."""
        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tok,
            "temperature": temp,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            body["system"] = system_prompt

        if stop:
            body["stop_sequences"] = stop

        def extract(response: Dict) -> tuple:
            content = response.get("content", [{}])[0].get("text", "")
            usage = TokenUsage(
                prompt_tokens=response.get("usage", {}).get("input_tokens", 0),
                completion_tokens=response.get("usage", {}).get("output_tokens", 0),
                total_tokens=0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            return content, usage

        return body, extract

    def _build_llama_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop: Optional[List[str]],
    ) -> tuple:
        """Build request body for Llama models."""
        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        body = {
            "prompt": full_prompt,
            "max_gen_len": max_tok,
            "temperature": temp,
        }

        def extract(response: Dict) -> tuple:
            content = response.get("generation", "")
            usage = TokenUsage(
                prompt_tokens=response.get("prompt_token_count", 0),
                completion_tokens=response.get("generation_token_count", 0),
                total_tokens=0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            return content, usage

        return body, extract

    def _build_titan_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop: Optional[List[str]],
    ) -> tuple:
        """Build request body for Amazon Titan models."""
        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        body = {
            "inputText": full_prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tok,
                "temperature": temp,
                "topP": 0.9,
            }
        }

        if stop:
            body["textGenerationConfig"]["stopSequences"] = stop

        def extract(response: Dict) -> tuple:
            results = response.get("results", [{}])
            content = results[0].get("outputText", "") if results else ""
            usage = TokenUsage(
                prompt_tokens=response.get("inputTextTokenCount", 0),
                completion_tokens=results[0].get("tokenCount", 0) if results else 0,
                total_tokens=0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            return content, usage

        return body, extract

    def _build_mistral_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop: Optional[List[str]],
    ) -> tuple:
        """Build request body for Mistral models."""
        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        full_prompt = f"<s>[INST] {prompt} [/INST]"
        if system_prompt:
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"

        body = {
            "prompt": full_prompt,
            "max_tokens": max_tok,
            "temperature": temp,
        }

        if stop:
            body["stop"] = stop

        def extract(response: Dict) -> tuple:
            outputs = response.get("outputs", [{}])
            content = outputs[0].get("text", "") if outputs else ""
            usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            return content, usage

        return body, extract

    def _build_cohere_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stop: Optional[List[str]],
    ) -> tuple:
        """Build request body for Cohere models."""
        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        body = {
            "message": prompt,
            "max_tokens": max_tok,
            "temperature": temp,
        }

        if system_prompt:
            body["preamble"] = system_prompt

        if stop:
            body["stop_sequences"] = stop

        def extract(response: Dict) -> tuple:
            content = response.get("text", "")
            usage = TokenUsage(
                prompt_tokens=response.get("meta", {}).get("tokens", {}).get("input_tokens", 0),
                completion_tokens=response.get("meta", {}).get("tokens", {}).get("output_tokens", 0),
                total_tokens=0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            return content, usage

        return body, extract

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings using Bedrock.

        Args:
            texts: List of texts to embed
            model: Embedding model ID (Titan or Cohere)

        Returns:
            List of embedding vectors
        """
        self._ensure_initialized()

        embed_model = model or "amazon.titan-embed-text-v2:0"

        embeddings = []
        for text in texts:
            try:
                import asyncio
                loop = asyncio.get_event_loop()

                if "titan" in embed_model:
                    body = {"inputText": text}
                else:  # cohere
                    body = {"texts": [text], "input_type": "search_document"}

                response = await loop.run_in_executor(
                    None,
                    lambda: self._runtime_client.invoke_model(
                        modelId=embed_model,
                        body=json.dumps(body),
                        contentType="application/json",
                        accept="application/json",
                    )
                )

                response_body = json.loads(response["body"].read())

                if "titan" in embed_model:
                    embeddings.append(response_body.get("embedding", []))
                else:
                    embs = response_body.get("embeddings", [[]])
                    embeddings.append(embs[0] if embs else [])

            except Exception as e:
                raise ProviderError("bedrock", f"Embedding failed: {e}") from e

        return embeddings

    def _handle_error(self, error: Exception) -> None:
        """Handle Bedrock API errors."""
        botocore = _get_botocore()
        error_msg = str(error)

        if botocore:
            if isinstance(error, botocore.exceptions.ClientError):
                error_code = error.response.get("Error", {}).get("Code", "")
                if error_code == "AccessDeniedException":
                    raise AuthenticationError("bedrock", error_msg) from error
                elif error_code == "ThrottlingException":
                    raise RateLimitError("bedrock", retry_after=60) from error
                elif error_code == "ValidationException":
                    if "token" in error_msg.lower():
                        raise TokenLimitExceededError("bedrock", 0, 0) from error
                    raise InvalidResponseError("bedrock", error_msg) from error
                elif error_code == "ResourceNotFoundException":
                    raise InvalidModelError(self._model, "bedrock") from error

        raise ProviderError("bedrock", error_msg) from error


# Register the provider
register_provider("bedrock", ProviderType.LLM, BedrockProvider)
