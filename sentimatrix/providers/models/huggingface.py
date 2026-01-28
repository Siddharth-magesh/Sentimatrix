"""
Sentimatrix HuggingFace Model Provider

Provides integration with HuggingFace Transformers for sentiment analysis
and emotion detection models. Supports automatic device detection, batch
processing, model caching, and comprehensive error handling.

Supported models:
- Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest (default)
- Emotion: SamLowe/roberta-base-go_emotions (default)

Example:
    >>> provider = HuggingFaceModelProvider(config)
    >>> await provider.initialize()
    >>> result = await provider.predict("I love this product!")
    >>> print(result.label, result.score)
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from sentimatrix.core.config import ModelConfig
from sentimatrix.core.exceptions import (
    DeviceError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotFoundError,
)
from sentimatrix.providers.base import (
    BaseModelProvider,
    PredictionResult,
    ProviderCapabilities,
    ProviderInfo,
    ProviderType,
    register_provider,
)


class ModelType(str, Enum):
    """Types of models supported."""

    SENTIMENT = "sentiment"
    EMOTION = "emotion"
    NER = "ner"
    CLASSIFICATION = "classification"


class DeviceType(str, Enum):
    """Device types for model inference."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    name: str
    model_type: ModelType
    device: str
    num_labels: int
    labels: List[str]
    max_length: int
    loaded_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "device": self.device,
            "num_labels": self.num_labels,
            "labels": self.labels,
            "max_length": self.max_length,
            "loaded_at": self.loaded_at,
        }


# Global model cache to avoid reloading
_model_cache: Dict[str, Tuple[Any, Any, ModelInfo]] = {}
_cache_lock = asyncio.Lock()


def _detect_device(device: str = "auto") -> str:
    """
    Detect the best available device for inference.

    Args:
        device: Requested device ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Device string for torch

    Raises:
        DeviceError: If requested device is not available
    """
    try:
        import torch
    except ImportError:
        # If torch is not available, default to CPU
        return "cpu"

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    if device == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError(
                model_name="",
                device=device,
                reason="CUDA is not available. Install PyTorch with CUDA support.",
            )
        return "cuda"

    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise DeviceError(
                model_name="",
                device=device,
                reason="MPS is not available. Requires macOS 12.3+ with Apple Silicon.",
            )
        return "mps"

    return "cpu"


def _get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {"cpu": True, "cuda": False, "mps": False, "cuda_device_count": 0}

    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count() if info["cuda"] else 0
        info["mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        pass

    return info


class HuggingFaceModelProvider(BaseModelProvider):
    """
    HuggingFace Transformers model provider.

    Provides sentiment analysis and emotion detection using HuggingFace
    models with support for:
    - Automatic device detection (CPU/CUDA/MPS)
    - Batch processing for efficiency
    - Model caching to avoid reloading
    - Comprehensive error handling

    Attributes:
        model_name: HuggingFace model identifier
        model_type: Type of model (sentiment, emotion, etc.)
        device: Device for inference
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
        model_type: ModelType = ModelType.SENTIMENT,
    ) -> None:
        """
        Initialize HuggingFace model provider.

        Args:
            config: Model configuration
            model_name: Override model name from config
            model_type: Type of model to load
        """
        super().__init__(config)
        self._config: ModelConfig = config or ModelConfig()
        self._model_type = model_type

        # Determine model name based on type
        if model_name:
            self._model_name = model_name
        elif model_type == ModelType.SENTIMENT:
            self._model_name = self._config.sentiment_model
        elif model_type == ModelType.EMOTION:
            self._model_name = self._config.emotion_model
        else:
            self._model_name = self._config.sentiment_model

        self._device: str = ""
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_info: Optional[ModelInfo] = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="huggingface",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="HuggingFace Transformers model provider for sentiment and emotion analysis",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "SamLowe/roberta-base-go_emotions",
                "nlptown/bert-base-multilingual-uncased-sentiment",
                "j-hartmann/emotion-english-distilroberta-base",
                "finiteautomata/bertweet-base-sentiment-analysis",
            ],
        )

    async def initialize(self) -> None:
        """
        Initialize the model provider.

        Loads the model and tokenizer, handles device placement,
        and validates the model configuration.

        Raises:
            ModelLoadError: If model cannot be loaded
            DeviceError: If device is not available
        """
        if self._initialized:
            return

        try:
            # Detect device
            self._device = _detect_device(self._config.device)

            # Load model (potentially from cache)
            await self._load_model()

            self._initialized = True

        except DeviceError:
            raise
        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def _load_model(self) -> None:
        """
        Load model and tokenizer from HuggingFace or cache.

        Uses global cache to avoid reloading models that are already
        in memory.
        """
        cache_key = f"{self._model_name}:{self._device}"

        async with _cache_lock:
            if cache_key in _model_cache and self._config.cache_models:
                self._model, self._tokenizer, self._model_info = _model_cache[cache_key]
                return

        # Load in executor to avoid blocking
        loop = asyncio.get_event_loop()
        model, tokenizer, model_info = await loop.run_in_executor(
            self._executor, self._load_model_sync
        )

        self._model = model
        self._tokenizer = tokenizer
        self._model_info = model_info

        # Cache if enabled
        if self._config.cache_models:
            async with _cache_lock:
                _model_cache[cache_key] = (model, tokenizer, model_info)

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """
        Synchronously load model and tokenizer.

        Returns:
            Tuple of (model, tokenizer, model_info)

        Raises:
            ModelNotFoundError: If model cannot be found
            ModelLoadError: If model cannot be loaded
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed. Install with: pip install transformers",
            ) from e

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                use_fast=True,
            )

            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name,
            )

            # Move to device
            model = model.to(self._device)

            # Set to evaluation mode
            model.eval()

            # Get model configuration
            num_labels = model.config.num_labels
            labels = self._get_labels(model)

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=num_labels,
                labels=labels,
                max_length=min(
                    tokenizer.model_max_length,
                    self._config.max_length,
                ),
            )

            return model, tokenizer, model_info

        except OSError as e:
            if "not a valid model identifier" in str(e) or "does not exist" in str(e):
                raise ModelNotFoundError(self._model_name) from e
            raise ModelLoadError(
                model_name=self._model_name,
                reason=f"Failed to load model: {e}",
            ) from e
        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    def _get_labels(self, model: Any) -> List[str]:
        """
        Extract label names from model configuration.

        Args:
            model: Loaded HuggingFace model

        Returns:
            List of label names
        """
        if hasattr(model.config, "id2label"):
            id2label = model.config.id2label
            return [id2label[i] for i in range(len(id2label))]
        return [f"LABEL_{i}" for i in range(model.config.num_labels)]

    async def close(self) -> None:
        """Cleanup provider resources."""
        self._model = None
        self._tokenizer = None
        self._model_info = None
        self._initialized = False
        self._executor.shutdown(wait=False)

    async def predict(self, text: str, **kwargs: Any) -> PredictionResult:
        """
        Make a prediction on input text.

        Args:
            text: Input text to analyze
            **kwargs: Additional arguments
                - return_all_scores: Return scores for all labels (default: True)
                - top_k: Only return top-k predictions (default: None = all)

        Returns:
            PredictionResult with label, score, and all_scores

        Raises:
            ModelInferenceError: If inference fails
        """
        self._ensure_initialized()

        if not text or not text.strip():
            return PredictionResult(
                label="neutral",
                score=0.0,
                confidence=0.0,
                all_scores={},
                model_name=self._model_name,
                processing_time_ms=0.0,
            )

        start_time = time.perf_counter()

        try:
            # Run inference in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._predict_sync,
                text,
                kwargs.get("return_all_scores", True),
            )

            processing_time = (time.perf_counter() - start_time) * 1000

            return PredictionResult(
                label=result["label"],
                score=result["score"],
                confidence=result["score"],
                all_scores=result.get("all_scores", {}),
                model_name=self._model_name,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    def _predict_sync(self, text: str, return_all_scores: bool = True) -> Dict[str, Any]:
        """
        Synchronous prediction helper.

        Args:
            text: Input text
            return_all_scores: Whether to return all label scores

        Returns:
            Dictionary with prediction results
        """
        import torch

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Get label with highest probability
        predicted_idx = int(probs.argmax())
        predicted_label = self._model_info.labels[predicted_idx]
        predicted_score = float(probs[predicted_idx])

        result = {
            "label": predicted_label,
            "score": predicted_score,
        }

        if return_all_scores:
            result["all_scores"] = {
                self._model_info.labels[i]: float(probs[i])
                for i in range(len(self._model_info.labels))
            }

        return result

    async def predict_batch(
        self, texts: List[str], **kwargs: Any
    ) -> List[PredictionResult]:
        """
        Make predictions on multiple texts.

        Args:
            texts: List of input texts
            **kwargs: Additional arguments
                - batch_size: Override default batch size
                - return_all_scores: Return scores for all labels

        Returns:
            List of PredictionResult objects

        Raises:
            ModelInferenceError: If inference fails
        """
        self._ensure_initialized()

        if not texts:
            return []

        batch_size = kwargs.get("batch_size", self._config.batch_size)
        return_all_scores = kwargs.get("return_all_scores", True)

        start_time = time.perf_counter()
        results: List[PredictionResult] = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Run batch inference in executor
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    self._executor,
                    self._predict_batch_sync,
                    batch_texts,
                    return_all_scores,
                )

                for result in batch_results:
                    results.append(
                        PredictionResult(
                            label=result["label"],
                            score=result["score"],
                            confidence=result["score"],
                            all_scores=result.get("all_scores", {}),
                            model_name=self._model_name,
                            processing_time_ms=0.0,
                        )
                    )

            # Calculate total processing time and distribute
            total_time = (time.perf_counter() - start_time) * 1000
            time_per_item = total_time / len(results) if results else 0

            for result in results:
                result.processing_time_ms = time_per_item

            return results

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Batch inference failed: {e}",
            ) from e

    def _predict_batch_sync(
        self, texts: List[str], return_all_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Synchronous batch prediction helper.

        Args:
            texts: List of input texts
            return_all_scores: Whether to return all label scores

        Returns:
            List of prediction result dictionaries
        """
        import torch

        # Filter empty texts and track indices
        non_empty_texts = []
        non_empty_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        # Prepare results with defaults for empty texts
        results: List[Dict[str, Any]] = [
            {"label": "neutral", "score": 0.0, "all_scores": {}}
            for _ in range(len(texts))
        ]

        if not non_empty_texts:
            return results

        # Tokenize batch
        inputs = self._tokenizer(
            non_empty_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Apply softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()

        # Process results
        for batch_idx, original_idx in enumerate(non_empty_indices):
            item_probs = probs[batch_idx]
            predicted_idx = int(item_probs.argmax())
            predicted_label = self._model_info.labels[predicted_idx]
            predicted_score = float(item_probs[predicted_idx])

            result = {
                "label": predicted_label,
                "score": predicted_score,
            }

            if return_all_scores:
                result["all_scores"] = {
                    self._model_info.labels[i]: float(item_probs[i])
                    for i in range(len(self._model_info.labels))
                }

            results[original_idx] = result

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        if not self._model_info:
            return {
                "name": self._model_name,
                "model_type": self._model_type.value,
                "loaded": False,
            }

        info = self._model_info.to_dict()
        info["loaded"] = True
        info["device_info"] = _get_device_info()
        return info

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device

    @property
    def labels(self) -> List[str]:
        """Get model labels."""
        if self._model_info:
            return self._model_info.labels
        return []


# Specialized provider classes for specific model types


class SentimentModelProvider(HuggingFaceModelProvider):
    """
    Specialized provider for sentiment analysis models.

    Uses cardiffnlp/twitter-roberta-base-sentiment-latest by default.
    Labels: negative, neutral, positive
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize sentiment model provider."""
        super().__init__(
            config=config,
            model_name=model_name or (config.sentiment_model if config else None),
            model_type=ModelType.SENTIMENT,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        base_info = super().info
        return ProviderInfo(
            name="huggingface-sentiment",
            provider_type=ProviderType.MODEL,
            version=base_info.version,
            description="Sentiment analysis model provider",
            capabilities=base_info.capabilities,
            supported_models=[
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "nlptown/bert-base-multilingual-uncased-sentiment",
                "finiteautomata/bertweet-base-sentiment-analysis",
            ],
        )


class EmotionModelProvider(HuggingFaceModelProvider):
    """
    Specialized provider for emotion detection models.

    Uses SamLowe/roberta-base-go_emotions by default.
    Labels: 28 GoEmotions categories
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize emotion model provider."""
        super().__init__(
            config=config,
            model_name=model_name or (config.emotion_model if config else None),
            model_type=ModelType.EMOTION,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        base_info = super().info
        return ProviderInfo(
            name="huggingface-emotion",
            provider_type=ProviderType.MODEL,
            version=base_info.version,
            description="Emotion detection model provider",
            capabilities=base_info.capabilities,
            supported_models=[
                "SamLowe/roberta-base-go_emotions",
                "j-hartmann/emotion-english-distilroberta-base",
                "bhadresh-savani/distilbert-base-uncased-emotion",
            ],
        )

    async def predict_top_k(
        self, text: str, k: int = 3, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get top-k emotions for input text.

        Args:
            text: Input text
            k: Number of top emotions to return
            threshold: Minimum score threshold

        Returns:
            List of dicts with 'label' and 'score' keys, sorted by score
        """
        result = await self.predict(text, return_all_scores=True)

        # Sort by score
        sorted_scores = sorted(
            result.all_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Filter by threshold and take top-k
        top_emotions = []
        for label, score in sorted_scores[:k]:
            if score >= threshold:
                top_emotions.append({"label": label, "score": score})

        return top_emotions

    async def predict_multi_label(
        self, text: str, threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Get all emotions above threshold (multi-label classification).

        Args:
            text: Input text
            threshold: Score threshold for including an emotion

        Returns:
            List of dicts with 'label' and 'score' for emotions above threshold
        """
        result = await self.predict(text, return_all_scores=True)

        emotions = []
        for label, score in result.all_scores.items():
            if score >= threshold:
                emotions.append({"label": label, "score": score})

        return sorted(emotions, key=lambda x: x["score"], reverse=True)


# Utility functions


def clear_model_cache() -> None:
    """Clear the global model cache."""
    global _model_cache
    _model_cache.clear()


def get_cached_models() -> List[str]:
    """Get list of cached model keys."""
    return list(_model_cache.keys())


# Register providers
register_provider("huggingface", ProviderType.MODEL, HuggingFaceModelProvider)
register_provider("huggingface-sentiment", ProviderType.MODEL, SentimentModelProvider)
register_provider("huggingface-emotion", ProviderType.MODEL, EmotionModelProvider)
