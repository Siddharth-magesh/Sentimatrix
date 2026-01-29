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


# Extended Sentiment Model Providers


class DistilBertSentimentProvider(HuggingFaceModelProvider):
    """
    DistilBERT SST-2 sentiment provider.

    Uses distilbert-base-uncased-finetuned-sst-2-english for fast binary
    sentiment classification (POSITIVE/NEGATIVE).

    Model characteristics:
    - 67M parameters (lightweight)
    - 91.3% accuracy on SST-2
    - Very fast inference, ideal for production
    """

    DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize DistilBERT sentiment provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.SENTIMENT,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="distilbert-sentiment",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Fast binary sentiment analysis using DistilBERT fine-tuned on SST-2",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "distilbert-base-uncased-finetuned-sst-2-english",
            ],
        )


class SiebertSentimentProvider(HuggingFaceModelProvider):
    """
    SiEBERT (Sentiment in English BERT) sentiment provider.

    Uses siebert/sentiment-roberta-large-english for high-accuracy
    binary sentiment classification. Fine-tuned on 15 diverse datasets.

    Model characteristics:
    - RoBERTa-large based (355M parameters)
    - High accuracy across diverse text types
    - Reliable binary classification (POSITIVE/NEGATIVE)
    """

    DEFAULT_MODEL = "siebert/sentiment-roberta-large-english"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize SiEBERT sentiment provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.SENTIMENT,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="siebert-sentiment",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="High-accuracy sentiment analysis using SiEBERT (RoBERTa-large)",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "siebert/sentiment-roberta-large-english",
            ],
        )


class TwitterSentimentProvider(HuggingFaceModelProvider):
    """
    CardiffNLP Twitter sentiment provider.

    Uses cardiffnlp/twitter-roberta-base-sentiment for Twitter-optimized
    3-class sentiment classification. Includes text preprocessing for
    social media content.

    Model characteristics:
    - RoBERTa-base fine-tuned on Twitter data
    - 3-class: negative, neutral, positive
    - Optimized for social media text with @mentions and URLs
    """

    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize Twitter sentiment provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.SENTIMENT,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="twitter-sentiment",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Twitter-optimized sentiment analysis using CardiffNLP RoBERTa",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "cardiffnlp/twitter-roberta-base-sentiment",
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
            ],
        )

    @staticmethod
    def preprocess_twitter_text(text: str) -> str:
        """
        Preprocess text for Twitter models.

        Replaces @mentions with @user and URLs with http placeholder.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        new_text = []
        for token in text.split(" "):
            token = "@user" if token.startswith("@") and len(token) > 1 else token
            token = "http" if token.startswith("http") else token
            new_text.append(token)
        return " ".join(new_text)

    def _predict_sync(self, text: str, return_all_scores: bool = True) -> Dict[str, Any]:
        """Override to add preprocessing."""
        preprocessed = self.preprocess_twitter_text(text)
        return super()._predict_sync(preprocessed, return_all_scores)

    def _predict_batch_sync(
        self, texts: List[str], return_all_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """Override to add preprocessing."""
        preprocessed = [self.preprocess_twitter_text(t) for t in texts]
        return super()._predict_batch_sync(preprocessed, return_all_scores)


class MultilingualSentimentProvider(HuggingFaceModelProvider):
    """
    Multilingual sentiment provider.

    Uses lxyuan/distilbert-base-multilingual-cased-sentiments-student
    for sentiment analysis across multiple languages.

    Model characteristics:
    - DistilBERT multilingual base
    - Supports 100+ languages
    - 3-class: positive, negative, neutral
    """

    DEFAULT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize multilingual sentiment provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.SENTIMENT,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="multilingual-sentiment",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Multilingual sentiment analysis supporting 100+ languages",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            ],
        )


# Extended Emotion Model Providers


class TwitterEmotionProvider(HuggingFaceModelProvider):
    """
    CardiffNLP Twitter emotion provider.

    Uses cardiffnlp/twitter-roberta-base-emotion for Twitter-optimized
    emotion classification. Includes text preprocessing.

    Model characteristics:
    - RoBERTa-base fine-tuned on Twitter data
    - 4 classes: anger, joy, optimism, sadness
    - Optimized for social media text
    """

    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-emotion"

    # Label mapping for this model
    LABELS = ["anger", "joy", "optimism", "sadness"]

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize Twitter emotion provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.EMOTION,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="twitter-emotion",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Twitter-optimized emotion detection using CardiffNLP RoBERTa",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "cardiffnlp/twitter-roberta-base-emotion",
                "cardiffnlp/twitter-roberta-base-emotion-latest",
            ],
        )

    @staticmethod
    def preprocess_twitter_text(text: str) -> str:
        """Preprocess text for Twitter models."""
        new_text = []
        for token in text.split(" "):
            token = "@user" if token.startswith("@") and len(token) > 1 else token
            token = "http" if token.startswith("http") else token
            new_text.append(token)
        return " ".join(new_text)

    def _predict_sync(self, text: str, return_all_scores: bool = True) -> Dict[str, Any]:
        """Override to add preprocessing."""
        preprocessed = self.preprocess_twitter_text(text)
        return super()._predict_sync(preprocessed, return_all_scores)


class T5EmotionProvider(HuggingFaceModelProvider):
    """
    T5-based emotion detection provider.

    Uses mrm8488/t5-base-finetuned-emotion for T5-based emotion classification.
    This model uses a text-to-text approach.

    Model characteristics:
    - T5-base architecture
    - 6 Ekman emotions: sadness, joy, love, anger, fear, surprise
    - Text generation approach
    """

    DEFAULT_MODEL = "mrm8488/t5-base-finetuned-emotion"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize T5 emotion provider."""
        # Note: T5 requires different loading approach
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.EMOTION,
        )
        self._is_t5 = True

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load T5 model (different from sequence classification)."""
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = T5Tokenizer.from_pretrained(self._model_name)
            model = T5ForConditionalGeneration.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=6,  # 6 basic emotions
                labels=["sadness", "joy", "love", "anger", "fear", "surprise"],
                max_length=min(512, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    def _predict_sync(self, text: str, return_all_scores: bool = True) -> Dict[str, Any]:
        """T5-based prediction using text generation."""
        import torch

        # T5 uses text-to-text format
        input_text = f"emotion: {text}"
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=10,
                num_beams=1,
                do_sample=False,
            )

        predicted_emotion = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_emotion = predicted_emotion.strip().lower()

        # Map to standard label
        if predicted_emotion not in self._model_info.labels:
            predicted_emotion = "joy"  # default

        result = {
            "label": predicted_emotion,
            "score": 1.0,  # T5 doesn't provide probability scores directly
        }

        if return_all_scores:
            # For T5, we only have the predicted label
            result["all_scores"] = {label: 0.0 for label in self._model_info.labels}
            result["all_scores"][predicted_emotion] = 1.0

        return result

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="t5-emotion",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="T5-based emotion detection",
            capabilities=ProviderCapabilities(
                batch_processing=False,  # T5 generation doesn't batch well
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "mrm8488/t5-base-finetuned-emotion",
            ],
        )


# Aspect-Based Sentiment Analysis (ABSA) Providers


class ABSAModelType(str, Enum):
    """Types of ABSA models."""

    ASPECT_EXTRACTION = "ate"  # Aspect Term Extraction
    ASPECT_SENTIMENT = "atsc"  # Aspect Term Sentiment Classification
    JOINT = "joint"  # Joint ATE + ATSC


@dataclass
class ABSAResult:
    """
    Result of Aspect-Based Sentiment Analysis.

    Attributes:
        text: Original input text
        aspect: The aspect being analyzed
        sentiment: Sentiment towards the aspect (positive/negative/neutral)
        confidence: Confidence score
        all_scores: Scores for all sentiment labels
        model_name: Model used for analysis
    """

    text: str
    aspect: str
    sentiment: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "aspect": self.aspect,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "all_scores": self.all_scores,
            "model_name": self.model_name,
        }


class DeBERTaABSAProvider(HuggingFaceModelProvider):
    """
    DeBERTa-v3 ABSA provider from yangheng.

    Uses yangheng/deberta-v3-base-absa-v1.1 for Aspect-Based Sentiment Analysis.
    Takes text and aspect as input, predicts sentiment towards the aspect.

    Model characteristics:
    - DeBERTa-v3 architecture
    - 1M+ downloads (most popular ABSA model)
    - Supports multiple languages including Chinese
    - 3-class output: Negative, Neutral, Positive

    Usage:
        >>> provider = DeBERTaABSAProvider()
        >>> await provider.initialize()
        >>> result = await provider.predict_aspect_sentiment(
        ...     "The food was great but service was slow",
        ...     aspect="food"
        ... )
    """

    DEFAULT_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
    LARGE_MODEL = "yangheng/deberta-v3-large-absa-v1.1"

    # ABSA label mapping
    LABELS = ["Negative", "Neutral", "Positive"]

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
        use_large: bool = False,
    ) -> None:
        """
        Initialize DeBERTa ABSA provider.

        Args:
            config: Model configuration
            model_name: Override model name
            use_large: Use large model variant (more accurate, slower)
        """
        if model_name is None:
            model_name = self.LARGE_MODEL if use_large else self.DEFAULT_MODEL

        super().__init__(
            config=config,
            model_name=model_name,
            model_type=ModelType.CLASSIFICATION,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="deberta-absa",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Aspect-Based Sentiment Analysis using DeBERTa-v3",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "yangheng/deberta-v3-base-absa-v1.1",
                "yangheng/deberta-v3-large-absa-v1.1",
            ],
        )

    async def predict_aspect_sentiment(
        self, text: str, aspect: str
    ) -> ABSAResult:
        """
        Predict sentiment towards a specific aspect in the text.

        Args:
            text: The input text/review
            aspect: The aspect to analyze sentiment for

        Returns:
            ABSAResult with sentiment towards the aspect

        Example:
            >>> result = await provider.predict_aspect_sentiment(
            ...     "The food was great but the service was terrible",
            ...     aspect="food"
            ... )
            >>> print(result.sentiment)  # "Positive"
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._predict_aspect_sync,
                text,
                aspect,
            )

            processing_time = (time.perf_counter() - start_time) * 1000

            return ABSAResult(
                text=text,
                aspect=aspect,
                sentiment=result["label"],
                confidence=result["score"],
                all_scores=result.get("all_scores", {}),
                model_name=self._model_name,
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"ABSA inference failed: {e}",
            ) from e

    def _predict_aspect_sync(self, text: str, aspect: str) -> Dict[str, Any]:
        """
        Synchronous aspect sentiment prediction.

        Uses the model's text pair classification capability where:
        - text: The input sentence/review
        - text_pair: The aspect term
        """
        import torch

        # Tokenize with text pair (text, aspect)
        inputs = self._tokenizer(
            text,
            aspect,  # text_pair
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        predicted_idx = int(probs.argmax())
        predicted_label = self._model_info.labels[predicted_idx]
        predicted_score = float(probs[predicted_idx])

        return {
            "label": predicted_label,
            "score": predicted_score,
            "all_scores": {
                self._model_info.labels[i]: float(probs[i])
                for i in range(len(self._model_info.labels))
            },
        }

    async def predict_multiple_aspects(
        self, text: str, aspects: List[str]
    ) -> List[ABSAResult]:
        """
        Predict sentiment for multiple aspects in the same text.

        Args:
            text: The input text/review
            aspects: List of aspects to analyze

        Returns:
            List of ABSAResult for each aspect
        """
        results = []
        for aspect in aspects:
            result = await self.predict_aspect_sentiment(text, aspect)
            results.append(result)
        return results


class InstructABSAProvider(HuggingFaceModelProvider):
    """
    InstructABSA provider using Tk-Instruct.

    Uses kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined for
    instruction-based ABSA. This is a joint model that can extract aspects
    and classify sentiment simultaneously.

    Model characteristics:
    - T5-based instruction-following model
    - Joint aspect extraction and sentiment classification
    - SOTA performance on SemEval 2014

    Note: This model uses a text-to-text format with instruction prompts.
    """

    DEFAULT_MODEL = "kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize InstructABSA provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.CLASSIFICATION,
        )
        self._is_t5_based = True

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the T5-based InstructABSA model."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=3,
                labels=["positive", "negative", "neutral"],
                max_length=min(512, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def extract_aspects_and_sentiment(self, text: str) -> List[ABSAResult]:
        """
        Extract aspects and their sentiments from text.

        This is a joint task that identifies aspect terms and classifies
        their sentiment in one pass.

        Args:
            text: Input review/text

        Returns:
            List of ABSAResult with extracted aspects and sentiments
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._extract_aspects_sync,
                text,
            )
            return result

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"InstructABSA extraction failed: {e}",
            ) from e

    def _extract_aspects_sync(self, text: str) -> List[ABSAResult]:
        """Synchronous aspect extraction."""
        import torch

        # InstructABSA prompt format
        prompt = (
            "Definition: The output will be the aspects (both implicit and explicit) "
            "and the aspects sentiment polarity. In cases where there are no aspects "
            "the output should be noaspectterm:none. "
            f"Now complete the following example- input: {text}"
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                do_sample=False,
            )

        output_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse output format: "aspect1:sentiment1, aspect2:sentiment2, ..."
        results = []

        if "noaspectterm" in output_text.lower() or not output_text.strip():
            return results

        # Parse aspect:sentiment pairs
        pairs = output_text.split(",")
        for pair in pairs:
            pair = pair.strip()
            if ":" in pair:
                parts = pair.split(":")
                if len(parts) == 2:
                    aspect = parts[0].strip()
                    sentiment = parts[1].strip().lower()

                    # Normalize sentiment
                    if sentiment in ["positive", "pos"]:
                        sentiment = "positive"
                    elif sentiment in ["negative", "neg"]:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"

                    results.append(ABSAResult(
                        text=text,
                        aspect=aspect,
                        sentiment=sentiment,
                        confidence=1.0,
                        model_name=self._model_name,
                    ))

        return results

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="instruct-absa",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Instruction-based ABSA using Tk-Instruct (joint extraction + sentiment)",
            capabilities=ProviderCapabilities(
                batch_processing=False,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined",
            ],
        )


# ============================================================================
# MULTILINGUAL MODELS
# ============================================================================


class XLMRobertaSentimentProvider(HuggingFaceModelProvider):
    """
    XLM-RoBERTa multilingual sentiment provider.

    Uses cardiffnlp/twitter-xlm-roberta-base-sentiment for multilingual
    sentiment analysis trained on ~198M tweets in 8 languages.

    Model characteristics:
    - XLM-RoBERTa base architecture
    - Trained on Ar, En, Fr, De, Hi, It, Sp, Pt
    - Works on 100+ languages
    - 3-class: Negative, Neutral, Positive
    """

    DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize XLM-RoBERTa sentiment provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.SENTIMENT,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="xlm-roberta-sentiment",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Multilingual sentiment analysis using XLM-RoBERTa (100+ languages)",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
            ],
        )


# ============================================================================
# DOMAIN-SPECIFIC MODELS
# ============================================================================


class FinBERTProvider(HuggingFaceModelProvider):
    """
    FinBERT financial sentiment provider.

    Uses ProsusAI/finbert for financial sentiment analysis.
    Trained on Financial PhraseBank dataset.

    Model characteristics:
    - BERT-base fine-tuned on financial corpus
    - 3-class: positive, negative, neutral
    - Optimized for financial news and reports
    """

    DEFAULT_MODEL = "ProsusAI/finbert"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize FinBERT provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.SENTIMENT,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="finbert",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Financial sentiment analysis using FinBERT",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "ProsusAI/finbert",
                "yiyanghkust/finbert-tone",
            ],
        )


class FinBERTToneProvider(HuggingFaceModelProvider):
    """
    FinBERT-Tone financial tone provider.

    Uses yiyanghkust/finbert-tone for financial tone analysis.

    Model characteristics:
    - BERT-base fine-tuned for financial tone
    - 3-class: positive, negative, neutral
    - Good for earnings calls and financial communications
    """

    DEFAULT_MODEL = "yiyanghkust/finbert-tone"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize FinBERT-Tone provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.SENTIMENT,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="finbert-tone",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Financial tone analysis using FinBERT-Tone",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "yiyanghkust/finbert-tone",
            ],
        )


# ============================================================================
# ZERO-SHOT CLASSIFICATION MODELS
# ============================================================================


@dataclass
class ZeroShotResult:
    """
    Result of zero-shot classification.

    Attributes:
        text: Original input text
        labels: Candidate labels
        scores: Scores for each label
        predicted_label: Label with highest score
        confidence: Confidence of prediction
        model_name: Model used
    """

    text: str
    labels: List[str]
    scores: Dict[str, float]
    predicted_label: str
    confidence: float
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "labels": self.labels,
            "scores": self.scores,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "model_name": self.model_name,
        }


class ZeroShotClassificationProvider(HuggingFaceModelProvider):
    """
    Zero-shot classification provider using BART-MNLI.

    Uses facebook/bart-large-mnli for zero-shot text classification.
    Can classify text into arbitrary categories without training.

    Model characteristics:
    - BART-large trained on MNLI
    - No fixed label set - specify at inference time
    - Supports multi-label classification
    """

    DEFAULT_MODEL = "facebook/bart-large-mnli"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize zero-shot classification provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.CLASSIFICATION,
        )

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the zero-shot classification model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=3,  # entailment, neutral, contradiction
                labels=["entailment", "neutral", "contradiction"],
                max_length=min(1024, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def classify(
        self,
        text: str,
        candidate_labels: List[str],
        hypothesis_template: str = "This text is about {}.",
        multi_label: bool = False,
    ) -> ZeroShotResult:
        """
        Classify text into candidate labels.

        Args:
            text: Text to classify
            candidate_labels: List of possible labels
            hypothesis_template: Template for hypothesis (use {} for label)
            multi_label: Whether to allow multiple labels

        Returns:
            ZeroShotResult with scores for each label
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._classify_sync,
                text,
                candidate_labels,
                hypothesis_template,
                multi_label,
            )
            return result

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Zero-shot classification failed: {e}",
            ) from e

    def _classify_sync(
        self,
        text: str,
        candidate_labels: List[str],
        hypothesis_template: str,
        multi_label: bool,
    ) -> ZeroShotResult:
        """Synchronous zero-shot classification."""
        import torch

        scores = {}

        for label in candidate_labels:
            hypothesis = hypothesis_template.format(label)

            inputs = self._tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=self._model_info.max_length,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

            # For MNLI: entailment=2, neutral=1, contradiction=0
            probs = torch.nn.functional.softmax(logits, dim=-1)
            entailment_score = float(probs[0, 2].cpu())
            scores[label] = entailment_score

        if not multi_label:
            # Normalize scores to sum to 1
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}

        predicted_label = max(scores, key=scores.get)
        confidence = scores[predicted_label]

        return ZeroShotResult(
            text=text,
            labels=candidate_labels,
            scores=scores,
            predicted_label=predicted_label,
            confidence=confidence,
            model_name=self._model_name,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="zero-shot-classification",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Zero-shot text classification using BART-MNLI",
            capabilities=ProviderCapabilities(
                batch_processing=False,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "facebook/bart-large-mnli",
                "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            ],
        )


# ============================================================================
# EMBEDDING MODELS
# ============================================================================


@dataclass
class EmbeddingResult:
    """
    Result of text embedding.

    Attributes:
        text: Original input text
        embedding: The embedding vector
        model_name: Model used
        dimension: Embedding dimension
    """

    text: str
    embedding: List[float]
    model_name: str
    dimension: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "embedding": self.embedding,
            "model_name": self.model_name,
            "dimension": self.dimension,
        }


class SentenceEmbeddingProvider(HuggingFaceModelProvider):
    """
    Sentence embedding provider using sentence-transformers.

    Uses sentence-transformers/all-MiniLM-L6-v2 for fast, quality embeddings.
    Maps sentences to 384-dimensional dense vectors.

    Model characteristics:
    - Only 22MB, very fast
    - 384-dimensional embeddings
    - Good for semantic search, clustering, similarity
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
        use_mpnet: bool = False,
    ) -> None:
        """
        Initialize sentence embedding provider.

        Args:
            config: Model configuration
            model_name: Override model name
            use_mpnet: Use MPNet model (higher quality, slower)
        """
        if model_name is None:
            model_name = self.MPNET_MODEL if use_mpnet else self.DEFAULT_MODEL

        super().__init__(
            config=config,
            model_name=model_name,
            model_type=ModelType.CLASSIFICATION,
        )
        self._embedding_dim = 768 if use_mpnet else 384

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the sentence embedding model."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModel.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=self._embedding_dim,
                labels=[],
                max_length=min(512, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def encode(self, text: str) -> EmbeddingResult:
        """
        Encode text into an embedding vector.

        Args:
            text: Text to encode

        Returns:
            EmbeddingResult with the embedding vector
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                self._encode_sync,
                text,
            )

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self._model_name,
                dimension=len(embedding),
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Embedding failed: {e}",
            ) from e

    async def encode_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Encode multiple texts into embedding vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of EmbeddingResult objects
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self._executor,
                self._encode_batch_sync,
                texts,
            )

            return [
                EmbeddingResult(
                    text=text,
                    embedding=emb,
                    model_name=self._model_name,
                    dimension=len(emb),
                )
                for text, emb in zip(texts, embeddings)
            ]

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Batch embedding failed: {e}",
            ) from e

    def _encode_sync(self, text: str) -> List[float]:
        """Synchronous single text encoding."""
        return self._encode_batch_sync([text])[0]

    def _encode_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch encoding with mean pooling."""
        import torch
        import torch.nn.functional as F

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    async def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (-1 to 1)
        """
        emb1 = await self.encode(text1)
        emb2 = await self.encode(text2)

        # Compute cosine similarity
        import numpy as np
        v1 = np.array(emb1.embedding)
        v2 = np.array(emb2.embedding)
        return float(np.dot(v1, v2))

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="sentence-embedding",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Sentence embeddings using all-MiniLM-L6-v2",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5",
            ],
        )


# ============================================================================
# SPEECH-TO-TEXT MODELS
# ============================================================================


@dataclass
class TranscriptionResult:
    """
    Result of speech-to-text transcription.

    Attributes:
        text: Transcribed text
        audio_path: Path to audio file
        language: Detected language (if available)
        duration_seconds: Audio duration
        model_name: Model used
    """

    text: str
    audio_path: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "audio_path": self.audio_path,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "model_name": self.model_name,
        }


class WhisperProvider(HuggingFaceModelProvider):
    """
    OpenAI Whisper speech-to-text provider.

    Uses openai/whisper models for automatic speech recognition.
    Supports multiple languages and model sizes.

    Model characteristics:
    - Encoder-decoder transformer
    - Supports 99+ languages
    - Available in base, small, medium, large variants
    """

    MODELS = {
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3",
    }

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
        size: str = "base",
    ) -> None:
        """
        Initialize Whisper provider.

        Args:
            config: Model configuration
            model_name: Override model name
            size: Model size (base, small, medium, large)
        """
        if model_name is None:
            model_name = self.MODELS.get(size, self.MODELS["base"])

        super().__init__(
            config=config,
            model_name=model_name,
            model_type=ModelType.CLASSIFICATION,
        )
        self._processor = None

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the Whisper model."""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import torch
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

            processor = AutoProcessor.from_pretrained(self._model_name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            model = model.to(self._device)
            model.eval()

            self._processor = processor

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=0,
                labels=[],
                max_length=448,
            )

            return model, processor, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Language code (optional, auto-detected)
            task: "transcribe" or "translate" (to English)

        Returns:
            TranscriptionResult with transcribed text
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._transcribe_sync,
                audio_path,
                language,
                task,
            )
            return result

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Transcription failed: {e}",
            ) from e

    def _transcribe_sync(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
    ) -> TranscriptionResult:
        """Synchronous transcription."""
        import torch

        try:
            import librosa
        except ImportError:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason="librosa library not installed. Install with: pip install librosa",
            )

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr

        # Process
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
        inputs = {k: v.to(self._device, dtype=torch_dtype) for k, v in inputs.items()}

        # Generate options
        generate_kwargs = {"max_new_tokens": 448}
        if language:
            generate_kwargs["language"] = language
        if task == "translate":
            generate_kwargs["task"] = "translate"

        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_features"],
                **generate_kwargs,
            )

        text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return TranscriptionResult(
            text=text.strip(),
            audio_path=audio_path,
            language=language,
            duration_seconds=duration,
            model_name=self._model_name,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="whisper",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Speech-to-text using OpenAI Whisper",
            capabilities=ProviderCapabilities(
                batch_processing=False,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=list(self.MODELS.values()),
        )


# ============================================================================
# VISION MODELS
# ============================================================================


@dataclass
class ImageCaptionResult:
    """
    Result of image captioning.

    Attributes:
        image_path: Path to image
        caption: Generated caption
        conditional_caption: Caption with prompt (if used)
        model_name: Model used
    """

    image_path: str
    caption: str
    conditional_caption: Optional[str] = None
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_path": self.image_path,
            "caption": self.caption,
            "conditional_caption": self.conditional_caption,
            "model_name": self.model_name,
        }


class BLIPCaptioningProvider(HuggingFaceModelProvider):
    """
    BLIP image captioning provider.

    Uses Salesforce/blip-image-captioning-base for generating
    captions from images.

    Model characteristics:
    - Vision-language pre-training
    - Supports conditional and unconditional captioning
    - Available in base and large variants
    """

    DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"
    LARGE_MODEL = "Salesforce/blip-image-captioning-large"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
        use_large: bool = False,
    ) -> None:
        """Initialize BLIP captioning provider."""
        if model_name is None:
            model_name = self.LARGE_MODEL if use_large else self.DEFAULT_MODEL

        super().__init__(
            config=config,
            model_name=model_name,
            model_type=ModelType.CLASSIFICATION,
        )
        self._processor = None

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the BLIP model."""
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            import torch
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

            processor = BlipProcessor.from_pretrained(self._model_name)
            model = BlipForConditionalGeneration.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
            )
            model = model.to(self._device)
            model.eval()

            self._processor = processor

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=0,
                labels=[],
                max_length=512,
            )

            return model, processor, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def caption(
        self,
        image_path: str,
        prompt: Optional[str] = None,
    ) -> ImageCaptionResult:
        """
        Generate caption for an image.

        Args:
            image_path: Path to image file
            prompt: Optional prompt for conditional captioning

        Returns:
            ImageCaptionResult with generated caption
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._caption_sync,
                image_path,
                prompt,
            )
            return result

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Captioning failed: {e}",
            ) from e

    def _caption_sync(
        self,
        image_path: str,
        prompt: Optional[str],
    ) -> ImageCaptionResult:
        """Synchronous image captioning."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

        # Unconditional captioning
        inputs = self._processor(image, return_tensors="pt")
        inputs = {k: v.to(self._device, dtype=torch_dtype) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=50)
        caption = self._processor.decode(out[0], skip_special_tokens=True)

        # Conditional captioning if prompt provided
        conditional_caption = None
        if prompt:
            inputs = self._processor(image, prompt, return_tensors="pt")
            inputs = {k: v.to(self._device, dtype=torch_dtype) for k, v in inputs.items()}

            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=50)
            conditional_caption = self._processor.decode(out[0], skip_special_tokens=True)

        return ImageCaptionResult(
            image_path=image_path,
            caption=caption,
            conditional_caption=conditional_caption,
            model_name=self._model_name,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="blip-captioning",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description="Image captioning using Salesforce BLIP",
            capabilities=ProviderCapabilities(
                batch_processing=False,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[
                "Salesforce/blip-image-captioning-base",
                "Salesforce/blip-image-captioning-large",
            ],
        )


# ============================================================================
# TRANSLATION MODELS
# ============================================================================


@dataclass
class TranslationResult:
    """
    Result of text translation.

    Attributes:
        source_text: Original text
        translated_text: Translated text
        source_lang: Source language code
        target_lang: Target language code
        model_name: Model used
    """

    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "model_name": self.model_name,
        }


class OpusMTTranslationProvider(HuggingFaceModelProvider):
    """
    OPUS-MT translation provider.

    Uses Helsinki-NLP/opus-mt models for machine translation.
    Supports many language pairs.

    Model characteristics:
    - Marian NMT architecture
    - Specific models for each language pair
    - High-quality translations
    """

    # Common language pairs
    MODELS = {
        "en-de": "Helsinki-NLP/opus-mt-en-de",
        "de-en": "Helsinki-NLP/opus-mt-de-en",
        "en-fr": "Helsinki-NLP/opus-mt-en-fr",
        "fr-en": "Helsinki-NLP/opus-mt-fr-en",
        "en-es": "Helsinki-NLP/opus-mt-en-es",
        "es-en": "Helsinki-NLP/opus-mt-es-en",
        "en-zh": "Helsinki-NLP/opus-mt-en-zh",
        "zh-en": "Helsinki-NLP/opus-mt-zh-en",
        "en-ja": "Helsinki-NLP/opus-mt-en-ja",
        "ja-en": "Helsinki-NLP/opus-mt-ja-en",
        "en-ru": "Helsinki-NLP/opus-mt-en-ru",
        "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    }

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
        source_lang: str = "en",
        target_lang: str = "de",
    ) -> None:
        """
        Initialize OPUS-MT translation provider.

        Args:
            config: Model configuration
            model_name: Override model name (Helsinki-NLP/opus-mt-xx-yy format)
            source_lang: Source language code
            target_lang: Target language code
        """
        self._source_lang = source_lang
        self._target_lang = target_lang

        if model_name is None:
            lang_pair = f"{source_lang}-{target_lang}"
            model_name = self.MODELS.get(lang_pair, f"Helsinki-NLP/opus-mt-{lang_pair}")

        super().__init__(
            config=config,
            model_name=model_name,
            model_type=ModelType.CLASSIFICATION,
        )

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the translation model."""
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = MarianTokenizer.from_pretrained(self._model_name)
            model = MarianMTModel.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=0,
                labels=[],
                max_length=min(512, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def translate(self, text: str) -> TranslationResult:
        """
        Translate text.

        Args:
            text: Text to translate

        Returns:
            TranslationResult with translated text
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            translated = await loop.run_in_executor(
                self._executor,
                self._translate_sync,
                text,
            )

            return TranslationResult(
                source_text=text,
                translated_text=translated,
                source_lang=self._source_lang,
                target_lang=self._target_lang,
                model_name=self._model_name,
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Translation failed: {e}",
            ) from e

    async def translate_batch(self, texts: List[str]) -> List[TranslationResult]:
        """
        Translate multiple texts.

        Args:
            texts: List of texts to translate

        Returns:
            List of TranslationResult objects
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            translated_texts = await loop.run_in_executor(
                self._executor,
                self._translate_batch_sync,
                texts,
            )

            return [
                TranslationResult(
                    source_text=src,
                    translated_text=tgt,
                    source_lang=self._source_lang,
                    target_lang=self._target_lang,
                    model_name=self._model_name,
                )
                for src, tgt in zip(texts, translated_texts)
            ]

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Batch translation failed: {e}",
            ) from e

    def _translate_sync(self, text: str) -> str:
        """Synchronous translation."""
        return self._translate_batch_sync([text])[0]

    def _translate_batch_sync(self, texts: List[str]) -> List[str]:
        """Synchronous batch translation."""
        import torch

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(**inputs)

        return [
            self._tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="opus-mt-translation",
            provider_type=ProviderType.MODEL,
            version="1.0.0",
            description=f"Translation ({self._source_lang} -> {self._target_lang}) using OPUS-MT",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=list(self.MODELS.values()),
        )


# ============================================================================
# DOMAIN-SPECIFIC PROVIDERS (Legal, Scientific)
# ============================================================================


class LegalBERTProvider(HuggingFaceModelProvider):
    """
    Legal-BERT provider for legal domain text.

    Uses nlpaueb/legal-bert-base-uncased for legal text understanding.
    Pre-trained on 12GB of diverse English legal text.

    Best for:
    - Legal document analysis
    - Contract understanding
    - Legal NLP tasks
    """

    DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize Legal-BERT provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.CLASSIFICATION,
        )
        self._embedding_dim = 768

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the Legal-BERT model."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModel.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=self._embedding_dim,
                labels=[],
                max_length=min(512, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def encode(self, text: str) -> EmbeddingResult:
        """
        Encode legal text into an embedding vector.

        Args:
            text: Legal text to encode

        Returns:
            EmbeddingResult with the embedding vector
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                self._encode_sync,
                text,
            )

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self._model_name,
                dimension=len(embedding),
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Legal embedding failed: {e}",
            ) from e

    def _encode_sync(self, text: str) -> List[float]:
        """Synchronous encoding with mean pooling."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        return embeddings[0].cpu().numpy().tolist()

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="legal-bert",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description="Legal-BERT for legal domain text understanding",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[self.DEFAULT_MODEL],
        )


class SciBERTProvider(HuggingFaceModelProvider):
    """
    SciBERT provider for scientific text.

    Uses allenai/scibert_scivocab_uncased for scientific text understanding.
    Pre-trained on scientific papers from Semantic Scholar.

    Best for:
    - Scientific paper analysis
    - Research text understanding
    - Biomedical/CS domain tasks
    """

    DEFAULT_MODEL = "allenai/scibert_scivocab_uncased"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize SciBERT provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.CLASSIFICATION,
        )
        self._embedding_dim = 768

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the SciBERT model."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModel.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=self._embedding_dim,
                labels=[],
                max_length=min(512, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def encode(self, text: str) -> EmbeddingResult:
        """
        Encode scientific text into an embedding vector.

        Args:
            text: Scientific text to encode

        Returns:
            EmbeddingResult with the embedding vector
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                self._encode_sync,
                text,
            )

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self._model_name,
                dimension=len(embedding),
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Scientific embedding failed: {e}",
            ) from e

    def _encode_sync(self, text: str) -> List[float]:
        """Synchronous encoding with mean pooling."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        return embeddings[0].cpu().numpy().tolist()

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="scibert",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description="SciBERT for scientific text understanding",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[self.DEFAULT_MODEL],
        )


# ============================================================================
# ADDITIONAL ZERO-SHOT PROVIDERS
# ============================================================================


class DeBERTaNLIProvider(HuggingFaceModelProvider):
    """
    DeBERTa-v3 NLI provider for zero-shot classification.

    Uses MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli for high-accuracy
    zero-shot classification. Trained on MNLI, FEVER-NLI, and ANLI datasets.

    Performance:
    - MNLI accuracy: 90.3%
    - ANLI accuracy: 57.9%
    - Outperforms most large models on ANLI
    """

    DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize DeBERTa NLI provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.CLASSIFICATION,
        )

    async def classify(
        self,
        text: str,
        candidate_labels: List[str],
        hypothesis_template: str = "This text is about {}.",
        multi_label: bool = False,
    ) -> ZeroShotResult:
        """
        Classify text into candidate labels using DeBERTa NLI.

        Args:
            text: Text to classify
            candidate_labels: List of possible labels
            hypothesis_template: Template for hypothesis (use {} for label)
            multi_label: Whether to treat as multi-label classification

        Returns:
            ZeroShotResult with predictions
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._classify_sync,
                text,
                candidate_labels,
                hypothesis_template,
                multi_label,
            )
            return result

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"DeBERTa classification failed: {e}",
            ) from e

    def _classify_sync(
        self,
        text: str,
        candidate_labels: List[str],
        hypothesis_template: str,
        multi_label: bool,
    ) -> ZeroShotResult:
        """Synchronous zero-shot classification."""
        import torch
        import torch.nn.functional as F

        scores = {}
        for label in candidate_labels:
            hypothesis = hypothesis_template.format(label)
            inputs = self._tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=self._model_info.max_length,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                # entailment is usually index 0 or 2 depending on model
                probs = F.softmax(logits, dim=-1)
                # For NLI models: 0=contradiction, 1=neutral, 2=entailment
                entailment_score = probs[0, 2].item() if probs.size(1) == 3 else probs[0, 0].item()
                scores[label] = entailment_score

        if not multi_label:
            # Normalize scores
            total = sum(scores.values())
            scores = {k: v / total for k, v in scores.items()}

        predicted_label = max(scores, key=scores.get)
        confidence = scores[predicted_label]

        return ZeroShotResult(
            text=text,
            labels=candidate_labels,
            scores=scores,
            predicted_label=predicted_label,
            confidence=confidence,
            model_name=self._model_name,
        )

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="deberta-nli",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description="DeBERTa-v3 NLI for high-accuracy zero-shot classification",
            capabilities=ProviderCapabilities(
                batch_processing=False,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[self.DEFAULT_MODEL],
        )


# ============================================================================
# ADDITIONAL EMBEDDING PROVIDERS (BGE, E5)
# ============================================================================


class BGEEmbeddingProvider(HuggingFaceModelProvider):
    """
    BGE (BAAI General Embedding) provider.

    Uses BAAI/bge-small-en-v1.5 for efficient embeddings.
    33M parameters, 384-dimensional embeddings.

    Best for:
    - Dense retrieval
    - Semantic search
    - Similarity tasks
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    LARGE_MODEL = "BAAI/bge-large-en-v1.5"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
        use_large: bool = False,
    ) -> None:
        """
        Initialize BGE embedding provider.

        Args:
            config: Model configuration
            model_name: Override model name
            use_large: Use large model (higher quality, slower)
        """
        if model_name is None:
            model_name = self.LARGE_MODEL if use_large else self.DEFAULT_MODEL

        super().__init__(
            config=config,
            model_name=model_name,
            model_type=ModelType.CLASSIFICATION,
        )
        self._embedding_dim = 1024 if use_large else 384

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the BGE embedding model."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModel.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=self._embedding_dim,
                labels=[],
                max_length=min(512, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def encode(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Encode text into an embedding vector.

        Args:
            text: Text to encode
            instruction: Optional instruction prefix for queries

        Returns:
            EmbeddingResult with the embedding vector
        """
        self._ensure_initialized()

        # Add instruction for query encoding
        if instruction:
            text = f"{instruction} {text}"

        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                self._encode_sync,
                text,
            )

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self._model_name,
                dimension=len(embedding),
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"BGE embedding failed: {e}",
            ) from e

    async def encode_query(self, query: str) -> EmbeddingResult:
        """
        Encode a query with instruction prefix.

        Args:
            query: Query text to encode

        Returns:
            EmbeddingResult with the embedding vector
        """
        instruction = "Represent this sentence for searching relevant passages:"
        return await self.encode(query, instruction=instruction)

    def _encode_sync(self, text: str) -> List[float]:
        """Synchronous encoding with mean pooling."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().numpy().tolist()

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="bge-embedding",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description="BGE embeddings for dense retrieval and semantic search",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[self.DEFAULT_MODEL, self.LARGE_MODEL],
        )


class E5EmbeddingProvider(HuggingFaceModelProvider):
    """
    E5 (EmbEddings from bidirectional Encoder rEpresentations) provider.

    Uses intfloat/e5-base-v2 or e5-large-v2 for embeddings.
    High-quality embeddings for retrieval tasks.

    Best for:
    - Dense passage retrieval
    - Semantic similarity
    - Document ranking
    """

    DEFAULT_MODEL = "intfloat/e5-base-v2"
    LARGE_MODEL = "intfloat/e5-large-v2"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
        use_large: bool = False,
    ) -> None:
        """
        Initialize E5 embedding provider.

        Args:
            config: Model configuration
            model_name: Override model name
            use_large: Use large model (higher quality, slower)
        """
        if model_name is None:
            model_name = self.LARGE_MODEL if use_large else self.DEFAULT_MODEL

        super().__init__(
            config=config,
            model_name=model_name,
            model_type=ModelType.CLASSIFICATION,
        )
        self._embedding_dim = 1024 if use_large else 768

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the E5 embedding model."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModel.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=self._embedding_dim,
                labels=[],
                max_length=min(512, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def encode(
        self,
        text: str,
        prefix: str = "query: ",
    ) -> EmbeddingResult:
        """
        Encode text into an embedding vector.

        Args:
            text: Text to encode
            prefix: Prefix to add ("query: " or "passage: ")

        Returns:
            EmbeddingResult with the embedding vector
        """
        self._ensure_initialized()

        # E5 requires prefix
        prefixed_text = f"{prefix}{text}"

        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                self._encode_sync,
                prefixed_text,
            )

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self._model_name,
                dimension=len(embedding),
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"E5 embedding failed: {e}",
            ) from e

    async def encode_query(self, query: str) -> EmbeddingResult:
        """Encode a query with 'query:' prefix."""
        return await self.encode(query, prefix="query: ")

    async def encode_passage(self, passage: str) -> EmbeddingResult:
        """Encode a passage with 'passage:' prefix."""
        return await self.encode(passage, prefix="passage: ")

    def _encode_sync(self, text: str) -> List[float]:
        """Synchronous encoding with mean pooling."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().numpy().tolist()

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="e5-embedding",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description="E5 embeddings for dense retrieval and semantic similarity",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[self.DEFAULT_MODEL, self.LARGE_MODEL],
        )


# ============================================================================
# MBART MULTILINGUAL TRANSLATION
# ============================================================================


@dataclass
class MBartTranslationResult:
    """Result from mBART translation."""

    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "model_name": self.model_name,
        }


class MBartTranslationProvider(HuggingFaceModelProvider):
    """
    mBART-50 provider for multilingual translation.

    Uses facebook/mbart-large-50-many-to-many-mmt for translation
    between any pair of 50 languages.

    Supported languages include:
    - en_XX (English), fr_XX (French), de_DE (German)
    - es_XX (Spanish), zh_CN (Chinese), ja_XX (Japanese)
    - And 44 more languages
    """

    DEFAULT_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

    # Language code mapping
    LANGUAGE_CODES = {
        "en": "en_XX", "fr": "fr_XX", "de": "de_DE", "es": "es_XX",
        "zh": "zh_CN", "ja": "ja_XX", "ko": "ko_KR", "ru": "ru_RU",
        "ar": "ar_AR", "hi": "hi_IN", "pt": "pt_XX", "it": "it_IT",
        "nl": "nl_XX", "pl": "pl_PL", "tr": "tr_TR", "vi": "vi_VN",
        "th": "th_TH", "id": "id_ID", "cs": "cs_CZ", "ro": "ro_RO",
        "sv": "sv_SE", "fi": "fi_FI", "da": "da_DK", "he": "he_IL",
        "uk": "uk_UA", "el": "el_GR", "hu": "hu_HU", "no": "no_NO",
        "ta": "ta_IN", "bn": "bn_IN", "ml": "ml_IN", "te": "te_IN",
        "gu": "gu_IN", "mr": "mr_IN", "ne": "ne_NP", "si": "si_LK",
        "my": "my_MM", "km": "km_KH", "af": "af_ZA", "sw": "sw_KE",
        "et": "et_EE", "lv": "lv_LV", "lt": "lt_LT", "sl": "sl_SI",
        "hr": "hr_HR", "mk": "mk_MK", "ka": "ka_GE", "az": "az_AZ",
        "kk": "kk_KZ", "mn": "mn_MN", "gl": "gl_ES", "xh": "xh_ZA",
    }

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        source_lang: str = "en",
        target_lang: str = "fr",
    ) -> None:
        """
        Initialize mBART translation provider.

        Args:
            config: Model configuration
            source_lang: Source language code (e.g., 'en', 'fr', 'de')
            target_lang: Target language code
        """
        super().__init__(
            config=config,
            model_name=self.DEFAULT_MODEL,
            model_type=ModelType.CLASSIFICATION,
        )
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._source_lang_code = self.LANGUAGE_CODES.get(source_lang, source_lang)
        self._target_lang_code = self.LANGUAGE_CODES.get(target_lang, target_lang)

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the mBART model and tokenizer."""
        try:
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = MBart50TokenizerFast.from_pretrained(self._model_name)
            tokenizer.src_lang = self._source_lang_code

            model = MBartForConditionalGeneration.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=len(self.LANGUAGE_CODES),
                labels=list(self.LANGUAGE_CODES.keys()),
                max_length=min(1024, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> MBartTranslationResult:
        """
        Translate text between languages.

        Args:
            text: Text to translate
            source_lang: Override source language
            target_lang: Override target language

        Returns:
            MBartTranslationResult with translation
        """
        self._ensure_initialized()

        src_lang = source_lang or self._source_lang
        tgt_lang = target_lang or self._target_lang

        try:
            loop = asyncio.get_event_loop()
            translated = await loop.run_in_executor(
                self._executor,
                self._translate_sync,
                text,
                src_lang,
                tgt_lang,
            )

            return MBartTranslationResult(
                source_text=text,
                translated_text=translated,
                source_lang=src_lang,
                target_lang=tgt_lang,
                model_name=self._model_name,
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"mBART translation failed: {e}",
            ) from e

    def _translate_sync(self, text: str, source_lang: str, target_lang: str) -> str:
        """Synchronous translation."""
        src_code = self.LANGUAGE_CODES.get(source_lang, source_lang)
        tgt_code = self.LANGUAGE_CODES.get(target_lang, target_lang)

        # Set source language
        self._tokenizer.src_lang = src_code

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate with target language
        generated = self._model.generate(
            **inputs,
            forced_bos_token_id=self._tokenizer.lang_code_to_id[tgt_code],
            max_length=self._model_info.max_length,
        )

        translated = self._tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0]

        return translated

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> List[MBartTranslationResult]:
        """
        Translate multiple texts.

        Args:
            texts: List of texts to translate
            source_lang: Override source language
            target_lang: Override target language

        Returns:
            List of MBartTranslationResult objects
        """
        results = []
        for text in texts:
            result = await self.translate(text, source_lang, target_lang)
            results.append(result)
        return results

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="mbart-translation",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description=f"mBART-50 translation ({self._source_lang} -> {self._target_lang})",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[self.DEFAULT_MODEL],
        )


# ============================================================================
# CROSS-ENCODER NLI PROVIDER
# ============================================================================


@dataclass
class CrossEncoderNLIResult:
    """Result from Cross-Encoder NLI inference."""

    sentence1: str
    sentence2: str
    label: str  # contradiction, entailment, neutral
    scores: Dict[str, float]
    confidence: float
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sentence1": self.sentence1,
            "sentence2": self.sentence2,
            "label": self.label,
            "scores": self.scores,
            "confidence": self.confidence,
            "model_name": self.model_name,
        }


class CrossEncoderNLIProvider(HuggingFaceModelProvider):
    """
    Cross-Encoder NLI provider for sentence pair classification.

    Uses cross-encoder/nli-deberta-v3-base for determining if two sentences
    contradict, entail, or are neutral to each other.

    Best for:
    - Sentence pair classification
    - Natural Language Inference
    - Semantic similarity (via entailment)
    """

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"
    LABELS = ["contradiction", "entailment", "neutral"]

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize Cross-Encoder NLI provider."""
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.CLASSIFICATION,
        )

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the Cross-Encoder model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            model = model.to(self._device)
            model.eval()

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=3,
                labels=self.LABELS,
                max_length=min(512, self._config.max_length),
            )

            return model, tokenizer, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def predict_nli(
        self,
        sentence1: str,
        sentence2: str,
    ) -> CrossEncoderNLIResult:
        """
        Predict NLI relationship between two sentences.

        Args:
            sentence1: First sentence (premise)
            sentence2: Second sentence (hypothesis)

        Returns:
            CrossEncoderNLIResult with prediction
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._predict_nli_sync,
                sentence1,
                sentence2,
            )
            return result

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"Cross-encoder NLI failed: {e}",
            ) from e

    def _predict_nli_sync(self, sentence1: str, sentence2: str) -> CrossEncoderNLIResult:
        """Synchronous NLI prediction."""
        import torch
        import torch.nn.functional as F

        inputs = self._tokenizer(
            sentence1,
            sentence2,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_info.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]

        scores = {label: probs[i].item() for i, label in enumerate(self.LABELS)}
        predicted_idx = probs.argmax().item()
        predicted_label = self.LABELS[predicted_idx]
        confidence = probs[predicted_idx].item()

        return CrossEncoderNLIResult(
            sentence1=sentence1,
            sentence2=sentence2,
            label=predicted_label,
            scores=scores,
            confidence=confidence,
            model_name=self._model_name,
        )

    async def predict_batch(
        self,
        sentence_pairs: List[Tuple[str, str]],
    ) -> List[CrossEncoderNLIResult]:
        """
        Predict NLI for multiple sentence pairs.

        Args:
            sentence_pairs: List of (sentence1, sentence2) tuples

        Returns:
            List of CrossEncoderNLIResult objects
        """
        results = []
        for s1, s2 in sentence_pairs:
            result = await self.predict_nli(s1, s2)
            results.append(result)
        return results

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="cross-encoder-nli",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description="Cross-Encoder NLI for sentence pair classification",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=True,
                quantization=self._config.use_quantization,
            ),
            supported_models=[self.DEFAULT_MODEL],
        )


# ============================================================================
# BLIP-2 VISION PROVIDER
# ============================================================================


@dataclass
class BLIP2Result:
    """Result from BLIP-2 inference."""

    image_path: str
    caption: Optional[str] = None
    answer: Optional[str] = None
    question: Optional[str] = None
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_path": self.image_path,
            "caption": self.caption,
            "answer": self.answer,
            "question": self.question,
            "model_name": self.model_name,
        }


class BLIP2Provider(HuggingFaceModelProvider):
    """
    BLIP-2 provider for advanced image understanding.

    Uses Salesforce/blip2-opt-2.7b for image captioning and VQA.
    Combines vision encoder with OPT-2.7B language model.

    Best for:
    - Image captioning
    - Visual Question Answering
    - Image-text understanding
    """

    DEFAULT_MODEL = "Salesforce/blip2-opt-2.7b"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        model_name: Optional[str] = None,
        load_in_8bit: bool = False,
    ) -> None:
        """
        Initialize BLIP-2 provider.

        Args:
            config: Model configuration
            model_name: Override model name
            load_in_8bit: Use 8-bit quantization for lower memory
        """
        super().__init__(
            config=config,
            model_name=model_name or self.DEFAULT_MODEL,
            model_type=ModelType.CLASSIFICATION,
        )
        self._load_in_8bit = load_in_8bit
        self._processor = None

    def _load_model_sync(self) -> Tuple[Any, Any, ModelInfo]:
        """Load the BLIP-2 model and processor."""
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
            import torch
        except ImportError as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="transformers library not installed",
            ) from e

        try:
            processor = Blip2Processor.from_pretrained(self._model_name)

            if self._load_in_8bit:
                model = Blip2ForConditionalGeneration.from_pretrained(
                    self._model_name,
                    load_in_8bit=True,
                    device_map="auto",
                )
            else:
                model = Blip2ForConditionalGeneration.from_pretrained(
                    self._model_name,
                    torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
                )
                model = model.to(self._device)

            model.eval()
            self._processor = processor

            model_info = ModelInfo(
                name=self._model_name,
                model_type=self._model_type,
                device=self._device,
                num_labels=0,
                labels=[],
                max_length=min(512, self._config.max_length),
            )

            return model, processor, model_info

        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=str(e),
            ) from e

    async def caption(self, image_path: str) -> BLIP2Result:
        """
        Generate caption for an image.

        Args:
            image_path: Path to the image file

        Returns:
            BLIP2Result with generated caption
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            caption = await loop.run_in_executor(
                self._executor,
                self._caption_sync,
                image_path,
            )

            return BLIP2Result(
                image_path=image_path,
                caption=caption,
                model_name=self._model_name,
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"BLIP-2 captioning failed: {e}",
            ) from e

    async def ask(self, image_path: str, question: str) -> BLIP2Result:
        """
        Answer a question about an image.

        Args:
            image_path: Path to the image file
            question: Question about the image

        Returns:
            BLIP2Result with answer
        """
        self._ensure_initialized()

        try:
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                self._executor,
                self._ask_sync,
                image_path,
                question,
            )

            return BLIP2Result(
                image_path=image_path,
                question=question,
                answer=answer,
                model_name=self._model_name,
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"BLIP-2 VQA failed: {e}",
            ) from e

    def _caption_sync(self, image_path: str) -> str:
        """Synchronous image captioning."""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        if self._device != "cpu" and not self._load_in_8bit:
            inputs = {k: v.to(self._device, torch.float16) for k, v in inputs.items()}
        elif not self._load_in_8bit:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_length=50)

        caption = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return caption

    def _ask_sync(self, image_path: str, question: str) -> str:
        """Synchronous visual question answering."""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        inputs = self._processor(images=image, text=question, return_tensors="pt")
        if self._device != "cpu" and not self._load_in_8bit:
            inputs = {k: v.to(self._device, torch.float16) for k, v in inputs.items()}
        elif not self._load_in_8bit:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_length=50)

        answer = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return answer

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="blip2",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description="BLIP-2 for image captioning and visual question answering",
            capabilities=ProviderCapabilities(
                batch_processing=False,
                gpu_support=True,
                quantization=True,
            ),
            supported_models=[self.DEFAULT_MODEL],
        )


# ============================================================================
# DEEP TRANSLATOR PROVIDER
# ============================================================================


@dataclass
class DeepTranslatorResult:
    """Result from deep_translator."""

    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    translator: str = "google"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "translator": self.translator,
        }


class DeepTranslatorProvider:
    """
    Deep Translator provider for online translation services.

    Uses the deep_translator library for Google Translate and other services.
    Requires internet connection.

    Best for:
    - Quick translations
    - No GPU required
    - Support for 100+ languages
    """

    SUPPORTED_TRANSLATORS = ["google", "mymemory", "linguee", "pons"]

    def __init__(
        self,
        source_lang: str = "auto",
        target_lang: str = "en",
        translator: str = "google",
    ) -> None:
        """
        Initialize Deep Translator provider.

        Args:
            source_lang: Source language code ('auto' for detection)
            target_lang: Target language code
            translator: Translator to use ('google', 'mymemory', etc.)
        """
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._translator_name = translator
        self._translator = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def initialize(self) -> None:
        """Initialize the translator."""
        if self._initialized:
            return

        try:
            from deep_translator import GoogleTranslator, MyMemoryTranslator
        except ImportError as e:
            raise ModelLoadError(
                model_name="deep_translator",
                reason="deep_translator library not installed. Run: pip install deep-translator",
            ) from e

        if self._translator_name == "google":
            self._translator = GoogleTranslator(
                source=self._source_lang,
                target=self._target_lang,
            )
        elif self._translator_name == "mymemory":
            self._translator = MyMemoryTranslator(
                source=self._source_lang,
                target=self._target_lang,
            )
        else:
            self._translator = GoogleTranslator(
                source=self._source_lang,
                target=self._target_lang,
            )

        self._initialized = True

    async def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> DeepTranslatorResult:
        """
        Translate text.

        Args:
            text: Text to translate
            source_lang: Override source language
            target_lang: Override target language

        Returns:
            DeepTranslatorResult with translation
        """
        if not self._initialized:
            await self.initialize()

        src = source_lang or self._source_lang
        tgt = target_lang or self._target_lang

        try:
            loop = asyncio.get_event_loop()
            translated = await loop.run_in_executor(
                self._executor,
                self._translate_sync,
                text,
                src,
                tgt,
            )

            return DeepTranslatorResult(
                source_text=text,
                translated_text=translated,
                source_lang=src,
                target_lang=tgt,
                translator=self._translator_name,
            )

        except Exception as e:
            raise ModelInferenceError(
                model_name="deep_translator",
                reason=f"Translation failed: {e}",
            ) from e

    def _translate_sync(self, text: str, source_lang: str, target_lang: str) -> str:
        """Synchronous translation."""
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source=source_lang, target=target_lang)
        return translator.translate(text)

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> List[DeepTranslatorResult]:
        """
        Translate multiple texts.

        Args:
            texts: List of texts to translate
            source_lang: Override source language
            target_lang: Override target language

        Returns:
            List of DeepTranslatorResult objects
        """
        if not self._initialized:
            await self.initialize()

        results = []
        for text in texts:
            result = await self.translate(text, source_lang, target_lang)
            results.append(result)
        return results

    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """Get supported languages."""
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator().get_supported_languages(as_dict=True)
        except ImportError:
            return {}

    @property
    def is_initialized(self) -> bool:
        """Check if initialized."""
        return self._initialized

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="deep-translator",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description="Deep Translator for Google Translate and other services",
            capabilities=ProviderCapabilities(
                batch_processing=True,
                gpu_support=False,
                quantization=False,
            ),
            supported_models=self.SUPPORTED_TRANSLATORS,
        )


# ============================================================================
# LLAVA VISION PROVIDER (OLLAMA)
# ============================================================================


@dataclass
class LLaVAResult:
    """Result from LLaVA inference."""

    image_path: str
    response: str
    prompt: Optional[str] = None
    model_name: str = "llava"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_path": self.image_path,
            "response": self.response,
            "prompt": self.prompt,
            "model_name": self.model_name,
        }


class LLaVAProvider:
    """
    LLaVA provider for image understanding via Ollama.

    Uses Ollama's LLaVA model for image understanding and VQA.
    Requires Ollama to be running locally.

    Best for:
    - Local image understanding
    - Visual question answering
    - Image description
    """

    DEFAULT_MODEL = "llava"

    def __init__(
        self,
        model_name: str = "llava",
        base_url: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize LLaVA provider.

        Args:
            model_name: Ollama model name (llava, llava:13b, etc.)
            base_url: Ollama server URL
        """
        self._model_name = model_name
        self._base_url = base_url
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def initialize(self) -> None:
        """Initialize and verify Ollama connection."""
        if self._initialized:
            return

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self._base_url}/api/tags")
                if response.status_code != 200:
                    raise ConnectionError("Cannot connect to Ollama server")
        except ImportError:
            raise ModelLoadError(
                model_name=self._model_name,
                reason="httpx library not installed",
            )
        except Exception as e:
            raise ModelLoadError(
                model_name=self._model_name,
                reason=f"Cannot connect to Ollama: {e}",
            )

        self._initialized = True

    async def analyze(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
    ) -> LLaVAResult:
        """
        Analyze an image with LLaVA.

        Args:
            image_path: Path to the image file
            prompt: Prompt/question about the image

        Returns:
            LLaVAResult with response
        """
        if not self._initialized:
            await self.initialize()

        try:
            import httpx
            import base64

            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model_name,
                        "prompt": prompt,
                        "images": [image_data],
                        "stream": False,
                    },
                )

                if response.status_code != 200:
                    raise ModelInferenceError(
                        model_name=self._model_name,
                        reason=f"Ollama returned status {response.status_code}",
                    )

                result = response.json()
                return LLaVAResult(
                    image_path=image_path,
                    response=result.get("response", ""),
                    prompt=prompt,
                    model_name=self._model_name,
                )

        except Exception as e:
            raise ModelInferenceError(
                model_name=self._model_name,
                reason=f"LLaVA inference failed: {e}",
            ) from e

    async def ask(self, image_path: str, question: str) -> LLaVAResult:
        """
        Ask a question about an image.

        Args:
            image_path: Path to the image file
            question: Question about the image

        Returns:
            LLaVAResult with answer
        """
        return await self.analyze(image_path, prompt=question)

    @property
    def is_initialized(self) -> bool:
        """Check if initialized."""
        return self._initialized

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="llava",
            version="1.0.0",
            provider_type=ProviderType.MODEL,
            description="LLaVA for image understanding via Ollama",
            capabilities=ProviderCapabilities(
                batch_processing=False,
                gpu_support=True,
                quantization=True,
            ),
            supported_models=["llava", "llava:7b", "llava:13b", "llava:34b"],
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def clear_model_cache() -> None:
    """Clear the global model cache."""
    global _model_cache
    _model_cache.clear()


def get_cached_models() -> List[str]:
    """Get list of cached model keys."""
    return list(_model_cache.keys())


def get_available_sentiment_models() -> Dict[str, str]:
    """Get all available sentiment models."""
    return {
        "twitter-roberta-latest": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "twitter-roberta": "cardiffnlp/twitter-roberta-base-sentiment",
        "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
        "nlptown-multilingual": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bertweet": "finiteautomata/bertweet-base-sentiment-analysis",
        "siebert-large": "siebert/sentiment-roberta-large-english",
        "multilingual": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        "xlm-roberta": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "finbert": "ProsusAI/finbert",
        "finbert-tone": "yiyanghkust/finbert-tone",
    }


def get_available_emotion_models() -> Dict[str, str]:
    """Get all available emotion models."""
    return {
        "go-emotions": "SamLowe/roberta-base-go_emotions",
        "distilbert-emotion": "bhadresh-savani/distilbert-base-uncased-emotion",
        "hartmann-emotion": "j-hartmann/emotion-english-distilroberta-base",
        "twitter-emotion": "cardiffnlp/twitter-roberta-base-emotion",
        "t5-emotion": "mrm8488/t5-base-finetuned-emotion",
    }


def get_available_absa_models() -> Dict[str, str]:
    """Get all available ABSA models."""
    return {
        "deberta-absa-base": "yangheng/deberta-v3-base-absa-v1.1",
        "deberta-absa-large": "yangheng/deberta-v3-large-absa-v1.1",
        "instruct-absa": "kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined",
    }


def get_available_embedding_models() -> Dict[str, str]:
    """Get all available embedding models."""
    return {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "e5-base": "intfloat/e5-base-v2",
        "e5-large": "intfloat/e5-large-v2",
    }


def get_available_domain_models() -> Dict[str, str]:
    """Get all available domain-specific models."""
    return {
        "legal-bert": "nlpaueb/legal-bert-base-uncased",
        "scibert": "allenai/scibert_scivocab_uncased",
        "finbert": "ProsusAI/finbert",
        "finbert-tone": "yiyanghkust/finbert-tone",
    }


def get_available_zero_shot_models() -> Dict[str, str]:
    """Get all available zero-shot classification models."""
    return {
        "bart-mnli": "facebook/bart-large-mnli",
        "deberta-nli": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    }


def get_available_translation_models() -> Dict[str, str]:
    """Get all available translation models."""
    return OpusMTTranslationProvider.MODELS.copy()


def get_available_whisper_models() -> Dict[str, str]:
    """Get all available Whisper models."""
    return WhisperProvider.MODELS.copy()


# ============================================================================
# PROVIDER REGISTRATION
# ============================================================================

# Register providers
register_provider("huggingface", ProviderType.MODEL, HuggingFaceModelProvider)
register_provider("huggingface-sentiment", ProviderType.MODEL, SentimentModelProvider)
register_provider("huggingface-emotion", ProviderType.MODEL, EmotionModelProvider)
register_provider("distilbert-sentiment", ProviderType.MODEL, DistilBertSentimentProvider)
register_provider("siebert-sentiment", ProviderType.MODEL, SiebertSentimentProvider)
register_provider("twitter-sentiment", ProviderType.MODEL, TwitterSentimentProvider)
register_provider("multilingual-sentiment", ProviderType.MODEL, MultilingualSentimentProvider)
register_provider("twitter-emotion", ProviderType.MODEL, TwitterEmotionProvider)
register_provider("t5-emotion", ProviderType.MODEL, T5EmotionProvider)
register_provider("deberta-absa", ProviderType.MODEL, DeBERTaABSAProvider)
register_provider("instruct-absa", ProviderType.MODEL, InstructABSAProvider)
# New providers
register_provider("xlm-roberta-sentiment", ProviderType.MODEL, XLMRobertaSentimentProvider)
register_provider("finbert", ProviderType.MODEL, FinBERTProvider)
register_provider("finbert-tone", ProviderType.MODEL, FinBERTToneProvider)
register_provider("zero-shot-classification", ProviderType.MODEL, ZeroShotClassificationProvider)
register_provider("sentence-embedding", ProviderType.MODEL, SentenceEmbeddingProvider)
register_provider("whisper", ProviderType.MODEL, WhisperProvider)
register_provider("blip-captioning", ProviderType.MODEL, BLIPCaptioningProvider)
register_provider("opus-mt-translation", ProviderType.MODEL, OpusMTTranslationProvider)
# Stage 16 providers
register_provider("legal-bert", ProviderType.MODEL, LegalBERTProvider)
register_provider("scibert", ProviderType.MODEL, SciBERTProvider)
register_provider("deberta-nli", ProviderType.MODEL, DeBERTaNLIProvider)
register_provider("bge-embedding", ProviderType.MODEL, BGEEmbeddingProvider)
register_provider("e5-embedding", ProviderType.MODEL, E5EmbeddingProvider)
register_provider("mbart-translation", ProviderType.MODEL, MBartTranslationProvider)
# Stage 16 additional providers
register_provider("cross-encoder-nli", ProviderType.MODEL, CrossEncoderNLIProvider)
register_provider("blip2", ProviderType.MODEL, BLIP2Provider)
register_provider("deep-translator", ProviderType.MODEL, DeepTranslatorProvider)
register_provider("llava", ProviderType.MODEL, LLaVAProvider)
