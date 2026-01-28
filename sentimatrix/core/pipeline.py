"""
Sentimatrix Pipeline Orchestration Module

Provides a flexible pipeline framework for chaining processing steps
with progress tracking, error handling, and async support.

Features:
- Step chaining with dependency management
- Progress callbacks for monitoring
- Error handling with retry support
- Async and sync step execution
- Pipeline context for shared state
- Conditional step execution
- Parallel step execution

Example:
    >>> pipeline = Pipeline("sentiment_analysis")
    >>> pipeline.add_step(ScrapeStep())
    >>> pipeline.add_step(PreprocessStep())
    >>> pipeline.add_step(AnalyzeStep())
    >>> result = await pipeline.run(url="https://example.com")
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

from sentimatrix.core.exceptions import (
    PipelineError,
    PipelineStateError,
    PipelineStepError,
)
from sentimatrix.core.logger import get_logger, LogContext

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class PipelineState(str, Enum):
    """Pipeline execution states."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepState(str, Enum):
    """Individual step execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""

    step_name: str
    state: StepState
    output: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if step completed successfully."""
        return self.state == StepState.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_name": self.step_name,
            "state": self.state.value,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "error": str(self.error) if self.error else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
        }


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution."""

    pipeline_id: str
    pipeline_name: str
    state: PipelineState
    step_results: List[StepResult] = field(default_factory=list)
    output: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.state == PipelineState.COMPLETED

    @property
    def failed_steps(self) -> List[StepResult]:
        """Get list of failed steps."""
        return [s for s in self.step_results if s.state == StepState.FAILED]

    @property
    def completed_steps(self) -> List[StepResult]:
        """Get list of completed steps."""
        return [s for s in self.step_results if s.state == StepState.COMPLETED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "state": self.state.value,
            "success": self.success,
            "total_duration_ms": self.total_duration_ms,
            "steps_completed": len(self.completed_steps),
            "steps_failed": len(self.failed_steps),
            "step_results": [s.to_dict() for s in self.step_results],
            "error": str(self.error) if self.error else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
        }


class PipelineContext:
    """
    Shared context for pipeline execution.

    Provides a way for steps to share data and state during execution.
    Thread-safe for concurrent access.
    """

    def __init__(self, initial_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize pipeline context.

        Args:
            initial_data: Initial data to populate context
        """
        self._data: Dict[str, Any] = initial_data.copy() if initial_data else {}
        self._metadata: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in context."""
        self._data[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        """Update context with multiple values."""
        self._data.update(data)

    def delete(self, key: str) -> None:
        """Delete key from context."""
        self._data.pop(key, None)

    def has(self, key: str) -> bool:
        """Check if key exists in context."""
        return key in self._data

    def keys(self) -> List[str]:
        """Get all keys in context."""
        return list(self._data.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Get copy of all context data."""
        return self._data.copy()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get context metadata."""
        return self._metadata

    async def async_get(self, key: str, default: Any = None) -> Any:
        """Thread-safe async get."""
        async with self._lock:
            return self._data.get(key, default)

    async def async_set(self, key: str, value: Any) -> None:
        """Thread-safe async set."""
        async with self._lock:
            self._data[key] = value


# Callback type definitions
ProgressCallback = Callable[[str, int, int, Optional[str]], None]
StepCallback = Callable[[str, StepState, Optional[StepResult]], None]
ErrorCallback = Callable[[str, Exception, int], bool]  # Returns True to retry


@dataclass
class StepConfig:
    """Configuration for a pipeline step."""

    max_retries: int = 0
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    timeout: Optional[float] = None
    skip_on_failure: bool = False
    condition: Optional[Callable[[PipelineContext], bool]] = None
    tags: Set[str] = field(default_factory=set)


class PipelineStep(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for pipeline steps.

    Each step processes input and produces output that can be passed
    to subsequent steps.

    Example:
        >>> class PreprocessStep(PipelineStep[str, List[str]]):
        ...     async def execute(self, input_data, context):
        ...         return input_data.split()
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[StepConfig] = None,
    ) -> None:
        """
        Initialize pipeline step.

        Args:
            name: Step name (defaults to class name)
            config: Step configuration
        """
        self._name = name or self.__class__.__name__
        self._config = config or StepConfig()
        self._logger = get_logger(f"sentimatrix.pipeline.{self._name}")

    @property
    def name(self) -> str:
        """Get step name."""
        return self._name

    @property
    def config(self) -> StepConfig:
        """Get step configuration."""
        return self._config

    @abstractmethod
    async def execute(
        self,
        input_data: InputT,
        context: PipelineContext,
    ) -> OutputT:
        """
        Execute the step.

        Args:
            input_data: Input data from previous step
            context: Shared pipeline context

        Returns:
            Output data to pass to next step
        """
        pass

    async def validate_input(self, input_data: InputT, context: PipelineContext) -> bool:
        """
        Validate input before execution.

        Override to add custom validation logic.

        Args:
            input_data: Input data to validate
            context: Pipeline context

        Returns:
            True if input is valid
        """
        return True

    async def on_start(self, context: PipelineContext) -> None:
        """Hook called before step execution."""
        pass

    async def on_complete(self, output: OutputT, context: PipelineContext) -> None:
        """Hook called after successful execution."""
        pass

    async def on_error(self, error: Exception, context: PipelineContext) -> None:
        """Hook called on execution error."""
        pass

    def should_skip(self, context: PipelineContext) -> bool:
        """
        Check if step should be skipped.

        Args:
            context: Pipeline context

        Returns:
            True if step should be skipped
        """
        if self._config.condition:
            return not self._config.condition(context)
        return False


class FunctionStep(PipelineStep[InputT, OutputT]):
    """
    Pipeline step that wraps a function.

    Convenience class for creating steps from functions.

    Example:
        >>> step = FunctionStep("lowercase", lambda x, ctx: x.lower())
    """

    def __init__(
        self,
        name: str,
        func: Callable[[InputT, PipelineContext], OutputT],
        config: Optional[StepConfig] = None,
    ) -> None:
        """
        Initialize function step.

        Args:
            name: Step name
            func: Function to execute (can be sync or async)
            config: Step configuration
        """
        super().__init__(name, config)
        self._func = func

    async def execute(self, input_data: InputT, context: PipelineContext) -> OutputT:
        """Execute the wrapped function."""
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(input_data, context)
        return self._func(input_data, context)


class Pipeline:
    """
    Pipeline orchestrator for chaining processing steps.

    Manages step execution, error handling, retries, and progress tracking.

    Example:
        >>> pipeline = Pipeline("analysis")
        >>> pipeline.add_step(FetchStep())
        >>> pipeline.add_step(ProcessStep())
        >>> pipeline.add_step(OutputStep())
        >>>
        >>> # With progress callback
        >>> pipeline.on_progress(lambda name, cur, total, msg: print(f"{cur}/{total}: {msg}"))
        >>>
        >>> result = await pipeline.run(initial_data)
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        stop_on_error: bool = True,
    ) -> None:
        """
        Initialize pipeline.

        Args:
            name: Pipeline name
            description: Pipeline description
            stop_on_error: Stop execution on first error
        """
        self._name = name
        self._description = description
        self._stop_on_error = stop_on_error
        self._steps: List[PipelineStep] = []
        self._state = PipelineState.PENDING
        self._current_step_index = 0
        self._pipeline_id: Optional[str] = None

        # Callbacks
        self._progress_callbacks: List[ProgressCallback] = []
        self._step_callbacks: List[StepCallback] = []
        self._error_callbacks: List[ErrorCallback] = []

        self._logger = get_logger(f"sentimatrix.pipeline.{name}")

    @property
    def name(self) -> str:
        """Get pipeline name."""
        return self._name

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    @property
    def steps(self) -> List[PipelineStep]:
        """Get list of steps."""
        return self._steps.copy()

    @property
    def step_count(self) -> int:
        """Get number of steps."""
        return len(self._steps)

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """
        Add a step to the pipeline.

        Args:
            step: Step to add

        Returns:
            Self for chaining
        """
        self._steps.append(step)
        self._logger.debug(f"Added step: {step.name}", step_count=len(self._steps))
        return self

    def add_steps(self, *steps: PipelineStep) -> "Pipeline":
        """
        Add multiple steps to the pipeline.

        Args:
            *steps: Steps to add

        Returns:
            Self for chaining
        """
        for step in steps:
            self.add_step(step)
        return self

    def insert_step(self, index: int, step: PipelineStep) -> "Pipeline":
        """
        Insert a step at a specific position.

        Args:
            index: Position to insert at
            step: Step to insert

        Returns:
            Self for chaining
        """
        self._steps.insert(index, step)
        return self

    def remove_step(self, name: str) -> bool:
        """
        Remove a step by name.

        Args:
            name: Name of step to remove

        Returns:
            True if step was removed
        """
        for i, step in enumerate(self._steps):
            if step.name == name:
                self._steps.pop(i)
                return True
        return False

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """
        Get a step by name.

        Args:
            name: Step name

        Returns:
            Step if found, None otherwise
        """
        for step in self._steps:
            if step.name == name:
                return step
        return None

    def on_progress(self, callback: ProgressCallback) -> "Pipeline":
        """
        Register progress callback.

        Callback signature: (step_name, current, total, message) -> None

        Args:
            callback: Progress callback function

        Returns:
            Self for chaining
        """
        self._progress_callbacks.append(callback)
        return self

    def on_step(self, callback: StepCallback) -> "Pipeline":
        """
        Register step state change callback.

        Callback signature: (step_name, state, result) -> None

        Args:
            callback: Step callback function

        Returns:
            Self for chaining
        """
        self._step_callbacks.append(callback)
        return self

    def on_error(self, callback: ErrorCallback) -> "Pipeline":
        """
        Register error callback.

        Callback signature: (step_name, error, retry_count) -> should_retry

        Args:
            callback: Error callback function

        Returns:
            Self for chaining
        """
        self._error_callbacks.append(callback)
        return self

    def _notify_progress(
        self,
        step_name: str,
        current: int,
        total: int,
        message: Optional[str] = None,
    ) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(step_name, current, total, message)
            except Exception as e:
                self._logger.warning(f"Progress callback error: {e}")

    def _notify_step(
        self,
        step_name: str,
        state: StepState,
        result: Optional[StepResult] = None,
    ) -> None:
        """Notify all step callbacks."""
        for callback in self._step_callbacks:
            try:
                callback(step_name, state, result)
            except Exception as e:
                self._logger.warning(f"Step callback error: {e}")

    def _should_retry(self, step_name: str, error: Exception, retry_count: int) -> bool:
        """Check if step should be retried via callbacks."""
        for callback in self._error_callbacks:
            try:
                if callback(step_name, error, retry_count):
                    return True
            except Exception as e:
                self._logger.warning(f"Error callback error: {e}")
        return False

    async def _execute_step(
        self,
        step: PipelineStep,
        input_data: Any,
        context: PipelineContext,
        step_index: int,
    ) -> StepResult:
        """Execute a single step with retry logic."""
        step_name = step.name
        config = step.config
        retry_count = 0
        last_error: Optional[Exception] = None

        # Check if step should be skipped
        if step.should_skip(context):
            self._logger.info(f"Skipping step: {step_name}", reason="condition_not_met")
            self._notify_step(step_name, StepState.SKIPPED)
            return StepResult(
                step_name=step_name,
                state=StepState.SKIPPED,
                metadata={"reason": "condition_not_met"},
            )

        while retry_count <= config.max_retries:
            start_time = datetime.now(timezone.utc)
            start_perf = time.perf_counter()

            try:
                # Notify step starting
                state = StepState.RETRYING if retry_count > 0 else StepState.RUNNING
                self._notify_step(step_name, state)
                self._notify_progress(
                    step_name,
                    step_index + 1,
                    len(self._steps),
                    f"Running {step_name}" + (f" (retry {retry_count})" if retry_count else ""),
                )

                # Validate input
                if not await step.validate_input(input_data, context):
                    raise PipelineStepError(step_name, "Input validation failed")

                # Execute pre-hook
                await step.on_start(context)

                # Execute step (with optional timeout)
                if config.timeout:
                    output = await asyncio.wait_for(
                        step.execute(input_data, context),
                        timeout=config.timeout,
                    )
                else:
                    output = await step.execute(input_data, context)

                # Execute post-hook
                await step.on_complete(output, context)

                end_time = datetime.now(timezone.utc)
                duration_ms = (time.perf_counter() - start_perf) * 1000

                self._logger.info(
                    f"Step completed: {step_name}",
                    duration_ms=round(duration_ms, 2),
                    retry_count=retry_count,
                )

                result = StepResult(
                    step_name=step_name,
                    state=StepState.COMPLETED,
                    output=output,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    retry_count=retry_count,
                )

                self._notify_step(step_name, StepState.COMPLETED, result)
                return result

            except asyncio.TimeoutError as e:
                last_error = PipelineStepError(step_name, f"Timeout after {config.timeout}s")
                await step.on_error(last_error, context)

            except Exception as e:
                last_error = e
                await step.on_error(e, context)
                self._logger.warning(
                    f"Step failed: {step_name}",
                    error=str(e),
                    retry_count=retry_count,
                )

            # Check if we should retry
            if retry_count < config.max_retries or self._should_retry(step_name, last_error, retry_count):
                retry_count += 1
                delay = config.retry_delay * (config.retry_backoff ** (retry_count - 1))
                self._logger.info(
                    f"Retrying step: {step_name}",
                    retry_count=retry_count,
                    delay=delay,
                )
                await asyncio.sleep(delay)
            else:
                break

        # Step failed after all retries
        end_time = datetime.now(timezone.utc)
        duration_ms = (time.perf_counter() - start_perf) * 1000

        result = StepResult(
            step_name=step_name,
            state=StepState.FAILED,
            error=last_error,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            retry_count=retry_count,
        )

        self._notify_step(step_name, StepState.FAILED, result)
        return result

    async def run(
        self,
        input_data: Any = None,
        context: Optional[PipelineContext] = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        Run the pipeline.

        Args:
            input_data: Initial input data
            context: Pipeline context (created if not provided)
            **kwargs: Additional context data

        Returns:
            PipelineResult with execution details

        Raises:
            PipelineStateError: If pipeline is in invalid state
            PipelineError: If pipeline execution fails
        """
        if self._state == PipelineState.RUNNING:
            raise PipelineStateError(self._name, self._state.value, "PENDING")

        if not self._steps:
            raise PipelineError(f"Pipeline '{self._name}' has no steps")

        # Initialize execution
        self._pipeline_id = str(uuid.uuid4())
        self._state = PipelineState.RUNNING
        self._current_step_index = 0

        # Create context
        context = context or PipelineContext()
        if kwargs:
            context.update(kwargs)

        # Execution tracking
        start_time = datetime.now(timezone.utc)
        start_perf = time.perf_counter()
        step_results: List[StepResult] = []
        current_data = input_data
        final_error: Optional[Exception] = None

        self._logger.info(
            f"Pipeline started: {self._name}",
            pipeline_id=self._pipeline_id,
            step_count=len(self._steps),
        )

        with LogContext(pipeline_id=self._pipeline_id, pipeline_name=self._name):
            try:
                for i, step in enumerate(self._steps):
                    self._current_step_index = i

                    # Execute step
                    result = await self._execute_step(step, current_data, context, i)
                    step_results.append(result)

                    if result.state == StepState.COMPLETED:
                        # Pass output to next step
                        current_data = result.output
                    elif result.state == StepState.FAILED:
                        if self._stop_on_error and not step.config.skip_on_failure:
                            final_error = result.error
                            break
                        # Continue with previous data if skip_on_failure is set
                    # SKIPPED steps don't change current_data

                # Determine final state
                if final_error:
                    self._state = PipelineState.FAILED
                else:
                    self._state = PipelineState.COMPLETED

            except asyncio.CancelledError:
                self._state = PipelineState.CANCELLED
                final_error = PipelineError(f"Pipeline '{self._name}' was cancelled")

            except Exception as e:
                self._state = PipelineState.FAILED
                final_error = e
                self._logger.exception(f"Pipeline error: {self._name}")

        end_time = datetime.now(timezone.utc)
        total_duration_ms = (time.perf_counter() - start_perf) * 1000

        result = PipelineResult(
            pipeline_id=self._pipeline_id,
            pipeline_name=self._name,
            state=self._state,
            step_results=step_results,
            output=current_data if self._state == PipelineState.COMPLETED else None,
            error=final_error,
            start_time=start_time,
            end_time=end_time,
            total_duration_ms=total_duration_ms,
            metadata=context.metadata,
        )

        self._logger.log_performance(
            f"pipeline_{self._name}",
            duration_ms=total_duration_ms,
            success=result.success,
            steps_completed=len(result.completed_steps),
            steps_failed=len(result.failed_steps),
        )

        # Reset state for potential rerun
        self._state = PipelineState.PENDING

        return result

    async def run_from_step(
        self,
        step_name: str,
        input_data: Any = None,
        context: Optional[PipelineContext] = None,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        Run pipeline starting from a specific step.

        Args:
            step_name: Name of step to start from
            input_data: Input data for the step
            context: Pipeline context
            **kwargs: Additional context data

        Returns:
            PipelineResult

        Raises:
            PipelineError: If step not found
        """
        # Find step index
        step_index = None
        for i, step in enumerate(self._steps):
            if step.name == step_name:
                step_index = i
                break

        if step_index is None:
            raise PipelineError(f"Step '{step_name}' not found in pipeline '{self._name}'")

        # Create temporary pipeline with remaining steps
        temp_pipeline = Pipeline(
            f"{self._name}_from_{step_name}",
            stop_on_error=self._stop_on_error,
        )
        for step in self._steps[step_index:]:
            temp_pipeline.add_step(step)

        # Copy callbacks
        temp_pipeline._progress_callbacks = self._progress_callbacks
        temp_pipeline._step_callbacks = self._step_callbacks
        temp_pipeline._error_callbacks = self._error_callbacks

        return await temp_pipeline.run(input_data, context, **kwargs)

    def cancel(self) -> None:
        """Cancel pipeline execution."""
        if self._state == PipelineState.RUNNING:
            self._state = PipelineState.CANCELLED
            self._logger.info(f"Pipeline cancelled: {self._name}")

    def reset(self) -> None:
        """Reset pipeline state."""
        self._state = PipelineState.PENDING
        self._current_step_index = 0
        self._pipeline_id = None


class ParallelSteps(PipelineStep[Any, Dict[str, Any]]):
    """
    Execute multiple steps in parallel.

    All steps receive the same input and their outputs are collected
    into a dictionary keyed by step name.

    Example:
        >>> parallel = ParallelSteps("fetch_all", [
        ...     FetchAmazonStep(),
        ...     FetchSteamStep(),
        ...     FetchYouTubeStep(),
        ... ])
        >>> pipeline.add_step(parallel)
    """

    def __init__(
        self,
        name: str,
        steps: List[PipelineStep],
        config: Optional[StepConfig] = None,
        fail_fast: bool = False,
    ) -> None:
        """
        Initialize parallel steps.

        Args:
            name: Step name
            steps: Steps to execute in parallel
            config: Step configuration
            fail_fast: Cancel remaining steps on first failure
        """
        super().__init__(name, config)
        self._parallel_steps = steps
        self._fail_fast = fail_fast

    async def execute(
        self,
        input_data: Any,
        context: PipelineContext,
    ) -> Dict[str, Any]:
        """Execute all steps in parallel."""
        tasks = []
        for step in self._parallel_steps:
            task = asyncio.create_task(
                step.execute(input_data, context),
                name=step.name,
            )
            tasks.append((step.name, task))

        results: Dict[str, Any] = {}
        errors: Dict[str, Exception] = {}

        if self._fail_fast:
            # Use gather with return_exceptions=False for fail-fast behavior
            try:
                done, pending = await asyncio.wait(
                    [t for _, t in tasks],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

                # Collect results
                for name, task in tasks:
                    if task.done() and not task.cancelled():
                        if task.exception():
                            errors[name] = task.exception()
                        else:
                            results[name] = task.result()

                if errors:
                    raise PipelineStepError(
                        self._name,
                        f"Parallel step failures: {list(errors.keys())}",
                    )

            except Exception as e:
                # Cancel all remaining tasks
                for _, task in tasks:
                    if not task.done():
                        task.cancel()
                raise
        else:
            # Wait for all tasks to complete
            await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

            for name, task in tasks:
                if task.exception():
                    errors[name] = task.exception()
                else:
                    results[name] = task.result()

        context.set(f"{self._name}_errors", errors)
        return results


class ConditionalStep(PipelineStep[Any, Any]):
    """
    Conditionally execute one of two steps based on a condition.

    Example:
        >>> conditional = ConditionalStep(
        ...     "check_cache",
        ...     condition=lambda ctx: ctx.get("use_cache"),
        ...     if_true=CachedStep(),
        ...     if_false=FetchStep(),
        ... )
    """

    def __init__(
        self,
        name: str,
        condition: Callable[[PipelineContext], bool],
        if_true: PipelineStep,
        if_false: Optional[PipelineStep] = None,
        config: Optional[StepConfig] = None,
    ) -> None:
        """
        Initialize conditional step.

        Args:
            name: Step name
            condition: Condition function
            if_true: Step to execute if condition is True
            if_false: Step to execute if condition is False (optional)
            config: Step configuration
        """
        super().__init__(name, config)
        self._condition = condition
        self._if_true = if_true
        self._if_false = if_false

    async def execute(self, input_data: Any, context: PipelineContext) -> Any:
        """Execute the appropriate step based on condition."""
        if self._condition(context):
            self._logger.debug(f"Condition true, executing: {self._if_true.name}")
            return await self._if_true.execute(input_data, context)
        elif self._if_false:
            self._logger.debug(f"Condition false, executing: {self._if_false.name}")
            return await self._if_false.execute(input_data, context)
        else:
            self._logger.debug("Condition false, no else step, passing through")
            return input_data


# Convenience function for creating pipelines
def create_pipeline(
    name: str,
    steps: Optional[List[PipelineStep]] = None,
    **kwargs: Any,
) -> Pipeline:
    """
    Create a pipeline with optional steps.

    Args:
        name: Pipeline name
        steps: Initial steps to add
        **kwargs: Additional pipeline configuration

    Returns:
        Configured Pipeline instance
    """
    pipeline = Pipeline(name, **kwargs)
    if steps:
        pipeline.add_steps(*steps)
    return pipeline


__all__ = [
    "PipelineState",
    "StepState",
    "StepResult",
    "PipelineResult",
    "PipelineContext",
    "StepConfig",
    "PipelineStep",
    "FunctionStep",
    "Pipeline",
    "ParallelSteps",
    "ConditionalStep",
    "create_pipeline",
    "ProgressCallback",
    "StepCallback",
    "ErrorCallback",
]
