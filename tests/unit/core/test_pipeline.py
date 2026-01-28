"""
Unit tests for the Pipeline orchestration module.

Tests cover:
- Pipeline creation and configuration
- Step execution and chaining
- Progress callbacks
- Error handling and retries
- Parallel and conditional steps
- Context management
"""

import asyncio
import pytest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, AsyncMock

from sentimatrix.core.pipeline import (
    Pipeline,
    PipelineContext,
    PipelineResult,
    PipelineState,
    PipelineStep,
    FunctionStep,
    ParallelSteps,
    ConditionalStep,
    StepConfig,
    StepResult,
    StepState,
    create_pipeline,
)
from sentimatrix.core.exceptions import PipelineError, PipelineStateError, PipelineStepError


# Test Step Implementations
class AddStep(PipelineStep[int, int]):
    """Test step that adds a value."""

    def __init__(self, value: int, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    async def execute(self, input_data: int, context: PipelineContext) -> int:
        return input_data + self.value


class MultiplyStep(PipelineStep[int, int]):
    """Test step that multiplies by a value."""

    def __init__(self, value: int, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    async def execute(self, input_data: int, context: PipelineContext) -> int:
        return input_data * self.value


class FailingStep(PipelineStep[Any, Any]):
    """Test step that always fails."""

    def __init__(self, error_msg: str = "Test failure", **kwargs):
        super().__init__(**kwargs)
        self.error_msg = error_msg

    async def execute(self, input_data: Any, context: PipelineContext) -> Any:
        raise ValueError(self.error_msg)


class ContextStep(PipelineStep[Any, Any]):
    """Test step that reads/writes context."""

    def __init__(self, key: str, value: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.value = value

    async def execute(self, input_data: Any, context: PipelineContext) -> Any:
        if self.value is not None:
            context.set(self.key, self.value)
        return context.get(self.key, input_data)


class SlowStep(PipelineStep[Any, Any]):
    """Test step with configurable delay."""

    def __init__(self, delay: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    async def execute(self, input_data: Any, context: PipelineContext) -> Any:
        await asyncio.sleep(self.delay)
        return input_data


class RetryableStep(PipelineStep[Any, Any]):
    """Test step that fails a certain number of times before succeeding."""

    def __init__(self, fail_count: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.fail_count = fail_count
        self.attempt = 0

    async def execute(self, input_data: Any, context: PipelineContext) -> Any:
        self.attempt += 1
        if self.attempt <= self.fail_count:
            raise ValueError(f"Attempt {self.attempt} failed")
        return input_data


class TestStepState:
    """Tests for StepState enum."""

    def test_step_states(self):
        """Test all step states exist."""
        assert StepState.PENDING == "pending"
        assert StepState.RUNNING == "running"
        assert StepState.COMPLETED == "completed"
        assert StepState.FAILED == "failed"
        assert StepState.SKIPPED == "skipped"
        assert StepState.RETRYING == "retrying"


class TestPipelineState:
    """Tests for PipelineState enum."""

    def test_pipeline_states(self):
        """Test all pipeline states exist."""
        assert PipelineState.PENDING == "pending"
        assert PipelineState.RUNNING == "running"
        assert PipelineState.COMPLETED == "completed"
        assert PipelineState.FAILED == "failed"
        assert PipelineState.CANCELLED == "cancelled"


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_creation(self):
        """Test creating a step result."""
        result = StepResult(
            step_name="test_step",
            state=StepState.COMPLETED,
            output=42,
        )
        assert result.step_name == "test_step"
        assert result.state == StepState.COMPLETED
        assert result.output == 42
        assert result.success is True

    def test_step_result_failed(self):
        """Test failed step result."""
        error = ValueError("Test error")
        result = StepResult(
            step_name="failing_step",
            state=StepState.FAILED,
            error=error,
        )
        assert result.success is False
        assert result.error == error

    def test_step_result_to_dict(self):
        """Test step result serialization."""
        result = StepResult(
            step_name="test_step",
            state=StepState.COMPLETED,
            output="result",
            duration_ms=150.5,
        )
        d = result.to_dict()
        assert d["step_name"] == "test_step"
        assert d["state"] == "completed"
        assert d["success"] is True
        assert d["duration_ms"] == 150.5


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_pipeline_result_creation(self):
        """Test creating a pipeline result."""
        result = PipelineResult(
            pipeline_id="test-123",
            pipeline_name="test_pipeline",
            state=PipelineState.COMPLETED,
            output="final_output",
        )
        assert result.pipeline_id == "test-123"
        assert result.success is True
        assert result.output == "final_output"

    def test_pipeline_result_with_steps(self):
        """Test pipeline result with step results."""
        step_results = [
            StepResult(step_name="step1", state=StepState.COMPLETED),
            StepResult(step_name="step2", state=StepState.FAILED),
            StepResult(step_name="step3", state=StepState.COMPLETED),
        ]
        result = PipelineResult(
            pipeline_id="test-123",
            pipeline_name="test_pipeline",
            state=PipelineState.FAILED,
            step_results=step_results,
        )
        assert len(result.completed_steps) == 2
        assert len(result.failed_steps) == 1

    def test_pipeline_result_to_dict(self):
        """Test pipeline result serialization."""
        result = PipelineResult(
            pipeline_id="test-123",
            pipeline_name="test_pipeline",
            state=PipelineState.COMPLETED,
            total_duration_ms=500.0,
        )
        d = result.to_dict()
        assert d["pipeline_id"] == "test-123"
        assert d["pipeline_name"] == "test_pipeline"
        assert d["state"] == "completed"
        assert d["total_duration_ms"] == 500.0


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_context_get_set(self):
        """Test getting and setting values."""
        ctx = PipelineContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_context_default(self):
        """Test default value."""
        ctx = PipelineContext()
        assert ctx.get("missing", "default") == "default"

    def test_context_update(self):
        """Test updating multiple values."""
        ctx = PipelineContext()
        ctx.update({"a": 1, "b": 2})
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2

    def test_context_delete(self):
        """Test deleting values."""
        ctx = PipelineContext({"key": "value"})
        ctx.delete("key")
        assert ctx.has("key") is False

    def test_context_has(self):
        """Test checking key existence."""
        ctx = PipelineContext({"existing": True})
        assert ctx.has("existing") is True
        assert ctx.has("missing") is False

    def test_context_keys(self):
        """Test getting all keys."""
        ctx = PipelineContext({"a": 1, "b": 2, "c": 3})
        keys = ctx.keys()
        assert set(keys) == {"a", "b", "c"}

    def test_context_to_dict(self):
        """Test converting to dictionary."""
        ctx = PipelineContext({"key": "value"})
        d = ctx.to_dict()
        assert d == {"key": "value"}
        # Ensure it's a copy
        d["new"] = "data"
        assert ctx.has("new") is False

    def test_context_initial_data(self):
        """Test initialization with data."""
        ctx = PipelineContext({"initial": "data"})
        assert ctx.get("initial") == "data"

    @pytest.mark.asyncio
    async def test_context_async_operations(self):
        """Test async get/set operations."""
        ctx = PipelineContext()
        await ctx.async_set("key", "value")
        result = await ctx.async_get("key")
        assert result == "value"


class TestStepConfig:
    """Tests for StepConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StepConfig()
        assert config.max_retries == 0
        assert config.retry_delay == 1.0
        assert config.retry_backoff == 2.0
        assert config.timeout is None
        assert config.skip_on_failure is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = StepConfig(
            max_retries=3,
            retry_delay=0.5,
            timeout=10.0,
            skip_on_failure=True,
        )
        assert config.max_retries == 3
        assert config.retry_delay == 0.5
        assert config.timeout == 10.0
        assert config.skip_on_failure is True


class TestPipelineStep:
    """Tests for PipelineStep base class."""

    def test_step_name_default(self):
        """Test default step name."""
        step = AddStep(5)
        assert step.name == "AddStep"

    def test_step_name_custom(self):
        """Test custom step name."""
        step = AddStep(5, name="add_five")
        assert step.name == "add_five"

    def test_step_config(self):
        """Test step configuration."""
        config = StepConfig(max_retries=2)
        step = AddStep(5, config=config)
        assert step.config.max_retries == 2

    @pytest.mark.asyncio
    async def test_step_execute(self):
        """Test step execution."""
        step = AddStep(10)
        ctx = PipelineContext()
        result = await step.execute(5, ctx)
        assert result == 15

    @pytest.mark.asyncio
    async def test_step_validate_input_default(self):
        """Test default input validation."""
        step = AddStep(5)
        ctx = PipelineContext()
        assert await step.validate_input(10, ctx) is True

    def test_step_should_skip_default(self):
        """Test default skip behavior."""
        step = AddStep(5)
        ctx = PipelineContext()
        assert step.should_skip(ctx) is False

    def test_step_should_skip_with_condition(self):
        """Test skip with condition."""
        config = StepConfig(condition=lambda ctx: ctx.get("run_step", False))
        step = AddStep(5, config=config)
        ctx = PipelineContext()
        assert step.should_skip(ctx) is True

        ctx.set("run_step", True)
        assert step.should_skip(ctx) is False


class TestFunctionStep:
    """Tests for FunctionStep."""

    @pytest.mark.asyncio
    async def test_sync_function(self):
        """Test wrapping a sync function."""
        step = FunctionStep("double", lambda x, ctx: x * 2)
        ctx = PipelineContext()
        result = await step.execute(5, ctx)
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Test wrapping an async function."""

        async def async_double(x, ctx):
            await asyncio.sleep(0.01)
            return x * 2

        step = FunctionStep("async_double", async_double)
        ctx = PipelineContext()
        result = await step.execute(5, ctx)
        assert result == 10


class TestPipeline:
    """Tests for Pipeline class."""

    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        pipeline = Pipeline("test_pipeline")
        assert pipeline.name == "test_pipeline"
        assert pipeline.state == PipelineState.PENDING
        assert pipeline.step_count == 0

    def test_add_step(self):
        """Test adding a step."""
        pipeline = Pipeline("test")
        pipeline.add_step(AddStep(5))
        assert pipeline.step_count == 1

    def test_add_steps(self):
        """Test adding multiple steps."""
        pipeline = Pipeline("test")
        pipeline.add_steps(AddStep(1), AddStep(2), AddStep(3))
        assert pipeline.step_count == 3

    def test_add_step_chaining(self):
        """Test method chaining."""
        pipeline = (
            Pipeline("test")
            .add_step(AddStep(1))
            .add_step(AddStep(2))
        )
        assert pipeline.step_count == 2

    def test_insert_step(self):
        """Test inserting a step."""
        pipeline = Pipeline("test")
        pipeline.add_step(AddStep(1, name="first"))
        pipeline.add_step(AddStep(3, name="third"))
        pipeline.insert_step(1, AddStep(2, name="second"))
        assert pipeline.steps[1].name == "second"

    def test_remove_step(self):
        """Test removing a step."""
        pipeline = Pipeline("test")
        pipeline.add_step(AddStep(1, name="to_remove"))
        assert pipeline.remove_step("to_remove") is True
        assert pipeline.step_count == 0
        assert pipeline.remove_step("nonexistent") is False

    def test_get_step(self):
        """Test getting a step by name."""
        pipeline = Pipeline("test")
        step = AddStep(5, name="my_step")
        pipeline.add_step(step)
        assert pipeline.get_step("my_step") == step
        assert pipeline.get_step("nonexistent") is None

    @pytest.mark.asyncio
    async def test_simple_pipeline_run(self):
        """Test running a simple pipeline."""
        pipeline = Pipeline("simple")
        pipeline.add_step(AddStep(5, name="add_five"))
        pipeline.add_step(MultiplyStep(2, name="multiply_two"))

        result = await pipeline.run(10)

        assert result.success is True
        assert result.output == 30  # (10 + 5) * 2
        assert len(result.step_results) == 2

    @pytest.mark.asyncio
    async def test_pipeline_with_context(self):
        """Test pipeline with context data."""
        pipeline = Pipeline("context_test")
        pipeline.add_step(ContextStep("value", value=42, name="set_value"))
        pipeline.add_step(ContextStep("value", name="get_value"))

        result = await pipeline.run(None)

        assert result.success is True
        assert result.output == 42

    @pytest.mark.asyncio
    async def test_pipeline_with_kwargs(self):
        """Test pipeline with initial context kwargs."""
        pipeline = Pipeline("kwargs_test")
        pipeline.add_step(ContextStep("initial_value", name="get_initial"))

        result = await pipeline.run(None, initial_value="from_kwargs")

        assert result.success is True
        assert result.output == "from_kwargs"

    @pytest.mark.asyncio
    async def test_pipeline_failure(self):
        """Test pipeline failure handling."""
        pipeline = Pipeline("failing")
        pipeline.add_step(AddStep(5, name="add"))
        pipeline.add_step(FailingStep(name="fail"))
        pipeline.add_step(AddStep(10, name="never_reached"))

        result = await pipeline.run(10)

        assert result.success is False
        assert result.state == PipelineState.FAILED
        assert len(result.completed_steps) == 1
        assert len(result.failed_steps) == 1

    @pytest.mark.asyncio
    async def test_pipeline_stop_on_error(self):
        """Test stop_on_error=False continues execution."""
        pipeline = Pipeline("continue_on_error", stop_on_error=False)
        pipeline.add_step(AddStep(5, name="add1"))
        pipeline.add_step(FailingStep(name="fail", config=StepConfig(skip_on_failure=True)))
        pipeline.add_step(AddStep(10, name="add2"))

        result = await pipeline.run(10)

        # Should complete despite failure
        assert result.state == PipelineState.COMPLETED
        assert len(result.step_results) == 3

    @pytest.mark.asyncio
    async def test_pipeline_no_steps_error(self):
        """Test running empty pipeline raises error."""
        pipeline = Pipeline("empty")

        with pytest.raises(PipelineError) as exc_info:
            await pipeline.run(None)

        assert "no steps" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_pipeline_already_running_error(self):
        """Test running pipeline that's already running."""
        pipeline = Pipeline("test")
        pipeline.add_step(SlowStep(delay=1.0))
        pipeline._state = PipelineState.RUNNING

        with pytest.raises(PipelineStateError):
            await pipeline.run(None)

        pipeline._state = PipelineState.PENDING


class TestPipelineCallbacks:
    """Tests for pipeline callbacks."""

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback is called."""
        progress_calls = []

        def on_progress(step_name, current, total, message):
            progress_calls.append((step_name, current, total, message))

        pipeline = Pipeline("callback_test")
        pipeline.add_step(AddStep(1, name="step1"))
        pipeline.add_step(AddStep(2, name="step2"))
        pipeline.on_progress(on_progress)

        await pipeline.run(0)

        assert len(progress_calls) >= 2
        assert progress_calls[0][0] == "step1"
        assert progress_calls[0][1] == 1  # current
        assert progress_calls[0][2] == 2  # total

    @pytest.mark.asyncio
    async def test_step_callback(self):
        """Test step state change callback."""
        step_calls = []

        def on_step(step_name, state, result):
            step_calls.append((step_name, state))

        pipeline = Pipeline("step_callback_test")
        pipeline.add_step(AddStep(1, name="step1"))
        pipeline.on_step(on_step)

        await pipeline.run(0)

        # Should have RUNNING and COMPLETED states
        states = [s[1] for s in step_calls if s[0] == "step1"]
        assert StepState.RUNNING in states
        assert StepState.COMPLETED in states

    @pytest.mark.asyncio
    async def test_error_callback(self):
        """Test error callback."""
        error_calls = []

        def on_error(step_name, error, retry_count):
            error_calls.append((step_name, str(error), retry_count))
            return False  # Don't retry

        pipeline = Pipeline("error_callback_test")
        pipeline.add_step(FailingStep(name="failing_step"))
        pipeline.on_error(on_error)

        await pipeline.run(None)

        assert len(error_calls) == 1
        assert error_calls[0][0] == "failing_step"


class TestPipelineRetries:
    """Tests for pipeline retry functionality."""

    @pytest.mark.asyncio
    async def test_step_retry_success(self):
        """Test successful retry after failures."""
        step = RetryableStep(
            fail_count=2,
            name="retryable",
            config=StepConfig(max_retries=3, retry_delay=0.01),
        )
        pipeline = Pipeline("retry_test")
        pipeline.add_step(step)

        result = await pipeline.run("input")

        assert result.success is True
        step_result = result.step_results[0]
        assert step_result.retry_count == 2

    @pytest.mark.asyncio
    async def test_step_retry_failure(self):
        """Test failure after max retries."""
        step = RetryableStep(
            fail_count=5,  # Will fail more than max retries
            name="retryable",
            config=StepConfig(max_retries=2, retry_delay=0.01),
        )
        pipeline = Pipeline("retry_test")
        pipeline.add_step(step)

        result = await pipeline.run("input")

        assert result.success is False
        step_result = result.step_results[0]
        assert step_result.state == StepState.FAILED

    @pytest.mark.asyncio
    async def test_error_callback_retry(self):
        """Test error callback can trigger retry."""
        callback_calls = []

        def on_error(step_name, error, count):
            callback_calls.append(count)
            return count < 1  # Retry once (when count is 0)

        step = FailingStep(name="always_fail")
        pipeline = Pipeline("error_retry_test")
        pipeline.add_step(step)
        pipeline.on_error(on_error)

        result = await pipeline.run(None)

        # Callback should have been called at least once
        assert len(callback_calls) >= 1
        # Step should have retried (check from step result)
        assert result.step_results[0].retry_count >= 1


class TestPipelineTimeout:
    """Tests for step timeout functionality."""

    @pytest.mark.asyncio
    async def test_step_timeout(self):
        """Test step times out correctly."""
        step = SlowStep(
            delay=1.0,
            name="slow_step",
            config=StepConfig(timeout=0.05),
        )
        pipeline = Pipeline("timeout_test")
        pipeline.add_step(step)

        result = await pipeline.run(None)

        assert result.success is False
        assert result.step_results[0].state == StepState.FAILED


class TestConditionalStep:
    """Tests for ConditionalStep."""

    @pytest.mark.asyncio
    async def test_condition_true(self):
        """Test condition evaluates to true."""
        step = ConditionalStep(
            "conditional",
            condition=lambda ctx: ctx.get("use_first", False),
            if_true=AddStep(10, name="add_ten"),
            if_false=AddStep(5, name="add_five"),
        )

        ctx = PipelineContext({"use_first": True})
        result = await step.execute(5, ctx)
        assert result == 15

    @pytest.mark.asyncio
    async def test_condition_false(self):
        """Test condition evaluates to false."""
        step = ConditionalStep(
            "conditional",
            condition=lambda ctx: ctx.get("use_first", False),
            if_true=AddStep(10, name="add_ten"),
            if_false=AddStep(5, name="add_five"),
        )

        ctx = PipelineContext({"use_first": False})
        result = await step.execute(5, ctx)
        assert result == 10

    @pytest.mark.asyncio
    async def test_condition_false_no_else(self):
        """Test condition false with no else step."""
        step = ConditionalStep(
            "conditional",
            condition=lambda ctx: False,
            if_true=AddStep(10),
        )

        ctx = PipelineContext()
        result = await step.execute(5, ctx)
        assert result == 5  # Passes through


class TestParallelSteps:
    """Tests for ParallelSteps."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel step execution."""
        parallel = ParallelSteps(
            "parallel",
            [
                FunctionStep("double", lambda x, ctx: x * 2),
                FunctionStep("triple", lambda x, ctx: x * 3),
                FunctionStep("square", lambda x, ctx: x ** 2),
            ],
        )

        ctx = PipelineContext()
        result = await parallel.execute(5, ctx)

        assert result["double"] == 10
        assert result["triple"] == 15
        assert result["square"] == 25

    @pytest.mark.asyncio
    async def test_parallel_with_failure(self):
        """Test parallel steps with one failure."""
        parallel = ParallelSteps(
            "parallel",
            [
                FunctionStep("success", lambda x, ctx: x),
                FailingStep(name="fail"),
            ],
            fail_fast=False,
        )

        ctx = PipelineContext()
        result = await parallel.execute(5, ctx)

        assert "success" in result
        assert ctx.get("parallel_errors") is not None


class TestSkipStep:
    """Tests for step skipping functionality."""

    @pytest.mark.asyncio
    async def test_skip_step_by_condition(self):
        """Test skipping step by condition."""
        config = StepConfig(condition=lambda ctx: ctx.get("should_run", False))
        step = AddStep(10, name="conditional_add", config=config)

        pipeline = Pipeline("skip_test")
        pipeline.add_step(step)

        result = await pipeline.run(5, should_run=False)

        assert result.success is True
        assert result.step_results[0].state == StepState.SKIPPED


class TestCreatePipeline:
    """Tests for create_pipeline helper function."""

    def test_create_empty_pipeline(self):
        """Test creating empty pipeline."""
        pipeline = create_pipeline("test")
        assert pipeline.name == "test"
        assert pipeline.step_count == 0

    def test_create_pipeline_with_steps(self):
        """Test creating pipeline with steps."""
        pipeline = create_pipeline(
            "test",
            steps=[AddStep(1), AddStep(2), AddStep(3)],
        )
        assert pipeline.step_count == 3

    @pytest.mark.asyncio
    async def test_created_pipeline_runs(self):
        """Test created pipeline executes correctly."""
        pipeline = create_pipeline(
            "test",
            steps=[AddStep(5), MultiplyStep(2)],
        )
        result = await pipeline.run(10)
        assert result.output == 30


class TestPipelineRunFromStep:
    """Tests for running pipeline from a specific step."""

    @pytest.mark.asyncio
    async def test_run_from_step(self):
        """Test running from a specific step."""
        pipeline = Pipeline("test")
        pipeline.add_step(AddStep(1, name="step1"))
        pipeline.add_step(AddStep(2, name="step2"))
        pipeline.add_step(AddStep(3, name="step3"))

        # Start from step2
        result = await pipeline.run_from_step("step2", input_data=10)

        # Should only run step2 and step3
        assert result.output == 15  # 10 + 2 + 3
        assert len(result.step_results) == 2

    @pytest.mark.asyncio
    async def test_run_from_step_not_found(self):
        """Test error when step not found."""
        pipeline = Pipeline("test")
        pipeline.add_step(AddStep(1, name="step1"))

        with pytest.raises(PipelineError) as exc_info:
            await pipeline.run_from_step("nonexistent", None)

        assert "not found" in str(exc_info.value).lower()


class TestPipelineReset:
    """Tests for pipeline reset functionality."""

    def test_reset_pipeline(self):
        """Test resetting pipeline state."""
        pipeline = Pipeline("test")
        pipeline._state = PipelineState.COMPLETED
        pipeline._pipeline_id = "test-123"
        pipeline._current_step_index = 5

        pipeline.reset()

        assert pipeline.state == PipelineState.PENDING
        assert pipeline._pipeline_id is None
        assert pipeline._current_step_index == 0


class TestPipelineHooks:
    """Tests for step lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_on_start_hook(self):
        """Test on_start hook is called."""
        hook_called = [False]

        class HookedStep(PipelineStep[int, int]):
            async def execute(self, input_data: int, context: PipelineContext) -> int:
                return input_data

            async def on_start(self, context: PipelineContext) -> None:
                hook_called[0] = True

        pipeline = Pipeline("hooks_test")
        pipeline.add_step(HookedStep(name="hooked"))

        await pipeline.run(5)

        assert hook_called[0] is True

    @pytest.mark.asyncio
    async def test_on_complete_hook(self):
        """Test on_complete hook is called."""
        hook_data = [None]

        class HookedStep(PipelineStep[int, int]):
            async def execute(self, input_data: int, context: PipelineContext) -> int:
                return input_data * 2

            async def on_complete(self, output: int, context: PipelineContext) -> None:
                hook_data[0] = output

        pipeline = Pipeline("hooks_test")
        pipeline.add_step(HookedStep(name="hooked"))

        await pipeline.run(5)

        assert hook_data[0] == 10

    @pytest.mark.asyncio
    async def test_on_error_hook(self):
        """Test on_error hook is called."""
        hook_error = [None]

        class HookedStep(PipelineStep[Any, Any]):
            async def execute(self, input_data: Any, context: PipelineContext) -> Any:
                raise ValueError("Test error")

            async def on_error(self, error: Exception, context: PipelineContext) -> None:
                hook_error[0] = error

        pipeline = Pipeline("hooks_test")
        pipeline.add_step(HookedStep(name="hooked"))

        await pipeline.run(5)

        assert hook_error[0] is not None
        assert "Test error" in str(hook_error[0])
