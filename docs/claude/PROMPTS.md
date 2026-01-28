# Sentimatrix V2 - Development Prompts

## Standard Prompts for Common Tasks

### Implementing a New LLM Provider

```
Implement the {ProviderName} LLM provider for Sentimatrix V2.

Requirements:
1. Create providers/llm/{provider}_provider.py
2. Inherit from BaseLLMProvider
3. Implement generate() and generate_stream() methods
4. Handle authentication via api_key parameter
5. Implement proper error handling with ProviderError
6. Support temperature, max_tokens, and other common parameters
7. Follow the existing code style (see OpenAI provider as reference)

The provider should support:
- Async operations
- Streaming responses
- Timeout handling
- Rate limit handling

Write comprehensive tests in tests/providers/test_{provider}_provider.py with:
- Mocked API calls for unit tests
- Optional live tests marked with @pytest.mark.live

Reference the documentation at: [provider docs URL]
```

### Implementing a New Platform Scraper

```
Implement the {Platform} scraper for Sentimatrix V2.

Requirements:
1. Create providers/scrapers/platforms/{platform}.py
2. Inherit from BasePlatformScraper
3. Implement:
   - get_platform_name() -> str
   - validate_url(url: str) -> bool
   - scrape_reviews(url: str, limit: int) -> List[Review]

The scraper should:
- Use Playwright for JavaScript-heavy pages
- Handle pagination to get up to `limit` reviews
- Extract: review text, rating, author, date, and any relevant metadata
- Handle common errors gracefully
- Respect rate limits

Write tests in tests/scrapers/test_{platform}.py with:
- Mocked HTTP responses
- Test for URL validation
- Test for pagination handling
- Test for error cases

Reference: {platform documentation or example URLs}
```

### Implementing a Core Feature

```
Implement the {feature_name} feature for Sentimatrix V2.

Location: {module_path}

Requirements:
1. Follow the existing architecture patterns
2. Use Pydantic for any configuration
3. Make all I/O operations async
4. Add proper type hints
5. Write comprehensive docstrings
6. Handle errors with custom exceptions

The feature should:
- {requirement 1}
- {requirement 2}
- {requirement 3}

Integration:
- Add to main Sentimatrix class if public API
- Update config schema if needed
- Register any new providers

Tests required:
- Unit tests for core logic
- Integration tests for component interaction
- Edge case handling

Documentation:
- Update API reference
- Add usage examples
```

### Writing Tests

```
Write tests for {module/class/function} in Sentimatrix V2.

Test file: tests/{path}/test_{name}.py

Requirements:
1. Use pytest with pytest-asyncio for async tests
2. Use fixtures for common setup
3. Mock external dependencies (APIs, file system, etc.)
4. Cover happy path and error cases
5. Aim for >90% coverage of the target code

Test categories needed:
- Unit tests for individual functions
- Integration tests for component interaction
- Edge cases (empty input, invalid input, etc.)
- Error handling verification

Follow existing test patterns in the codebase.
```

### Fixing a Bug

```
Fix the bug in {location} where {description of bug}.

Steps:
1. First, understand the current behavior by reading the relevant code
2. Identify the root cause
3. Implement the fix
4. Write a regression test that would have caught this bug
5. Verify no other tests are broken

Constraints:
- Maintain backward compatibility if possible
- Follow existing code style
- Add comments explaining the fix if non-obvious
```

### Code Review Prompt

```
Review the implementation of {feature/module}.

Check for:
1. Correctness - Does it do what it's supposed to?
2. Error handling - Are all error cases handled?
3. Type safety - Are type hints complete and correct?
4. Documentation - Are docstrings adequate?
5. Tests - Is test coverage sufficient?
6. Performance - Any obvious performance issues?
7. Security - Any security concerns?
8. Style - Does it follow project conventions?

Provide specific suggestions for improvements.
```

---

## Context Setting Prompts

### Starting a New Session

```
I'm working on Sentimatrix V2, a Python library for sentiment analysis and web scraping.

Current state:
- Version: 0.2.0 (in development)
- Python: 3.10+
- Key features: sentiment analysis, emotion detection, web scraping, LLM integration

Documentation is in: /path/to/Sentimatrix-V2/docs/

Today I want to work on: {specific task}
```

### Continuing Previous Work

```
Continuing development of Sentimatrix V2.

Last session we:
- {completed task 1}
- {completed task 2}
- {started but didn't finish task 3}

Today I want to:
- {continue/complete task 3}
- {new task if applicable}

Relevant files:
- {file 1}
- {file 2}
```

---

## Task-Specific Prompts

### Configuration System

```
Implement the configuration system for Sentimatrix V2.

Requirements:
- Use Pydantic v2 for validation
- Support YAML file loading
- Support environment variable interpolation (${VAR_NAME})
- Support nested configuration
- Provide sensible defaults
- Allow runtime overrides

Config sections needed:
- llm: LLM provider settings
- scrapers: Scraper settings
- models: ML model settings
- cache: Caching settings
- logging: Logging settings

Create:
1. core/config.py with all Pydantic models
2. Tests in tests/core/test_config.py
```

### Pipeline System

```
Implement the pipeline orchestration system for Sentimatrix V2.

Requirements:
- Support chaining multiple processing steps
- Support parallel execution where possible
- Provide progress callbacks
- Handle errors gracefully (continue or stop options)
- Support middleware/hooks

Interface:
```python
pipeline = Pipeline()
pipeline.add_step(ScrapeStep())
pipeline.add_step(SentimentStep())
pipeline.add_step(SummarizeStep())
pipeline.on_progress(callback)
result = await pipeline.execute(input_data)
```

Create:
1. core/pipeline.py
2. Tests in tests/core/test_pipeline.py
```

### Batch Processing

```
Implement batch processing for sentiment analysis in Sentimatrix V2.

Requirements:
- Process texts in configurable batch sizes
- Use GPU efficiently if available
- Show progress for large batches
- Handle partial failures gracefully
- Support streaming results

Interface:
```python
# Standard batch
results = await sm.analyze_sentiment_batch(texts, batch_size=32)

# Streaming batch
async for result in sm.analyze_sentiment_stream(texts):
    process(result)
```

Optimize for:
- Memory efficiency (don't load all texts at once)
- GPU utilization (batch for efficient inference)
- Error resilience (continue on individual failures)
```
