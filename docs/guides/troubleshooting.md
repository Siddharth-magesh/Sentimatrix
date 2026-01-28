# Sentimatrix V2 Troubleshooting Guide

This guide helps you diagnose and resolve common issues with Sentimatrix V2.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Import Errors](#import-errors)
3. [Provider Errors](#provider-errors)
4. [Scraper Issues](#scraper-issues)
5. [Pipeline Errors](#pipeline-errors)
6. [Performance Issues](#performance-issues)
7. [Cache Issues](#cache-issues)
8. [Common Error Messages](#common-error-messages)

---

## Installation Issues

### Package Not Found

**Problem:**
```
pip install sentimatrix
ERROR: Could not find a version that satisfies the requirement sentimatrix
```

**Solutions:**
1. Ensure you're using Python 3.9+:
   ```bash
   python --version
   ```

2. Install from source:
   ```bash
   git clone https://github.com/your-org/sentimatrix.git
   cd sentimatrix
   pip install -e .
   ```

3. Install with all dependencies:
   ```bash
   pip install -e ".[all]"
   ```

### Dependency Conflicts

**Problem:**
```
ERROR: Cannot install sentimatrix because of conflicting dependencies
```

**Solutions:**
1. Create a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   pip install sentimatrix
   ```

2. Install specific dependency versions:
   ```bash
   pip install "pydantic>=2.0,<3.0"
   pip install sentimatrix
   ```

---

## Import Errors

### Module Not Found

**Problem:**
```python
from sentimatrix.analysis.sentiment import SentimentAnalyzer
ModuleNotFoundError: No module named 'sentimatrix'
```

**Solutions:**
1. Verify installation:
   ```bash
   pip list | grep sentimatrix
   ```

2. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

3. Reinstall the package:
   ```bash
   pip uninstall sentimatrix
   pip install sentimatrix
   ```

### Circular Import

**Problem:**
```
ImportError: cannot import name 'X' from partially initialized module
```

**Solutions:**
1. Use lazy imports in your code:
   ```python
   # Instead of:
   from sentimatrix.analysis.sentiment import SentimentAnalyzer

   # Use:
   def get_analyzer():
       from sentimatrix.analysis.sentiment import SentimentAnalyzer
       return SentimentAnalyzer()
   ```

2. Check for circular dependencies in your own code.

---

## Provider Errors

### API Key Not Set

**Problem:**
```
ProviderError: API key not provided for OpenAI
```

**Solutions:**
1. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. Pass API key directly:
   ```python
   from sentimatrix.providers.llm import OpenAIProvider

   provider = OpenAIProvider(api_key="sk-...")
   ```

3. Use a `.env` file:
   ```
   # .env
   OPENAI_API_KEY=sk-...
   ```

### Rate Limit Exceeded

**Problem:**
```
ProviderError: Rate limit exceeded. Please retry after X seconds.
```

**Solutions:**
1. Add retry logic:
   ```python
   import asyncio
   from sentimatrix.analysis.sentiment import SentimentAnalyzer

   async def analyze_with_retry(text, max_retries=3):
       analyzer = SentimentAnalyzer()
       for attempt in range(max_retries):
           try:
               return await analyzer.analyze(text)
           except Exception as e:
               if "rate limit" in str(e).lower():
                   await asyncio.sleep(2 ** attempt)
               else:
                   raise
       raise Exception("Max retries exceeded")
   ```

2. Use built-in rate limiting:
   ```python
   from sentimatrix.providers.llm import OpenAIProvider

   provider = OpenAIProvider(
       api_key="sk-...",
       rate_limit=10,  # requests per minute
   )
   ```

### Connection Timeout

**Problem:**
```
ProviderError: Connection timed out
```

**Solutions:**
1. Increase timeout:
   ```python
   from sentimatrix.providers.llm import OpenAIProvider

   provider = OpenAIProvider(
       api_key="sk-...",
       timeout=60.0,  # 60 seconds
   )
   ```

2. Check network connectivity:
   ```bash
   curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

### Invalid API Response

**Problem:**
```
ProviderError: Invalid response from API
```

**Solutions:**
1. Verify API key permissions
2. Check model availability:
   ```python
   from sentimatrix.providers.llm import OpenAIProvider

   provider = OpenAIProvider(api_key="sk-...")
   models = await provider.list_models()
   print(models)
   ```

3. Use a different model:
   ```python
   provider = OpenAIProvider(api_key="sk-...", model="gpt-3.5-turbo")
   ```

---

## Scraper Issues

### Blocked by Website

**Problem:**
```
ScraperError: Request blocked (403 Forbidden)
```

**Solutions:**
1. Reduce request rate:
   ```python
   from sentimatrix.providers.scrapers import AmazonScraper
   from sentimatrix.providers.scrapers.rate_limiter import RateLimiter, RateLimitStrategy

   limiter = RateLimiter(
       strategy=RateLimitStrategy.TOKEN_BUCKET,
       requests_per_second=0.5,  # 1 request every 2 seconds
   )
   scraper = AmazonScraper(rate_limiter=limiter)
   ```

2. Use proxy rotation (if available):
   ```python
   scraper = AmazonScraper(
       proxies=["http://proxy1:8080", "http://proxy2:8080"]
   )
   ```

3. Add delays between requests:
   ```python
   import asyncio

   for product_id in product_ids:
       reviews = await scraper.scrape(product_id)
       await asyncio.sleep(5)  # Wait 5 seconds between products
   ```

### Empty Results

**Problem:**
```python
reviews = await scraper.scrape("B08N5WRWNW")
print(len(reviews))  # 0
```

**Solutions:**
1. Verify product ID is correct
2. Check if product has reviews
3. Increase max_pages:
   ```python
   reviews = await scraper.scrape("B08N5WRWNW", max_pages=10)
   ```

4. Check scraper logs:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Parse Error

**Problem:**
```
ScraperError: Failed to parse page content
```

**Solutions:**
1. Website structure may have changed - check for updates
2. Enable debug mode:
   ```python
   scraper = AmazonScraper(debug=True)
   ```

3. Report the issue on GitHub

---

## Pipeline Errors

### Step Execution Failed

**Problem:**
```
PipelineStepError: Step 'analyze' failed: ...
```

**Solutions:**
1. Add error handling to steps:
   ```python
   async def safe_analyze(ctx: PipelineContext, data):
       try:
           return await actual_analysis(data)
       except Exception as e:
           ctx.set("error", str(e))
           return {"error": str(e), "data": data}
   ```

2. Use pipeline error handlers:
   ```python
   pipeline = Pipeline(
       name="my_pipeline",
       on_error="continue",  # or "stop"
   )
   ```

### Context Not Available

**Problem:**
```python
async def my_step(ctx: PipelineContext, prev):
    value = ctx.get("key")  # Returns None unexpectedly
```

**Solutions:**
1. Verify previous step sets the value:
   ```python
   async def previous_step(ctx: PipelineContext, prev):
       ctx.set("key", "value")
       return prev
   ```

2. Use default values:
   ```python
   value = ctx.get("key", "default_value")
   ```

3. Check step order in pipeline

### Parallel Step Failures

**Problem:**
One parallel step fails, causing all to fail.

**Solutions:**
1. Use `return_exceptions=True`:
   ```python
   parallel = ParallelSteps(
       name="parallel",
       steps=[step1, step2],
       return_exceptions=True,
   )
   ```

2. Handle exceptions in individual steps:
   ```python
   async def safe_step(ctx, data):
       try:
           return await risky_operation(data)
       except Exception as e:
           return {"error": str(e)}
   ```

---

## Performance Issues

### Slow Analysis

**Problem:**
Analysis takes too long.

**Solutions:**
1. Use batch processing:
   ```python
   # Slow
   for text in texts:
       result = await analyzer.analyze(text)

   # Fast
   results = await analyzer.analyze_batch(texts)
   ```

2. Enable caching:
   ```python
   from sentimatrix.utils.cache import CacheManager

   cache = CacheManager(backend="memory", ttl=3600)
   analyzer = SentimentAnalyzer(cache=cache)
   ```

3. Use concurrent processing:
   ```python
   import asyncio

   results = await asyncio.gather(*[
       analyzer.analyze(text) for text in texts
   ])
   ```

### Memory Issues

**Problem:**
```
MemoryError: Unable to allocate...
```

**Solutions:**
1. Process in smaller batches:
   ```python
   batch_size = 100
   for i in range(0, len(texts), batch_size):
       batch = texts[i:i + batch_size]
       results = await analyzer.analyze_batch(batch)
       # Process results immediately
   ```

2. Use streaming:
   ```python
   async for result in analyzer.analyze_stream(texts):
       process(result)
   ```

3. Clear cache periodically:
   ```python
   cache.clear()
   ```

---

## Cache Issues

### Cache Not Working

**Problem:**
Same requests are making API calls.

**Solutions:**
1. Verify cache is enabled:
   ```python
   from sentimatrix.utils.cache import CacheManager

   cache = CacheManager(backend="memory", ttl=3600)
   analyzer = SentimentAnalyzer(cache=cache)
   ```

2. Check cache key generation:
   ```python
   # Ensure inputs are identical
   result1 = await analyzer.analyze("Hello")
   result2 = await analyzer.analyze("Hello")  # Should use cache
   result3 = await analyzer.analyze("Hello ")  # Different key!
   ```

### Redis Connection Failed

**Problem:**
```
CacheError: Could not connect to Redis
```

**Solutions:**
1. Verify Redis is running:
   ```bash
   redis-cli ping
   ```

2. Check connection settings:
   ```python
   from sentimatrix.utils.cache import RedisCache

   cache = RedisCache(
       host="localhost",
       port=6379,
       password="your_password",  # if needed
   )
   ```

3. Use fallback cache:
   ```python
   try:
       cache = RedisCache(host="localhost")
       await cache.ping()
   except:
       cache = CacheManager(backend="memory")
   ```

---

## Common Error Messages

### "ValidationError: Input text cannot be empty"

**Cause:** Empty string passed to analyzer.

**Solution:**
```python
if text and text.strip():
    result = await analyzer.analyze(text)
```

### "ProviderError: Model not found"

**Cause:** Invalid model name.

**Solution:**
```python
# Check available models
provider = OpenAIProvider(api_key="sk-...")
models = await provider.list_models()
print(models)

# Use valid model
provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
```

### "PipelineError: No steps defined"

**Cause:** Pipeline has no steps.

**Solution:**
```python
pipeline = Pipeline(name="my_pipeline")
pipeline.add_step(FunctionStep("step1", my_function))  # Add at least one step
result = await pipeline.run()
```

### "TimeoutError: Operation timed out"

**Cause:** Operation took too long.

**Solution:**
```python
import asyncio

async def with_timeout(coro, timeout=30):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None

result = await with_timeout(analyzer.analyze(text))
```

---

## Getting Help

If you're still experiencing issues:

1. **Check the documentation**: [API Reference](../api/README.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/your-org/sentimatrix/issues)
3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. **Create a minimal reproduction** and open a new issue with:
   - Python version
   - Sentimatrix version
   - Full error traceback
   - Minimal code to reproduce
