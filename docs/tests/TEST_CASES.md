# Sentimatrix V2 - Test Cases

## Core Module Tests

### Config Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| CFG-001 | Load valid YAML config | Config object created |
| CFG-002 | Load invalid YAML | ConfigError raised |
| CFG-003 | Environment variable interpolation | Variables replaced |
| CFG-004 | Missing required field | ValidationError raised |
| CFG-005 | Default values applied | Defaults set correctly |
| CFG-006 | Config override at runtime | Override applied |
| CFG-007 | Nested config access | Correct values returned |

### Pipeline Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| PIP-001 | Execute empty pipeline | Empty result |
| PIP-002 | Single step execution | Step result returned |
| PIP-003 | Multi-step execution | All steps executed in order |
| PIP-004 | Step failure handling | Error captured, pipeline stops |
| PIP-005 | Step failure with continue | Next step executed |
| PIP-006 | Parallel step execution | Steps run concurrently |
| PIP-007 | Progress callback | Callbacks invoked |
| PIP-008 | Pipeline timeout | TimeoutError raised |

### Cache Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| CAC-001 | Set and get value | Value retrieved |
| CAC-002 | Get non-existent key | None returned |
| CAC-003 | TTL expiration | None after TTL |
| CAC-004 | Delete key | Key removed |
| CAC-005 | Clear cache | All keys removed |
| CAC-006 | Namespace isolation | Keys isolated |
| CAC-007 | Cache hit tracking | Hit count incremented |
| CAC-008 | Cache miss tracking | Miss count incremented |

---

## Sentiment Analysis Tests

| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| SEN-001 | Positive sentiment | "I love this!" | label: positive, score > 0.7 |
| SEN-002 | Negative sentiment | "This is terrible" | label: negative, score > 0.7 |
| SEN-003 | Neutral sentiment | "It's okay" | label: neutral |
| SEN-004 | Empty input | "" | ValidationError |
| SEN-005 | Very long input | 10000 chars | Truncated, result returned |
| SEN-006 | Special characters | "Great!!! <3" | Valid result |
| SEN-007 | Multiple languages | "C'est magnifique" | Valid result |
| SEN-008 | Batch processing | List of 100 texts | 100 results |
| SEN-009 | Mixed batch | Mix of sentiments | Correct labels |
| SEN-010 | Unicode handling | Emoji text | Valid result |

---

## Emotion Detection Tests

| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| EMO-001 | Joy detection | "I'm so happy!" | joy in top emotions |
| EMO-002 | Anger detection | "This makes me furious" | anger in top emotions |
| EMO-003 | Sadness detection | "I feel so sad" | sadness in top emotions |
| EMO-004 | Fear detection | "I'm scared" | fear in top emotions |
| EMO-005 | Surprise detection | "Wow, unexpected!" | surprise in top emotions |
| EMO-006 | Multiple emotions | Complex text | Multiple emotions detected |
| EMO-007 | Top-K selection | k=3 | Exactly 3 emotions |
| EMO-008 | Threshold filtering | threshold=0.5 | Only high-confidence |

---

## Scraper Tests

### Generic Scraper

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| SCR-001 | Scrape valid URL | Content returned |
| SCR-002 | Scrape invalid URL | URLError raised |
| SCR-003 | Scrape 404 page | HTTPError raised |
| SCR-004 | Scrape with timeout | TimeoutError raised |
| SCR-005 | Scrape with proxy | Request via proxy |
| SCR-006 | User-agent rotation | Different UA per request |
| SCR-007 | Rate limiting | Requests throttled |
| SCR-008 | Retry on failure | Retried N times |

### Platform Scrapers

| Test ID | Platform | Description | Expected Result |
|---------|----------|-------------|-----------------|
| PLT-001 | Amazon | Scrape product reviews | Reviews extracted |
| PLT-002 | Amazon | Invalid product URL | Error raised |
| PLT-003 | Steam | Scrape game reviews | Reviews extracted |
| PLT-004 | Steam | Invalid app ID | Error raised |
| PLT-005 | YouTube | Get video comments | Comments extracted |
| PLT-006 | YouTube | Invalid video ID | Error raised |
| PLT-007 | Reddit | Search posts | Posts extracted |
| PLT-008 | IMDB | Scrape movie reviews | Reviews extracted |
| PLT-009 | Yelp | Scrape business reviews | Reviews extracted |
| PLT-010 | Trustpilot | Scrape company reviews | Reviews extracted |

---

## LLM Provider Tests

### OpenAI Provider

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| OAI-001 | Generate text | Response returned |
| OAI-002 | Stream response | Chunks yielded |
| OAI-003 | Invalid API key | AuthenticationError |
| OAI-004 | Rate limit hit | RateLimitError |
| OAI-005 | Timeout | TimeoutError |
| OAI-006 | Function calling | Function result returned |
| OAI-007 | JSON mode | Valid JSON returned |
| OAI-008 | Vision input | Vision response returned |

### Anthropic Provider

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| ANT-001 | Generate text | Response returned |
| ANT-002 | Stream response | Chunks yielded |
| ANT-003 | System prompt | System prompt applied |
| ANT-004 | Tool use | Tool result returned |
| ANT-005 | Long context | Handles 100K+ tokens |

### Groq Provider

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| GRQ-001 | Generate text | Response returned |
| GRQ-002 | Verify speed | < 2s for 500 tokens |
| GRQ-003 | Rate limit handling | Backoff and retry |

### Local Provider (Ollama)

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| OLL-001 | Generate text | Response returned |
| OLL-002 | Server not running | ConnectionError |
| OLL-003 | Model not found | ModelNotFoundError |
| OLL-004 | Streaming | Chunks yielded |
| OLL-005 | Embeddings | Vector returned |

---

## Output/Export Tests

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| OUT-001 | Export to JSON | Valid JSON file |
| OUT-002 | Export to CSV | Valid CSV file |
| OUT-003 | Export to Excel | Valid XLSX file |
| OUT-004 | Generate bar chart | PNG/SVG created |
| OUT-005 | Generate pie chart | PNG/SVG created |
| OUT-006 | Generate report | HTML report created |
| OUT-007 | Empty data export | Empty file/error |
| OUT-008 | Large data export | File created successfully |

---

## Error Handling Tests

| Test ID | Description | Expected Behavior |
|---------|-------------|-------------------|
| ERR-001 | Network timeout | Retry with backoff |
| ERR-002 | API rate limit | Wait and retry |
| ERR-003 | Invalid input | ValidationError with message |
| ERR-004 | Provider unavailable | Fallback to next provider |
| ERR-005 | Out of memory | Graceful degradation |
| ERR-006 | Concurrent errors | All errors captured |
| ERR-007 | Partial batch failure | Successful items returned |

---

## Security Tests

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| SEC-001 | API key not logged | No keys in logs |
| SEC-002 | SQL injection attempt | Input sanitized |
| SEC-003 | XSS in input | Input sanitized |
| SEC-004 | Path traversal | Access denied |
| SEC-005 | Rate limiting | Requests throttled |

---

## Performance Tests

| Test ID | Description | Target |
|---------|-------------|--------|
| PRF-001 | Single sentiment inference | < 50ms |
| PRF-002 | Batch sentiment (100 items) | < 2s |
| PRF-003 | Page scrape | < 5s |
| PRF-004 | LLM summary | < 3s |
| PRF-005 | Memory usage (1000 items) | < 500MB |
| PRF-006 | Concurrent requests (10) | All complete |
| PRF-007 | Cache hit latency | < 1ms |
