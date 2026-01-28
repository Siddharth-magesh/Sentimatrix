# Sentimatrix V2 - Local Scrapers

## Overview

Local scrapers run entirely on your machine without third-party API dependencies. They are free, fast, and provide full control but may struggle with anti-bot measures.

---

## 1. HTTP Client Scrapers

### Requests + BeautifulSoup

**Best For:** Static HTML pages, simple structures

**Pros:**
- Simple and reliable
- No browser overhead
- Fast execution

**Cons:**
- No JavaScript rendering
- Limited anti-detection

**Implementation:**
```python
# Module: providers/scrapers/requests_scraper.py
class RequestsScraper(BaseScraperProvider):
    async def scrape(self, url: str, **kwargs) -> ScrapedContent
    async def scrape_reviews(self, url: str, limit: int) -> List[Review]
```

**Configuration:**
```yaml
requests:
  timeout: 30
  headers:
    User-Agent: "Mozilla/5.0..."
  verify_ssl: true
  follow_redirects: true
```

---

### HTTPX (Async)

**Best For:** High-concurrency scraping, async workflows

**Pros:**
- Native async support
- HTTP/2 support
- Connection pooling

**Cons:**
- No JavaScript rendering

**Implementation:**
```python
# Module: providers/scrapers/httpx_scraper.py
class HTTPXScraper(BaseScraperProvider):
    async def scrape(self, url: str, **kwargs) -> ScrapedContent
    async def scrape_batch(self, urls: List[str], **kwargs) -> List[ScrapedContent]
```

**Configuration:**
```yaml
httpx:
  timeout: 30
  http2: true
  max_connections: 100
  max_keepalive_connections: 20
```

---

### Scrapy Framework

**Best For:** Large-scale crawling, structured extraction

**Pros:**
- Built-in middleware system
- Automatic rate limiting
- Export pipelines
- Robust error handling

**Cons:**
- Steeper learning curve
- Overkill for simple tasks

**Integration Approach:**
- Wrap Scrapy spiders as providers
- Use Scrapy for batch crawling
- Feed results to Sentimatrix analysis

---

## 2. Browser Automation Scrapers

### Playwright

**Best For:** JavaScript-heavy sites, modern web apps

**Pros:**
- Multi-browser support (Chromium, Firefox, WebKit)
- Async-native
- Excellent anti-detection
- Auto-waiting
- Network interception

**Cons:**
- Higher resource usage
- Slower than HTTP

**Implementation:**
```python
# Module: providers/scrapers/playwright_scraper.py
class PlaywrightScraper(BaseScraperProvider):
    async def scrape(self, url: str, **kwargs) -> ScrapedContent
    async def scrape_with_interaction(self, url: str, actions: List[Action]) -> ScrapedContent
    async def screenshot(self, url: str) -> bytes
```

**Configuration:**
```yaml
playwright:
  browser: "chromium"  # chromium, firefox, webkit
  headless: true
  timeout: 30000
  viewport:
    width: 1920
    height: 1080
  user_agent: "auto"
  stealth: true  # Use playwright-stealth
  proxy: null
```

**Anti-Detection Features:**
- Stealth mode (playwright-stealth plugin)
- Random user-agent rotation
- Human-like delays
- Mouse movement simulation
- WebGL/Canvas fingerprint spoofing

---

### Selenium

**Best For:** Legacy systems, specific browser requirements

**Pros:**
- Mature ecosystem
- Wide browser support
- Extensive documentation

**Cons:**
- Not async-native
- Slower than Playwright
- More detectable

**Implementation:**
```python
# Module: providers/scrapers/selenium_scraper.py
class SeleniumScraper(BaseScraperProvider):
    async def scrape(self, url: str, **kwargs) -> ScrapedContent
    def scrape_sync(self, url: str, **kwargs) -> ScrapedContent
```

**Configuration:**
```yaml
selenium:
  browser: "chrome"  # chrome, firefox, edge
  headless: true
  timeout: 30
  driver_path: null  # Auto-detect
  options:
    - "--disable-blink-features=AutomationControlled"
    - "--disable-dev-shm-usage"
```

---

## 3. HTML Parsing

### BeautifulSoup

**Parsers:**
| Parser | Speed | Lenient | Dependencies |
|--------|-------|---------|--------------|
| html.parser | Moderate | Yes | None |
| lxml | Fast | Moderate | lxml |
| html5lib | Slow | Very | html5lib |

**Recommended:** lxml for speed, html5lib for broken HTML

### lxml

**Direct Usage:**
- XPath support
- Very fast parsing
- Good for structured extraction

### Selectolax

**Best For:** Maximum parsing speed

**Performance:** 5-10x faster than BeautifulSoup

---

## 4. Review Pattern System

V2 maintains a pattern database for common review structures.

**Pattern Structure:**
```python
{
    "name": "amazon_reviews",
    "selectors": {
        "review_container": "div.review",
        "review_text": "span.review-text",
        "rating": "span.a-icon-alt",
        "author": "span.a-profile-name",
        "date": "span.review-date"
    },
    "pagination": {
        "type": "url_param",
        "param": "pageNumber"
    }
}
```

**Pattern Management:**
```python
scraper.add_pattern(pattern_dict)
scraper.get_patterns()
scraper.remove_pattern(name)
scraper.update_pattern(name, pattern_dict)
```

**Built-in Patterns:** 100+ patterns for common sites

---

## 5. Proxy Support

### Local Proxy Configuration

```yaml
proxy:
  enabled: true
  rotation: true
  type: "http"  # http, socks5
  urls:
    - "http://proxy1:8080"
    - "http://proxy2:8080"
  rotation_strategy: "round_robin"  # round_robin, random, least_used
  health_check: true
  health_check_interval: 60
```

### Proxy Rotation Strategies

| Strategy | Description |
|----------|-------------|
| Round Robin | Cycle through proxies sequentially |
| Random | Random selection each request |
| Least Used | Prefer less-used proxies |
| Weighted | Weight by success rate |
| Sticky | Same proxy for session |

---

## 6. Rate Limiting

**Configuration:**
```yaml
rate_limiting:
  enabled: true
  strategy: "token_bucket"  # fixed, sliding_window, token_bucket
  requests_per_second: 1.0
  burst_size: 5
  per_domain: true
  cooldown_on_429: 60
```

**Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Fixed Window | N requests per window | Simple limiting |
| Sliding Window | Rolling window limit | Smooth distribution |
| Token Bucket | Burst-friendly limiting | Variable traffic |

---

## 7. User-Agent Rotation

**Built-in User-Agent Pool:**
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Mobile browsers (iOS Safari, Android Chrome)
- Realistic version distributions

**Configuration:**
```yaml
user_agent:
  rotation: true
  type: "desktop"  # desktop, mobile, mixed
  update_frequency: "per_request"  # per_request, per_session
```

---

## Error Recovery

**Retry Configuration:**
```yaml
retry:
  enabled: true
  max_retries: 3
  backoff_factor: 2.0
  backoff_max: 60
  retry_on:
    - 429
    - 500
    - 502
    - 503
    - 504
  exceptions:
    - ConnectionError
    - Timeout
```

**Recovery Actions:**
1. Exponential backoff
2. Proxy rotation
3. User-agent change
4. Cookie refresh
5. Fallback to alternative scraper
