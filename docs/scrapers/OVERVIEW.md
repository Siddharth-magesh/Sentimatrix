# Sentimatrix V2 - Scraping Overview

## Architecture

V2 provides a unified scraping layer with multiple provider options. Users can choose based on their needs for speed, reliability, cost, and target platforms.

```
┌─────────────────────────────────────────────────────────────┐
│                    SCRAPER MANAGER                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   LOCAL     │  │  BROWSER    │  │    API      │         │
│  │  SCRAPERS   │  │  SCRAPERS   │  │  SCRAPERS   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│  - Requests        - Playwright     - ScraperAPI           │
│  - HTTPX           - Selenium       - Bright Data          │
│  - BeautifulSoup   - Puppeteer      - Oxylabs              │
│  - Scrapy                           - Apify                 │
│                                     - Zyte                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PLATFORM SCRAPERS                       │   │
│  │  Amazon | Steam | YouTube | Reddit | IMDB | ...      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              AI-POWERED SCRAPERS                     │   │
│  │  Firecrawl | Crawl4AI | ScrapeGraphAI               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Scraper Categories

### 1. Local HTTP Scrapers
Simple HTTP-based scrapers for static content.

| Library | Async | Speed | Use Case |
|---------|-------|-------|----------|
| Requests | No | Fast | Simple pages |
| HTTPX | Yes | Fast | Async workflows |
| aiohttp | Yes | Fast | High concurrency |

### 2. Browser Automation
For JavaScript-rendered content and complex interactions.

| Tool | Async | Anti-Detection | Use Case |
|------|-------|----------------|----------|
| Playwright | Yes | Good | Modern choice |
| Selenium | No | Moderate | Legacy support |
| Puppeteer (via pyppeteer) | Yes | Good | Chrome-specific |

### 3. Commercial APIs
Managed scraping services with anti-bot bypass.

| Service | Proxy Pool | CAPTCHA | Pricing |
|---------|------------|---------|---------|
| ScraperAPI | 40M+ | Yes | From $49/mo |
| Bright Data | 72M+ | Yes | From $500/mo |
| Oxylabs | 100M+ | Yes | Custom |
| Apify | N/A | Actors | Pay-per-use |
| Zyte | 50M+ | Yes | From $450/mo |
| ScrapingBee | 1M+ | Yes | From $49/mo |

### 4. AI-Powered Scrapers
LLM-based intelligent extraction.

| Tool | Type | Features |
|------|------|----------|
| Firecrawl | API | Auto-extraction, LLM-ready output |
| Crawl4AI | OSS | GPT-powered, async |
| ScrapeGraphAI | OSS | Natural language queries |

---

## Platform Support Matrix

| Platform | Local | Browser | API | Priority |
|----------|-------|---------|-----|----------|
| Amazon | Partial | Yes | Yes | P0 |
| Steam | Yes | No | API | P0 |
| YouTube | No | No | API | P0 |
| Reddit | No | No | API | P0 |
| IMDB | No | Yes | OMDb | P1 |
| Twitter/X | No | Difficult | API | P1 |
| TikTok | No | Difficult | Third-party | P2 |
| Yelp | Partial | Yes | Unofficial | P1 |
| Trustpilot | Partial | Yes | No | P1 |
| Google Reviews | No | Difficult | Third-party | P2 |
| App Store | Partial | Yes | No | P1 |
| Play Store | Partial | Yes | No | P1 |
| Metacritic | No | Yes | No | P1 |
| Rotten Tomatoes | No | Yes | No | P1 |
| LetterBoxD | No | Yes | No | P2 |
| Glassdoor | No | Difficult | No | P2 |
| LinkedIn | No | Difficult | API (limited) | P2 |
| Facebook | No | Difficult | API (limited) | P2 |
| Instagram | No | Difficult | No | P2 |
| Tripadvisor | Partial | Yes | No | P1 |

---

## Selection Guide

### By Budget

| Budget | Recommendation |
|--------|----------------|
| Free | Playwright + local proxies |
| < $50/mo | ScraperAPI or ScrapingBee |
| $50-500/mo | Apify or Zyte |
| > $500/mo | Bright Data or Oxylabs |

### By Volume

| Volume | Recommendation |
|--------|----------------|
| < 1K pages/day | Local scrapers |
| 1K-10K pages/day | ScraperAPI + local |
| 10K-100K pages/day | Commercial API |
| > 100K pages/day | Bright Data enterprise |

### By Target Site

| Target | Recommendation |
|--------|----------------|
| Static sites | Requests + BeautifulSoup |
| JS-heavy sites | Playwright |
| Anti-bot sites | Commercial API |
| Social media | Official APIs + fallback |

---

## Configuration

```yaml
scrapers:
  default_provider: "playwright"

  providers:
    playwright:
      headless: true
      timeout: 30000
      user_agent: "auto"

    scraperapi:
      api_key: "${SCRAPERAPI_KEY}"
      render: true
      country: "us"

    brightdata:
      api_key: "${BRIGHTDATA_KEY}"
      zone: "residential"

  rate_limiting:
    requests_per_second: 1
    concurrent_requests: 5
    backoff_factor: 2.0

  retry:
    max_retries: 3
    retry_on: [429, 500, 502, 503, 504]

  proxy:
    enabled: false
    rotation: true
    provider: "custom"
    urls: []
```

---

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| 403 Forbidden | Anti-bot | Rotate proxy/user-agent |
| 429 Too Many Requests | Rate limit | Exponential backoff |
| CAPTCHA | Bot detection | Use CAPTCHA service |
| Timeout | Slow response | Increase timeout, retry |
| Connection Error | Network issue | Retry with backoff |
| Empty Response | Blocked/changed | Try different scraper |

---

## Best Practices

1. **Respect robots.txt** - Check and honor crawl restrictions
2. **Rate limit** - Avoid overwhelming servers
3. **Rotate identifiers** - User-agent, proxy, cookies
4. **Handle failures gracefully** - Retry with backoff
5. **Cache results** - Avoid redundant requests
6. **Monitor success rates** - Detect blocking early
7. **Keep patterns updated** - Sites change frequently
