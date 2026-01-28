# Sentimatrix V2 - API-Based Scrapers

## Overview

Commercial scraping APIs handle anti-bot measures, proxy rotation, and CAPTCHA solving. They are more reliable but incur costs.

---

## 1. ScraperAPI

**Website:** https://www.scraperapi.com

**Features:**
- 40M+ rotating proxies
- JavaScript rendering
- CAPTCHA handling
- Geotargeting (50+ countries)
- Auto-retry

**Pricing:**
| Plan | Requests/mo | Price |
|------|-------------|-------|
| Hobby | 5,000 | Free |
| Startup | 100,000 | $49/mo |
| Business | 250,000 | $149/mo |
| Enterprise | Custom | Custom |

**Implementation:**
```python
# Module: providers/scrapers/scraperapi_provider.py
class ScraperAPIProvider(BaseScraperProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.scraperapi.com"

    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        params = {
            "api_key": self.api_key,
            "url": url,
            "render": kwargs.get("render", True),
            "country_code": kwargs.get("country", "us")
        }
        # Implementation...
```

**Configuration:**
```yaml
scraperapi:
  api_key: "${SCRAPERAPI_KEY}"
  render: true
  country: "us"
  premium: false
  session_number: null  # For sticky sessions
```

---

## 2. Bright Data (Luminati)

**Website:** https://brightdata.com

**Features:**
- 72M+ residential IPs
- Web Unlocker (anti-bot bypass)
- SERP API
- Pre-built datasets
- Dedicated account managers

**Products:**
| Product | Use Case |
|---------|----------|
| Residential Proxies | General scraping |
| Datacenter Proxies | High-speed, low cost |
| ISP Proxies | Residential + speed |
| Mobile Proxies | Mobile-specific |
| Web Unlocker | Anti-bot sites |
| SERP API | Search results |

**Pricing:** Custom, enterprise-focused (from $500/mo)

**Implementation:**
```python
# Module: providers/scrapers/brightdata_provider.py
class BrightDataProvider(BaseScraperProvider):
    def __init__(self, username: str, password: str, zone: str):
        self.proxy_url = f"http://{username}:{password}@zproxy.lum-superproxy.io:22225"

    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        # Uses proxy with existing HTTP client
        # Implementation...
```

**Configuration:**
```yaml
brightdata:
  username: "${BRIGHTDATA_USER}"
  password: "${BRIGHTDATA_PASS}"
  zone: "residential"  # residential, datacenter, isp, mobile
  country: "us"
  session_id: null
```

---

## 3. Oxylabs

**Website:** https://oxylabs.io

**Features:**
- 100M+ proxy pool
- Web Scraper API
- SERP Scraper API
- E-commerce Scraper API
- Real-time crawler

**Products:**
| Product | Best For |
|---------|----------|
| Residential Proxies | General |
| Datacenter Proxies | Speed |
| Web Scraper API | Turnkey solution |
| SERP API | Search engines |
| E-commerce API | Amazon, eBay |

**Pricing:** Custom (from $99/mo for proxies)

**Implementation:**
```python
# Module: providers/scrapers/oxylabs_provider.py
class OxylabsProvider(BaseScraperProvider):
    def __init__(self, username: str, password: str):
        self.credentials = (username, password)
        self.api_url = "https://realtime.oxylabs.io/v1/queries"

    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        payload = {
            "source": "universal",
            "url": url,
            "render": kwargs.get("render", "html")
        }
        # Implementation...
```

---

## 4. Apify

**Website:** https://apify.com

**Features:**
- 2,000+ pre-built scrapers (Actors)
- Visual scraper builder
- Scheduled runs
- Proxy integration
- API + SDK

**Notable Actors:**
| Actor | Platform |
|-------|----------|
| Amazon Scraper | Amazon reviews |
| Instagram Scraper | Instagram posts |
| Google Maps Scraper | Reviews |
| Twitter Scraper | Tweets |
| YouTube Scraper | Comments |
| Tripadvisor Scraper | Reviews |

**Pricing:**
| Plan | Compute Units | Price |
|------|---------------|-------|
| Free | 5/mo | Free |
| Starter | 49/mo | $49/mo |
| Scale | 499/mo | $499/mo |

**Implementation:**
```python
# Module: providers/scrapers/apify_provider.py
class ApifyProvider(BaseScraperProvider):
    def __init__(self, api_token: str):
        self.client = ApifyClient(api_token)

    async def run_actor(self, actor_id: str, input_data: dict) -> dict:
        run = self.client.actor(actor_id).call(run_input=input_data)
        return self.client.dataset(run["defaultDatasetId"]).list_items().items
```

**Configuration:**
```yaml
apify:
  api_token: "${APIFY_TOKEN}"
  actors:
    amazon: "junglee/amazon-reviews-scraper"
    youtube: "streamers/youtube-scraper"
    instagram: "apify/instagram-scraper"
  memory_mbytes: 1024
  timeout_secs: 300
```

---

## 5. Zyte (Scrapy Cloud)

**Website:** https://www.zyte.com

**Features:**
- Scrapy Cloud hosting
- Smart Proxy Manager
- Automatic Extraction
- AI-powered parsing

**Products:**
| Product | Description |
|---------|-------------|
| Scrapy Cloud | Host Scrapy spiders |
| Smart Proxy | Intelligent rotation |
| Zyte API | Turnkey scraping |
| AutoExtract | ML-based extraction |

**Pricing:** From $450/mo

**Implementation:**
```python
# Module: providers/scrapers/zyte_provider.py
class ZyteProvider(BaseScraperProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.zyte.com/v1/extract"

    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        payload = {
            "url": url,
            "browserHtml": True,
            "javascript": kwargs.get("javascript", True)
        }
        # Implementation...
```

---

## 6. ScrapingBee

**Website:** https://www.scrapingbee.com

**Features:**
- JavaScript rendering
- Proxy rotation
- Screenshot API
- Google Search API

**Pricing:**
| Plan | Credits | Price |
|------|---------|-------|
| Freelance | 1,000 | $49/mo |
| Startup | 10,000 | $99/mo |
| Business | 50,000 | $249/mo |

**Implementation:**
```python
# Module: providers/scrapers/scrapingbee_provider.py
class ScrapingBeeProvider(BaseScraperProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        params = {
            "api_key": self.api_key,
            "url": url,
            "render_js": kwargs.get("render", True)
        }
        # Implementation...
```

---

## 7. Comparison Matrix

| Service | Proxy Pool | JS Render | CAPTCHA | Min Price | Best For |
|---------|------------|-----------|---------|-----------|----------|
| ScraperAPI | 40M+ | Yes | Yes | Free | Startups |
| Bright Data | 72M+ | Yes | Yes | $500/mo | Enterprise |
| Oxylabs | 100M+ | Yes | Yes | $99/mo | Large scale |
| Apify | N/A | Actors | Varies | Free | Pre-built scrapers |
| Zyte | 50M+ | Yes | Yes | $450/mo | Scrapy users |
| ScrapingBee | 1M+ | Yes | No | $49/mo | Simple needs |

---

## Selection Criteria

### Choose ScraperAPI if:
- Starting out / limited budget
- Need simple API integration
- Moderate volume (< 250K requests/mo)

### Choose Bright Data if:
- Enterprise requirements
- Need dedicated support
- Complex anti-bot sites
- High volume

### Choose Oxylabs if:
- E-commerce focus
- Need specialized APIs
- High reliability required

### Choose Apify if:
- Want pre-built scrapers
- Need specific platform support
- Prefer visual tools

### Choose Zyte if:
- Already using Scrapy
- Need ML-based extraction
- Want cloud spider hosting

---

## Fallback Strategy

Configure multiple providers with automatic fallback:

```yaml
scrapers:
  fallback_chain:
    - provider: "playwright"
      max_retries: 2
    - provider: "scraperapi"
      max_retries: 2
    - provider: "brightdata"
      max_retries: 1

  fallback_triggers:
    - status_code: 403
    - status_code: 429
    - exception: "CaptchaError"
    - exception: "BlockedError"
```
