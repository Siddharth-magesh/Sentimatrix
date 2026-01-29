# Sentimatrix V2 - Scraper & Platform Tracking

## Status Legend

| Status | Meaning |
|--------|---------|
| Planned | Not started |
| In Progress | Currently being implemented |
| Implemented | Code complete |
| Tested | Unit tests passing |
| Working | Integration tested and verified |
| Stable | Production ready |

---

## Scraping Libraries/Tools

| Tool | Type | Priority | Status | Implemented | Tested | Working | Notes |
|------|------|----------|--------|-------------|--------|---------|-------|
| Playwright | Browser | P0 | Working | [x] | [x] | [x] | Default, async, stealth mode |
| Selenium | Browser | P1 | Planned | [ ] | [ ] | [ ] | Legacy support |
| HTTPX | HTTP | P0 | Working | [x] | [x] | [x] | Async HTTP with rate limiting |
| Requests | HTTP | P1 | Planned | [ ] | [ ] | [ ] | Simple HTTP |
| BeautifulSoup | Parser | P0 | Working | [x] | [x] | [x] | HTML parsing (integrated) |
| lxml | Parser | P1 | Planned | [ ] | [ ] | [ ] | Fast parsing |
| Scrapy | Framework | P2 | Planned | [ ] | [ ] | [ ] | Full framework |

---

## Commercial Scraping APIs

| Service | Priority | Status | Implemented | Tested | Working | Pricing | Notes |
|---------|----------|--------|-------------|--------|---------|---------|-------|
| ScraperAPI | P1 | Working | [x] | [x] | [x] | From $49/mo | JS render, CAPTCHA, geo-targeting |
| Bright Data | P2 | Working | [x] | [x] | [x] | From $500/mo | 72M+ proxies, SERP, platform scrapers |
| Oxylabs | P2 | Working | [x] | [x] | [x] | From $49/mo | 100M+ proxies, e-commerce, SERP |
| Apify | P1 | Working | [x] | [x] | [x] | Pay-per-use | 2000+ actors, datasets, scheduling |
| Zyte | P2 | Working | [x] | [x] | [x] | From $450/mo | AI extraction, Scrapy integration |
| ScrapingBee | P2 | Working | [x] | [x] | [x] | From $49/mo | Simple API, AI extraction, screenshots |
| ScrapingAnt | P3 | Working | [x] | [x] | [x] | From $19/mo | Budget option, markdown output |

---

## AI-Powered Scrapers

| Tool | Type | Priority | Status | Implemented | Tested | Working | Notes |
|------|------|----------|--------|-------------|--------|---------|-------|
| Firecrawl | API | P2 | Planned | [ ] | [ ] | [ ] | LLM-ready output |
| Crawl4AI | OSS | P2 | Planned | [ ] | [ ] | [ ] | GPT-powered |
| ScrapeGraphAI | OSS | P3 | Planned | [ ] | [ ] | [ ] | NL queries |
| Skyvern | OSS | P3 | Planned | [ ] | [ ] | [ ] | Visual AI |

---

## E-Commerce Platforms

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| Amazon | P0 | Working | [x] | [x] | [x] | Playwright | Product reviews, ASIN support |
| Walmart | P2 | Planned | [ ] | [ ] | [ ] | Playwright | US retail |
| eBay | P2 | Planned | [ ] | [ ] | [ ] | API/Playwright | Auctions |
| Etsy | P3 | Planned | [ ] | [ ] | [ ] | API | Handmade |
| AliExpress | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Chinese e-com |
| Best Buy | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Electronics |
| Target | P3 | Planned | [ ] | [ ] | [ ] | Playwright | US retail |
| Flipkart | P3 | Planned | [ ] | [ ] | [ ] | Playwright | India |

---

## Entertainment/Media

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| IMDB | P1 | Working | [x] | [x] | [x] | Playwright/OMDb API | Movie/TV reviews, search, movie info |
| Rotten Tomatoes | P1 | Planned | [ ] | [ ] | [ ] | Playwright | Critic + audience |
| Metacritic | P1 | Planned | [ ] | [ ] | [ ] | Playwright | Aggregated scores |
| LetterBoxD | P2 | Planned | [ ] | [ ] | [ ] | Playwright | Film community |
| Goodreads | P2 | Planned | [ ] | [ ] | [ ] | Playwright | Book reviews |
| Spotify | P3 | Planned | [ ] | [ ] | [ ] | API | Music |

---

## Gaming Platforms

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| Steam | P0 | Working | [x] | [x] | [x] | Steam API | Reviews, game info, search |
| Epic Games | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Game store |
| GOG | P3 | Planned | [ ] | [ ] | [ ] | Playwright | DRM-free |
| PlayStation Store | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Console |
| Xbox Store | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Console |
| Nintendo | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Console |

---

## Social Media

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| YouTube | P0 | Working | [x] | [x] | [x] | YouTube API | Comments, transcripts, search |
| Reddit | P0 | Working | [x] | [x] | [x] | JSON API | Posts, comments, OAuth support |
| Twitter/X | P1 | Planned | [ ] | [ ] | [ ] | X API (paid) | Tweets |
| TikTok | P2 | Planned | [ ] | [ ] | [ ] | Third-party | Videos, comments |
| Instagram | P2 | Planned | [ ] | [ ] | [ ] | Third-party | Posts, comments |
| Facebook | P2 | Planned | [ ] | [ ] | [ ] | Graph API | Limited access |
| LinkedIn | P3 | Planned | [ ] | [ ] | [ ] | Difficult | Company reviews |
| Threads | P3 | Planned | [ ] | [ ] | [ ] | API | Meta |
| Mastodon | P3 | Planned | [ ] | [ ] | [ ] | API | Federated |
| Bluesky | P3 | Planned | [ ] | [ ] | [ ] | API | Decentralized |

---

## Review Platforms

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| Yelp | P1 | Working | [x] | [x] | [x] | Playwright/Fusion API | Business reviews, search |
| Trustpilot | P1 | Working | [x] | [x] | [x] | Playwright | Company reviews, search |
| Google Reviews | P1 | Working | [x] | [x] | [x] | Places API/SerpAPI | Place reviews, search |
| Tripadvisor | P1 | Planned | [ ] | [ ] | [ ] | Playwright | Travel |
| G2 | P2 | Planned | [ ] | [ ] | [ ] | Playwright | Software |
| Capterra | P2 | Planned | [ ] | [ ] | [ ] | Playwright | Software |
| BBB | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Business reviews |
| ConsumerAffairs | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Consumer reviews |
| Sitejabber | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Website reviews |
| PissedConsumer | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Complaints |

---

## App Stores

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| Apple App Store | P1 | Planned | [ ] | [ ] | [ ] | API/Scraper | iOS apps |
| Google Play Store | P1 | Planned | [ ] | [ ] | [ ] | Scraper | Android apps |
| Microsoft Store | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Windows apps |

---

## Professional/Jobs

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| Glassdoor | P2 | Planned | [ ] | [ ] | [ ] | Playwright | Company reviews |
| Indeed | P2 | Planned | [ ] | [ ] | [ ] | Playwright | Job reviews |
| Blind | P3 | Planned | [ ] | [ ] | [ ] | Difficult | Anonymous |
| Comparably | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Workplace |

---

## News/Forums

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| Hacker News | P2 | Planned | [ ] | [ ] | [ ] | API | Tech news |
| Product Hunt | P2 | Planned | [ ] | [ ] | [ ] | API | Product launches |
| Quora | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Q&A |
| Stack Overflow | P3 | Planned | [ ] | [ ] | [ ] | API | Developer Q&A |
| Medium | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Articles |

---

## Food/Restaurant

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| DoorDash | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Delivery |
| Uber Eats | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Delivery |
| Grubhub | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Delivery |
| OpenTable | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Reservations |
| Zomato | P3 | Planned | [ ] | [ ] | [ ] | Playwright | India |

---

## Travel/Hotels

| Platform | Priority | Status | Implemented | Tested | Working | Method | Notes |
|----------|----------|--------|-------------|--------|---------|--------|-------|
| Booking.com | P2 | Planned | [ ] | [ ] | [ ] | Playwright | Hotels |
| Airbnb | P2 | Planned | [ ] | [ ] | [ ] | Playwright | Rentals |
| Hotels.com | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Hotels |
| Expedia | P3 | Planned | [ ] | [ ] | [ ] | Playwright | Travel |

---

## Implementation Summary

| Category | Total | P0 | P1 | P2 | P3 | Implemented | Working |
|----------|-------|----|----|----|----|-------------|---------|
| Libraries/Tools | 7 | 3 | 3 | 1 | 0 | 3 | 3 |
| Commercial APIs | 7 | 0 | 2 | 4 | 1 | 7 | 7 |
| AI Scrapers | 4 | 0 | 0 | 2 | 2 | 0 | 0 |
| E-Commerce | 8 | 1 | 0 | 2 | 5 | 1 | 1 |
| Entertainment | 6 | 0 | 3 | 2 | 1 | 1 | 1 |
| Gaming | 6 | 1 | 0 | 0 | 5 | 1 | 1 |
| Social Media | 10 | 2 | 1 | 3 | 4 | 2 | 2 |
| Review Platforms | 10 | 0 | 4 | 2 | 4 | 3 | 3 |
| App Stores | 3 | 0 | 2 | 0 | 1 | 0 | 0 |
| Professional | 4 | 0 | 0 | 2 | 2 | 0 | 0 |
| News/Forums | 5 | 0 | 0 | 2 | 3 | 0 | 0 |
| Food | 5 | 0 | 0 | 0 | 5 | 0 | 0 |
| Travel | 4 | 0 | 0 | 2 | 2 | 0 | 0 |
| **Total** | **79** | **7** | **15** | **22** | **35** | **18** | **18** |

---

## Implementation Order

### Phase 1 (P0) - Core ✅ COMPLETE
1. ✅ Playwright scraper - JS rendering, stealth mode, multi-browser
2. ✅ HTTPX scraper - Async HTTP/2, connection pooling
3. ✅ BeautifulSoup parser - HTML parsing (integrated)
4. ✅ Amazon scraper - Product reviews, ASIN support, pagination
5. ✅ Steam scraper - Steam API, reviews, game info, search
6. ✅ YouTube scraper - Data API v3, comments, transcripts
7. ✅ Reddit scraper - JSON API, posts, comments, OAuth ready

### Phase 2 (P1) - Important
1. ✅ IMDB - Movie/TV reviews with OMDb API support
2. ✅ Yelp - Business reviews with Fusion API support
3. ✅ Trustpilot - Company reviews via Playwright
4. ✅ Google Reviews - Place reviews via Places API/SerpAPI
5. ✅ ScraperAPI integration - JS rendering, CAPTCHA, geo-targeting
6. ✅ Apify integration - 2000+ actors, datasets, scheduling
7. Selenium scraper
8. Requests HTTP
9. lxml parser
10. Rotten Tomatoes, Metacritic
11. Twitter/X
12. Tripadvisor
13. App Store, Play Store

### Phase 2.5 (P2 Commercial APIs) ✅ COMPLETE
1. ✅ Bright Data - 72M+ proxies, SERP, platform scrapers
2. ✅ Oxylabs - 100M+ proxies, e-commerce specialization
3. ✅ Zyte - AI extraction, Scrapy integration
4. ✅ ScrapingBee - Simple API, AI extraction, screenshots
5. ✅ ScrapingAnt - Budget option, markdown output

### Phase 3 (P2) - Extended
- Remaining P2 platforms

### Phase 4 (P3) - As Needed
- Based on user requests

**Phase 1 Progress:** 7/7 complete (100%)
**Phase 2 Progress:** 6/13 complete (46%)
**Commercial APIs Progress:** 7/7 complete (100%)

---

## Scraper Files

### Core Scrapers
| Scraper | File |
|---------|------|
| HTTPXScraper | `sentimatrix/providers/scrapers/httpx_scraper.py` |
| PlaywrightScraper | `sentimatrix/providers/scrapers/playwright_scraper.py` |
| RateLimiter | `sentimatrix/providers/scrapers/rate_limiter.py` |
| Utils (Proxy, UA, Retry) | `sentimatrix/providers/scrapers/utils.py` |

### Platform Scrapers
| Scraper | File |
|---------|------|
| BasePlatformScraper | `sentimatrix/providers/scrapers/platforms/base.py` |
| AmazonScraper | `sentimatrix/providers/scrapers/platforms/amazon.py` |
| SteamScraper | `sentimatrix/providers/scrapers/platforms/steam.py` |
| YouTubeScraper | `sentimatrix/providers/scrapers/platforms/youtube.py` |
| RedditScraper | `sentimatrix/providers/scrapers/platforms/reddit.py` |
| IMDBScraper | `sentimatrix/providers/scrapers/platforms/imdb.py` |
| YelpScraper | `sentimatrix/providers/scrapers/platforms/yelp.py` |
| TrustpilotScraper | `sentimatrix/providers/scrapers/platforms/trustpilot.py` |
| GoogleReviewsScraper | `sentimatrix/providers/scrapers/platforms/google_reviews.py` |

### Commercial API Clients
| Client | File | Features |
|--------|------|----------|
| ScraperAPIClient | `sentimatrix/providers/scrapers/commercial/scraper_api.py` | JS rendering, CAPTCHA, geo-targeting, screenshots |
| ApifyClient | `sentimatrix/providers/scrapers/commercial/apify.py` | 2000+ actors, datasets, webhooks |
| BrightDataClient | `sentimatrix/providers/scrapers/commercial/bright_data.py` | 72M+ proxies, SERP, platform scrapers |
| OxylabsClient | `sentimatrix/providers/scrapers/commercial/oxylabs.py` | E-commerce, SERP, real-time/batch |
| ZyteClient | `sentimatrix/providers/scrapers/commercial/zyte.py` | AI extraction, browser actions, Scrapy |
| ScrapingBeeClient | `sentimatrix/providers/scrapers/commercial/scrapingbee.py` | AI extraction, screenshots, JS scenarios |
| ScrapingAntClient | `sentimatrix/providers/scrapers/commercial/scrapingant.py` | Budget option, markdown output |

---

## Test Coverage

**Total Tests: 282 passing** (216 scraper + 26 platform + 40 commercial API tests)

| Test File | Tests | Status |
|-----------|-------|--------|
| test_httpx_scraper.py | HTTPXScraper tests | Passing |
| test_playwright_scraper.py | PlaywrightScraper tests | Passing |
| test_rate_limiter.py | Rate limiter tests | Passing |
| test_utils.py | Utility function tests | Passing |
| test_base.py | BasePlatformScraper tests | Passing |
| test_amazon.py | AmazonScraper tests | Passing |
| test_steam.py | SteamScraper tests | Passing |
| test_youtube.py | YouTubeScraper tests | Passing |
| test_reddit.py | RedditScraper tests | Passing |
| test_new_platforms.py | IMDB, Yelp, Trustpilot, Google Reviews tests (26) | Passing |
| test_commercial_apis.py | All commercial API clients tests (40) | Passing |

---

## Notes

- Update checkboxes as platforms are implemented
- Add new platforms as they become relevant
- Mark deprecated/blocked platforms
- Track API changes and updates
