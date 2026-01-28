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
| ScraperAPI | P1 | Planned | [ ] | [ ] | [ ] | From $49/mo | JS render, CAPTCHA |
| Bright Data | P2 | Planned | [ ] | [ ] | [ ] | From $500/mo | 72M+ proxies |
| Oxylabs | P2 | Planned | [ ] | [ ] | [ ] | Custom | 100M+ proxies |
| Apify | P1 | Planned | [ ] | [ ] | [ ] | Pay-per-use | 2000+ actors |
| Zyte | P2 | Planned | [ ] | [ ] | [ ] | From $450/mo | Scrapy Cloud |
| ScrapingBee | P2 | Planned | [ ] | [ ] | [ ] | From $49/mo | Simple API |
| ScrapingAnt | P3 | Planned | [ ] | [ ] | [ ] | From $19/mo | Budget option |

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
| IMDB | P1 | Planned | [ ] | [ ] | [ ] | Selenium/OMDb | Movie reviews |
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
| Yelp | P1 | Planned | [ ] | [ ] | [ ] | Playwright | Businesses |
| Trustpilot | P1 | Planned | [ ] | [ ] | [ ] | Playwright | Companies |
| Google Reviews | P1 | Planned | [ ] | [ ] | [ ] | SerpAPI/Places | Local businesses |
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
| Commercial APIs | 7 | 0 | 2 | 4 | 1 | 0 | 0 |
| AI Scrapers | 4 | 0 | 0 | 2 | 2 | 0 | 0 |
| E-Commerce | 8 | 1 | 0 | 2 | 5 | 1 | 1 |
| Entertainment | 6 | 0 | 3 | 2 | 1 | 0 | 0 |
| Gaming | 6 | 1 | 0 | 0 | 5 | 1 | 1 |
| Social Media | 10 | 2 | 1 | 3 | 4 | 2 | 2 |
| Review Platforms | 10 | 0 | 4 | 2 | 4 | 0 | 0 |
| App Stores | 3 | 0 | 2 | 0 | 1 | 0 | 0 |
| Professional | 4 | 0 | 0 | 2 | 2 | 0 | 0 |
| News/Forums | 5 | 0 | 0 | 2 | 3 | 0 | 0 |
| Food | 5 | 0 | 0 | 0 | 5 | 0 | 0 |
| Travel | 4 | 0 | 0 | 2 | 2 | 0 | 0 |
| **Total** | **79** | **7** | **15** | **22** | **35** | **7** | **7** |

---

## Implementation Order

### Phase 1 (P0) - Core
1. Playwright scraper
2. HTTPX scraper
3. BeautifulSoup parser
4. Amazon scraper
5. Steam scraper
6. YouTube scraper
7. Reddit scraper

### Phase 2 (P1) - Important
1. Selenium scraper
2. Requests HTTP
3. lxml parser
4. ScraperAPI integration
5. Apify integration
6. IMDB, Rotten Tomatoes, Metacritic
7. Twitter/X
8. Yelp, Trustpilot, Google Reviews, Tripadvisor
9. App Store, Play Store

### Phase 3 (P2) - Extended
- Remaining P2 platforms

### Phase 4 (P3) - As Needed
- Based on user requests

---

## Notes

- Update checkboxes as platforms are implemented
- Add new platforms as they become relevant
- Mark deprecated/blocked platforms
- Track API changes and updates
