# Sentimatrix V2 - Platform-Specific Scrapers

## Overview

Platform scrapers are specialized modules for extracting reviews from specific websites. Each scraper handles platform-specific authentication, pagination, and data extraction.

---

## E-Commerce Platforms

### 1. Amazon

**Methods:**
| Method | Reliability | Speed | Setup |
|--------|-------------|-------|-------|
| Direct Scraping | Low | Fast | None |
| Product API | High | Fast | Associate account |
| Rainforest API | High | Medium | API key |
| Apify Actor | High | Medium | API key |

**Data Extracted:**
- Review text
- Star rating
- Review date
- Verified purchase status
- Helpful votes
- Product variant

**Implementation:**
```python
# Module: providers/scrapers/platforms/amazon.py
class AmazonScraper(BasePlatformScraper):
    async def scrape_reviews(self, product_url: str, limit: int = 100) -> List[Review]
    async def scrape_by_asin(self, asin: str, limit: int = 100) -> List[Review]
    async def get_product_info(self, asin: str) -> ProductInfo
```

**Configuration:**
```yaml
amazon:
  method: "playwright"  # playwright, scraperapi, rainforest, apify
  country: "us"  # us, uk, de, fr, jp, etc.
  sort: "recent"  # recent, helpful
  filter_verified: false
  rainforest_api_key: "${RAINFOREST_KEY}"
```

---

### 2. Walmart

**Methods:**
- Playwright scraping
- Walmart API (affiliate)
- Third-party APIs

**Data Extracted:**
- Review text
- Rating
- Review title
- Author
- Date
- Verified purchase

---

### 3. eBay

**Methods:**
- Direct scraping
- eBay Browse API
- Third-party services

**Data Extracted:**
- Feedback text
- Rating (positive/negative/neutral)
- Date
- Item details

---

### 4. Etsy

**Methods:**
- Playwright scraping
- Etsy Open API (limited)

**Data Extracted:**
- Review text
- Rating
- Photos
- Transaction date

---

## Entertainment Platforms

### 5. IMDB

**Methods:**
| Method | Reliability | Data Scope |
|--------|-------------|------------|
| Selenium scraping | Medium | Full |
| OMDb API | High | Limited |
| Apify Actor | High | Full |

**Data Extracted:**
- Review text
- Rating (1-10)
- Review title
- Spoiler flag
- Helpful count
- Date

**Implementation:**
```python
# Module: providers/scrapers/platforms/imdb.py
class IMDBScraper(BasePlatformScraper):
    async def scrape_reviews(self, movie_url: str, limit: int = 100) -> List[Review]
    async def search_movie(self, title: str) -> MovieInfo
    async def get_ratings(self, movie_id: str) -> RatingInfo
```

---

### 6. Rotten Tomatoes

**Methods:**
- Playwright scraping (load more button)
- No official API

**Data Extracted:**
- Critic reviews
- Audience reviews
- Tomatometer score
- Audience score
- Review text
- Fresh/Rotten designation

---

### 7. LetterBoxD

**Methods:**
- Playwright scraping
- No official API

**Data Extracted:**
- Review text
- Rating (half-star scale)
- Like count
- Date watched

---

### 8. Metacritic

**Methods:**
- Playwright scraping
- No official API

**Data Extracted:**
- Critic reviews
- User reviews
- Metascore
- User score
- Review text

---

## Gaming Platforms

### 9. Steam

**Methods:**
| Method | Reliability | Rate Limit |
|--------|-------------|------------|
| Steam Store API | High | Moderate |
| Steam Reviews API | High | Moderate |
| Direct scraping | Medium | N/A |

**Data Extracted:**
- Review text
- Recommendation (positive/negative)
- Playtime
- Helpful votes
- Funny votes
- Early access flag
- Date

**Implementation:**
```python
# Module: providers/scrapers/platforms/steam.py
class SteamScraper(BasePlatformScraper):
    async def scrape_reviews(self, app_id: int, limit: int = 100) -> List[Review]
    async def search_game(self, query: str) -> List[GameInfo]
    async def get_app_details(self, app_id: int) -> GameDetails
```

**Configuration:**
```yaml
steam:
  language: "english"
  filter: "all"  # all, positive, negative
  purchase_type: "all"  # all, steam, non_steam
  day_range: null  # null for all time
```

---

## Social Media Platforms

### 10. YouTube

**Methods:**
| Method | Reliability | Quota |
|--------|-------------|-------|
| YouTube Data API | High | 10K units/day |
| youtube-transcript-api | High | No limit |
| Apify Actor | High | Pay-per-use |

**Data Extracted:**
- Comments
- Replies
- Like count
- Author
- Timestamp
- Video transcript

**Implementation:**
```python
# Module: providers/scrapers/platforms/youtube.py
class YouTubeScraper(BasePlatformScraper):
    async def get_comments(self, video_id: str, limit: int = 100) -> List[Comment]
    async def get_transcript(self, video_id: str) -> Transcript
    async def search_videos(self, query: str, limit: int = 10) -> List[VideoInfo]
```

---

### 11. Reddit

**Methods:**
| Method | Reliability | Rate Limit |
|--------|-------------|------------|
| PRAW (Reddit API) | High | 60 req/min |
| Pushshift API | Medium | Varies |
| Apify Actor | High | Pay-per-use |

**Data Extracted:**
- Post title
- Post body
- Comments
- Score
- Awards
- Subreddit

**Implementation:**
```python
# Module: providers/scrapers/platforms/reddit.py
class RedditScraper(BasePlatformScraper):
    async def search_posts(self, query: str, subreddit: str = None) -> List[Post]
    async def get_comments(self, post_url: str) -> List[Comment]
    async def get_subreddit_posts(self, subreddit: str, limit: int) -> List[Post]
```

---

### 12. Twitter/X

**Methods:**
| Method | Status | Access |
|--------|--------|--------|
| X API v2 | Active | Paid ($100/mo basic) |
| Nitter instances | Unstable | Free |
| Apify Actor | Active | Pay-per-use |
| Third-party APIs | Active | Varies |

**Data Extracted:**
- Tweet text
- Retweet count
- Like count
- Reply count
- Author
- Media
- Hashtags

**Note:** X API changes frequently. Maintain multiple fallback methods.

---

### 13. TikTok

**Methods:**
| Method | Status |
|--------|--------|
| TikTok Research API | Limited access |
| Unofficial APIs | Unstable |
| Apify Actor | Active |
| Bright Data | Active |

**Data Extracted:**
- Video description
- Comments
- Like count
- Share count
- View count

---

### 14. Instagram

**Methods:**
- Instagram Basic Display API (limited)
- Apify Actor
- Bright Data
- PhantomBuster

**Data Extracted:**
- Post caption
- Comments
- Like count
- Hashtags

---

### 15. Facebook

**Methods:**
- Facebook Graph API (limited)
- Apify Actor
- Bright Data

**Data Extracted:**
- Post text
- Comments
- Reactions
- Shares

---

## Review Platforms

### 16. Yelp

**Methods:**
- Yelp Fusion API (limited to 3 reviews)
- Playwright scraping
- Third-party services

**Data Extracted:**
- Review text
- Rating
- Date
- Photos
- Useful/Funny/Cool counts

---

### 17. Trustpilot

**Methods:**
- Playwright scraping
- No official API for reviews

**Data Extracted:**
- Review text
- Rating
- Title
- Date
- Verified status

---

### 18. Google Reviews

**Methods:**
- Places API (limited)
- SerpAPI
- Outscraper
- Apify

**Data Extracted:**
- Review text
- Rating
- Date
- Author
- Photos

---

### 19. Tripadvisor

**Methods:**
- Playwright scraping
- Content API (limited access)
- Apify Actor

**Data Extracted:**
- Review text
- Rating
- Travel type
- Date of visit
- Photos

---

## App Stores

### 20. Apple App Store

**Methods:**
- App Store Connect API
- app-store-scraper (npm)
- Playwright

**Data Extracted:**
- Review text
- Rating
- Version reviewed
- Date
- Title

---

### 21. Google Play Store

**Methods:**
- google-play-scraper (npm)
- Playwright

**Data Extracted:**
- Review text
- Rating
- Date
- Helpful count
- App version

---

## Professional Platforms

### 22. Glassdoor

**Methods:**
- Playwright scraping (requires login)
- No official API

**Data Extracted:**
- Review text
- Pros
- Cons
- Rating
- Job title
- Employment status

---

### 23. LinkedIn

**Methods:**
- LinkedIn API (very limited)
- Bright Data
- PhantomBuster

**Data Extracted:**
- Company reviews
- Recommendations

---

## Implementation Pattern

All platform scrapers follow a common interface:

```python
class BasePlatformScraper(ABC):
    @abstractmethod
    async def scrape_reviews(self, identifier: str, limit: int) -> List[Review]

    @abstractmethod
    def get_platform_name(self) -> str

    @abstractmethod
    def validate_url(self, url: str) -> bool

    async def scrape_with_fallback(self, identifier: str, limit: int) -> List[Review]:
        """Try multiple methods with automatic fallback"""
        pass
```
