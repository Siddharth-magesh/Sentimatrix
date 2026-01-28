"""
Sample Test Data

Contains sample data for testing various modules.
"""

# Sample review texts for sentiment analysis testing
POSITIVE_REVIEWS = [
    "This product is absolutely amazing! Best purchase I've ever made.",
    "Exceeded all my expectations. Highly recommend to everyone!",
    "Perfect quality and fast shipping. Will definitely buy again.",
    "Love it! Works exactly as described, maybe even better.",
    "Outstanding customer service and product quality.",
]

NEGATIVE_REVIEWS = [
    "Terrible experience. Product broke after one day.",
    "Worst purchase ever. Complete waste of money.",
    "Do not buy this! It's cheaply made garbage.",
    "Customer service was unhelpful and rude.",
    "Arrived damaged and took forever to ship.",
]

NEUTRAL_REVIEWS = [
    "It's okay, nothing special but does the job.",
    "Average product. Not great, not terrible.",
    "Works as expected, no complaints.",
    "Standard quality for the price point.",
    "Does what it's supposed to do.",
]

MIXED_REVIEWS = [
    "Great quality but shipping was slow.",
    "Love the product but customer service needs improvement.",
    "Works well but overpriced for what you get.",
    "Nice design but poor battery life.",
    "Good value but instructions were confusing.",
]

# Sample review objects for scraper testing
SAMPLE_REVIEW_OBJECTS = [
    {
        "id": "rev_001",
        "text": "This laptop is incredible! Fast, lightweight, and the display is stunning.",
        "rating": 5.0,
        "author": "TechEnthusiast",
        "platform": "amazon",
        "timestamp": "2024-01-15T10:30:00Z",
        "metadata": {
            "verified_purchase": True,
            "helpful_votes": 42,
            "product_id": "B0123456789",
        },
    },
    {
        "id": "rev_002",
        "text": "Disappointed. Battery died within 3 months.",
        "rating": 1.0,
        "author": "UnhappyBuyer",
        "platform": "amazon",
        "timestamp": "2024-02-01T15:45:00Z",
        "metadata": {
            "verified_purchase": True,
            "helpful_votes": 28,
            "product_id": "B0123456789",
        },
    },
    {
        "id": "rev_003",
        "text": "Good game with minor bugs. Enjoying it so far.",
        "rating": 4.0,
        "author": "CasualGamer",
        "platform": "steam",
        "timestamp": "2024-01-20T09:00:00Z",
        "metadata": {
            "playtime_hours": 25,
            "recommended": True,
        },
    },
]

# Sample configuration data for testing
SAMPLE_CONFIG_YAML = """
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: sk-test-key-12345
  temperature: 0.7
  max_tokens: 2048

scrapers:
  provider: playwright
  headless: true
  timeout: 30

models:
  sentiment_model: cardiffnlp/twitter-roberta-base-sentiment-latest
  emotion_model: SamLowe/roberta-base-go_emotions
  device: cpu

cache:
  enabled: true
  backend: memory
  ttl: 3600
  max_size: 1000

logging:
  level: INFO
  format: json
  console_output: true
"""

SAMPLE_CONFIG_JSON = {
    "llm": {
        "provider": "anthropic",
        "model": "claude-3-sonnet",
        "api_key": "test-key",
        "temperature": 0.5,
    },
    "cache": {
        "enabled": True,
        "backend": "memory",
        "ttl": 7200,
    },
    "debug": True,
}

# Sample URLs for scraper testing
SAMPLE_URLS = {
    "amazon": "https://www.amazon.com/dp/B0123456789",
    "steam": "https://store.steampowered.com/app/123456",
    "youtube": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "reddit": "https://www.reddit.com/r/technology/comments/abc123/",
    "imdb": "https://www.imdb.com/title/tt1234567/",
}

# Expected emotion labels (GoEmotions taxonomy)
EMOTION_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

# Sample LLM responses for mocking
SAMPLE_LLM_RESPONSES = {
    "sentiment_analysis": {
        "content": '{"sentiment": "positive", "confidence": 0.95, "reasoning": "The text expresses strong satisfaction."}',
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
    },
    "summary": {
        "content": "The reviews are generally positive, with customers praising product quality and fast shipping. Some concerns about pricing.",
        "model": "gpt-4o-mini",
        "usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
    },
    "comparison": {
        "content": "Product A has better reviews overall with 4.5 stars vs Product B's 3.8 stars. Product A excels in quality while Product B offers better value.",
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
    },
}

# Error scenarios for testing error handling
ERROR_SCENARIOS = {
    "rate_limit": {
        "status_code": 429,
        "message": "Rate limit exceeded",
        "retry_after": 60,
    },
    "auth_failure": {
        "status_code": 401,
        "message": "Invalid API key",
    },
    "server_error": {
        "status_code": 500,
        "message": "Internal server error",
    },
    "timeout": {
        "timeout_seconds": 30,
        "message": "Request timed out",
    },
}
