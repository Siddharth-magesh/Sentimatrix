# Sentimatrix V2 - AI-Powered Scrapers

## Overview

AI-powered scrapers use LLMs to intelligently extract structured data from web pages. They require minimal configuration and can adapt to page changes automatically.

---

## 1. Firecrawl

**Website:** https://firecrawl.dev

**Type:** Commercial API

**Features:**
- Automatic content extraction
- Markdown conversion
- Screenshot capture
- LLM-ready output
- Batch crawling
- Sitemap discovery

**How It Works:**
1. Send URL to Firecrawl API
2. Firecrawl renders and extracts content
3. Returns clean markdown/structured data
4. Optionally applies LLM for extraction

**Pricing:**
| Plan | Credits/mo | Price |
|------|------------|-------|
| Free | 500 | $0 |
| Hobby | 3,000 | $16/mo |
| Standard | 100,000 | $83/mo |
| Growth | 500,000 | $333/mo |

**Implementation:**
```python
# Module: providers/scrapers/firecrawl_provider.py
class FirecrawlProvider(BaseScraperProvider):
    def __init__(self, api_key: str):
        self.client = FirecrawlApp(api_key=api_key)

    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        result = self.client.scrape_url(
            url,
            params={
                "formats": ["markdown", "html"],
                "onlyMainContent": True
            }
        )
        return ScrapedContent(
            url=url,
            content=result.get("markdown"),
            html=result.get("html")
        )

    async def extract_structured(self, url: str, schema: dict) -> dict:
        """Extract data according to a JSON schema"""
        result = self.client.scrape_url(
            url,
            params={
                "formats": ["extract"],
                "extract": {"schema": schema}
            }
        )
        return result.get("extract")
```

**Configuration:**
```yaml
firecrawl:
  api_key: "${FIRECRAWL_KEY}"
  formats:
    - markdown
    - html
  only_main_content: true
  wait_for: null  # CSS selector to wait for
  timeout: 30000
```

**Use Case - Review Extraction:**
```python
schema = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "rating": {"type": "number"},
                    "author": {"type": "string"},
                    "date": {"type": "string"}
                }
            }
        }
    }
}
reviews = await firecrawl.extract_structured(url, schema)
```

---

## 2. Crawl4AI

**Repository:** https://github.com/unclecode/crawl4ai

**Type:** Open Source

**Features:**
- Async browser automation
- LLM-powered extraction
- Multiple extraction strategies
- Chunking strategies
- Session management
- Proxy support

**Extraction Strategies:**
| Strategy | Use Case |
|----------|----------|
| LLMExtractionStrategy | AI-powered extraction |
| JsonCssExtractionStrategy | CSS selector-based |
| NoExtractionStrategy | Raw content only |

**Implementation:**
```python
# Module: providers/scrapers/crawl4ai_provider.py
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class Crawl4AIProvider(BaseScraperProvider):
    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(url=url)
            return ScrapedContent(
                url=url,
                content=result.markdown,
                html=result.html
            )

    async def extract_with_llm(self, url: str, instruction: str) -> dict:
        strategy = LLMExtractionStrategy(
            provider="openai/gpt-4o-mini",
            api_token=self.api_key,
            instruction=instruction
        )
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
                extraction_strategy=strategy
            )
            return json.loads(result.extracted_content)
```

**Configuration:**
```yaml
crawl4ai:
  llm_provider: "openai/gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
  headless: true
  verbose: false
  extraction_strategy: "llm"  # llm, css, none
```

---

## 3. ScrapeGraphAI

**Repository:** https://github.com/VinciGit00/Scrapegraph-ai

**Type:** Open Source

**Features:**
- Natural language scraping queries
- Multiple LLM support (OpenAI, Ollama, etc.)
- Graph-based pipelines
- Built-in browser automation

**Scraping Modes:**
| Mode | Description |
|------|-------------|
| SmartScraperGraph | Single page extraction |
| SearchGraph | Search + scrape |
| SpeechGraph | Audio extraction |
| ScriptCreatorGraph | Generate reusable scripts |

**Implementation:**
```python
# Module: providers/scrapers/scrapegraphai_provider.py
from scrapegraphai.graphs import SmartScraperGraph

class ScrapeGraphAIProvider(BaseScraperProvider):
    def __init__(self, llm_config: dict):
        self.llm_config = llm_config

    async def scrape_with_query(self, url: str, query: str) -> dict:
        graph = SmartScraperGraph(
            prompt=query,
            source=url,
            config={"llm": self.llm_config}
        )
        result = graph.run()
        return result
```

**Configuration:**
```yaml
scrapegraphai:
  llm:
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"
  # Or use local model:
  # llm:
  #   model: "ollama/llama3"
  #   base_url: "http://localhost:11434"
  headless: true
  verbose: false
```

**Example Query:**
```python
result = await provider.scrape_with_query(
    url="https://example.com/product/reviews",
    query="Extract all customer reviews with their ratings, dates, and review text"
)
```

---

## 4. LLM Scraper (by Mendable)

**Repository:** https://github.com/mendableai/llm-scraper

**Type:** Open Source (TypeScript, Python wrapper)

**Features:**
- Zod schema-based extraction
- Playwright integration
- TypeScript-first

**Note:** Primarily TypeScript; use via subprocess or API wrapper.

---

## 5. Skyvern

**Repository:** https://github.com/Skyvern-AI/skyvern

**Type:** Open Source

**Features:**
- Browser automation with GPT-4V
- Visual understanding
- Complex workflows
- Form filling
- Navigation

**Use Case:** Complex multi-step scraping requiring visual understanding.

---

## Comparison Matrix

| Tool | Type | LLM Provider | Async | Cost |
|------|------|--------------|-------|------|
| Firecrawl | API | Built-in | Yes | Paid |
| Crawl4AI | OSS | Any | Yes | Free + LLM costs |
| ScrapeGraphAI | OSS | Any | Partial | Free + LLM costs |
| Skyvern | OSS | OpenAI (vision) | Yes | Free + LLM costs |

---

## When to Use AI Scrapers

**Best For:**
- Pages with complex/variable structures
- Unstructured content extraction
- Natural language queries
- Rapid prototyping
- Sites that change frequently

**Not Ideal For:**
- High-volume scraping (cost)
- Simple, structured pages
- Real-time requirements
- Budget-constrained projects

---

## Cost Analysis

**Per-Page Cost Estimate:**

| Tool | Estimated Cost/Page |
|------|---------------------|
| Firecrawl (with extraction) | $0.002-0.01 |
| Crawl4AI + GPT-4o-mini | $0.001-0.005 |
| ScrapeGraphAI + GPT-4o-mini | $0.001-0.005 |
| ScrapeGraphAI + Ollama | ~$0 (compute only) |

**Volume Considerations:**
- 1K pages/day with GPT-4o-mini: ~$1-5/day
- 10K pages/day: ~$10-50/day
- Consider local LLMs for high volume

---

## Hybrid Approach

Combine AI scrapers with traditional methods:

```yaml
scrapers:
  strategy: "hybrid"

  rules:
    - pattern: "amazon.com/*"
      scraper: "amazon_platform"  # Use specialized scraper

    - pattern: "*.reviews.*"
      scraper: "crawl4ai"  # Use AI for unknown review sites

    - pattern: "*"
      scraper: "playwright"  # Default to browser scraping
      fallback: "crawl4ai"   # AI fallback if extraction fails
```

---

## Best Practices

1. **Cache LLM Extractions** - Same structure pages don't need re-extraction
2. **Use Schemas** - Define expected output structure for consistency
3. **Local LLMs for Volume** - Use Ollama for cost-sensitive applications
4. **Validate Outputs** - AI can hallucinate; validate extracted data
5. **Monitor Costs** - Track token usage and costs per scrape
6. **Fallback Chains** - Use AI as fallback when traditional methods fail
