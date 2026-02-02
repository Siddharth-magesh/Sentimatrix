---
title: Chain-of-Thought Reasoning
description: Deep analysis with reasoning steps
---

# Chain-of-Thought Reasoning

Use LLM reasoning capabilities for complex analysis that requires step-by-step thinking.

## Usage

```python
async with Sentimatrix(config) as sm:
    result = await sm.llm_analyze(
        reviews,
        prompt="""Analyze these product reviews and identify:
        1. Quality issues mentioned
        2. Comparison with competitors
        3. Suggestions for improvement

        Think step by step and explain your reasoning.
        """,
        output_format="json"
    )
```

## Custom Prompts

```python
result = await sm.llm_analyze(
    reviews,
    prompt="""You are a product analyst. Based on these reviews:

    1. Identify the top 3 complaints
    2. Rate the severity of each (1-10)
    3. Suggest specific product improvements

    Format as JSON with reasoning for each item.
    """,
    output_format="json"
)
```

## Use Cases

- Complex product analysis
- Root cause analysis
- Strategic recommendations
- Multi-factor comparisons

## Best Models for Reasoning

| Task | Recommended Model |
|------|-------------------|
| Quick analysis | Groq llama-3.3-70b |
| Deep reasoning | Claude 3.5 Sonnet |
| Detailed analysis | GPT-4o |
| Cost-effective | DeepSeek Chat |

## Related

- [LLM Features](index.md)
- [Summarization](summarization.md)
- [Insights](insights.md)
