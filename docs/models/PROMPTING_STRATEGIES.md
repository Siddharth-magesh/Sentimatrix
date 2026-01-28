# Sentimatrix V2 - Prompting Strategies

## Overview

This document covers advanced prompting strategies for LLM interactions in Sentimatrix V2.

---

## 1. Direct Prompting

**Description:** Simple prompt-response without explicit reasoning steps.

**When to Use:**
- Simple classification tasks
- Short text analysis
- Low-latency requirements
- Cost-sensitive applications

**Implementation:**
```python
class DirectStrategy(BaseStrategy):
    async def execute(self, prompt: str, **kwargs) -> str:
        return await self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=kwargs.get("max_tokens", 512)
        )
```

**Prompt Template:**
```
Analyze the sentiment of the following text and respond with only one word: positive, negative, or neutral.

Text: {text}

Sentiment:
```

**Pros:** Fast, cheap, simple
**Cons:** Limited reasoning, lower accuracy on complex tasks

---

## 2. Chain of Thought (CoT)

**Description:** Encourages step-by-step reasoning before final answer.

**When to Use:**
- Complex reasoning tasks
- Multi-step analysis
- When explainability is needed
- Aspect-based sentiment

**Implementation:**
```python
class ChainOfThoughtStrategy(BaseStrategy):
    async def execute(self, prompt: str, **kwargs) -> CoTResult:
        cot_prompt = f"""
{prompt}

Let's think through this step by step:
1. First, I'll identify the key elements...
2. Then, I'll analyze each element...
3. Finally, I'll synthesize my findings...

Step-by-step analysis:
"""
        response = await self.llm.generate(
            prompt=cot_prompt,
            temperature=0.5,
            max_tokens=2048
        )
        return self._parse_cot_response(response)
```

**Prompt Template:**
```
Analyze the sentiment of the following product review.

Review: {text}

Think through this step by step:
1. Identify the main topics discussed
2. For each topic, determine if the sentiment is positive, negative, or neutral
3. Consider the overall tone and emphasis
4. Weigh the importance of each aspect
5. Provide your final sentiment assessment

Analysis:
```

**Variants:**
- Zero-shot CoT: "Let's think step by step"
- Few-shot CoT: Include examples with reasoning
- Auto-CoT: Automatically generate reasoning chains

---

## 3. ReAct (Reasoning + Acting)

**Description:** Interleaves reasoning with tool use actions.

**When to Use:**
- Tasks requiring external data
- Multi-step workflows
- Dynamic decision making
- Tool-augmented analysis

**Implementation:**
```python
class ReActStrategy(BaseStrategy):
    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}

    async def execute(self, task: str, **kwargs) -> ReActResult:
        history = []
        for i in range(kwargs.get("max_iterations", 10)):
            # Generate thought + action
            response = await self._generate_step(task, history)

            if response.action == "finish":
                return ReActResult(
                    answer=response.answer,
                    reasoning=history
                )

            # Execute action
            observation = await self._execute_action(
                response.action,
                response.action_input
            )
            history.append({
                "thought": response.thought,
                "action": response.action,
                "action_input": response.action_input,
                "observation": observation
            })

        return ReActResult(answer=None, reasoning=history, timeout=True)
```

**Prompt Template:**
```
You are a sentiment analysis agent with access to the following tools:

Tools:
- scrape_reviews(url): Scrape reviews from a URL
- analyze_sentiment(text): Analyze sentiment of text
- search_web(query): Search the web for information
- summarize(texts): Summarize multiple texts

Task: {task}

Use the following format:
Thought: Consider what to do next
Action: tool_name
Action Input: input for the tool
Observation: result from the tool
... (repeat as needed)
Thought: I now have enough information
Action: finish
Action Input: final answer

Begin!

Thought:
```

**ReAct Loop:**
```
Thought → Action → Observation → Thought → Action → Observation → ... → Finish
```

---

## 4. Tree of Thoughts (ToT)

**Description:** Explores multiple reasoning paths in a tree structure.

**When to Use:**
- Complex problems with multiple solutions
- When exploration is valuable
- Strategic decision making
- Comprehensive analysis

**Implementation:**
```python
class TreeOfThoughtsStrategy(BaseStrategy):
    async def execute(self, problem: str, **kwargs) -> ToTResult:
        branches = kwargs.get("branches", 3)
        depth = kwargs.get("depth", 3)

        root = ThoughtNode(content=problem)
        await self._expand_tree(root, branches, depth)

        # Evaluate all leaf nodes
        best_path = await self._find_best_path(root)
        return ToTResult(
            answer=best_path[-1].content,
            tree=root,
            path=best_path
        )

    async def _expand_tree(self, node: ThoughtNode, branches: int, depth: int):
        if depth == 0:
            return

        # Generate multiple thoughts
        thoughts = await self._generate_thoughts(node.content, branches)

        for thought in thoughts:
            child = ThoughtNode(content=thought, parent=node)
            node.children.append(child)
            # Evaluate thought quality
            child.score = await self._evaluate_thought(thought)
            # Recursively expand promising branches
            if child.score > self.threshold:
                await self._expand_tree(child, branches, depth - 1)
```

**Prompt Template (Generation):**
```
Given the following problem and current reasoning state, generate {n} distinct next steps:

Problem: {problem}
Current state: {current_thought}

Generate {n} different approaches to continue:
1.
2.
3.
```

**Prompt Template (Evaluation):**
```
Evaluate the following reasoning step for the given problem.
Score from 1-10 based on: relevance, progress toward solution, logical soundness.

Problem: {problem}
Reasoning step: {thought}

Score (1-10):
Justification:
```

---

## 5. Self-Consistency

**Description:** Generate multiple responses and aggregate for robustness.

**When to Use:**
- Critical decisions
- Reducing hallucinations
- Improving reliability
- When accuracy > speed

**Implementation:**
```python
class SelfConsistencyStrategy(BaseStrategy):
    async def execute(self, prompt: str, **kwargs) -> ConsistencyResult:
        samples = kwargs.get("samples", 5)
        temperature = kwargs.get("temperature", 0.8)

        # Generate multiple responses
        responses = await asyncio.gather(*[
            self.llm.generate(prompt, temperature=temperature)
            for _ in range(samples)
        ])

        # Parse and aggregate
        parsed = [self._parse_response(r) for r in responses]
        aggregated = self._aggregate(parsed, kwargs.get("method", "majority"))

        return ConsistencyResult(
            answer=aggregated,
            responses=parsed,
            confidence=self._calculate_confidence(parsed, aggregated)
        )

    def _aggregate(self, responses: List, method: str):
        if method == "majority":
            return Counter(responses).most_common(1)[0][0]
        elif method == "weighted":
            # Weight by confidence scores
            pass
```

**Aggregation Methods:**
- Majority vote
- Weighted average
- Confidence-based
- Unanimous agreement

---

## 6. Reflection/Self-Critique

**Description:** Model critiques and improves its own output.

**When to Use:**
- Quality-sensitive tasks
- Error detection
- Iterative refinement
- Complex reasoning validation

**Implementation:**
```python
class ReflectionStrategy(BaseStrategy):
    async def execute(self, prompt: str, **kwargs) -> ReflectionResult:
        max_iterations = kwargs.get("max_iterations", 3)

        # Initial response
        response = await self.llm.generate(prompt)

        for i in range(max_iterations):
            # Critique
            critique = await self._critique(prompt, response)

            if critique.is_satisfactory:
                break

            # Refine based on critique
            response = await self._refine(prompt, response, critique)

        return ReflectionResult(
            final_answer=response,
            iterations=i + 1,
            critiques=critiques
        )
```

**Critique Prompt:**
```
Review the following response for accuracy, completeness, and reasoning quality.

Original task: {task}
Response: {response}

Critique the response:
1. Are there any factual errors?
2. Is the reasoning sound?
3. Is anything missing?
4. How could it be improved?

Critique:
```

---

## 7. Few-Shot Prompting

**Description:** Provide examples to guide model behavior.

**When to Use:**
- Consistent output format needed
- Domain-specific tasks
- Teaching new patterns
- Improving accuracy

**Implementation:**
```python
class FewShotStrategy(BaseStrategy):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    async def execute(self, prompt: str, **kwargs) -> str:
        few_shot_prompt = self._build_few_shot_prompt(prompt)
        return await self.llm.generate(few_shot_prompt)

    def _build_few_shot_prompt(self, prompt: str) -> str:
        examples_text = "\n\n".join([
            f"Input: {ex.input}\nOutput: {ex.output}"
            for ex in self.examples
        ])
        return f"""Here are some examples:

{examples_text}

Now, for the following input:
Input: {prompt}
Output:"""
```

**Example Template:**
```
Analyze the sentiment of product reviews.

Example 1:
Review: "This phone has an amazing camera but the battery life is terrible."
Analysis: {"overall": "mixed", "aspects": {"camera": "positive", "battery": "negative"}}

Example 2:
Review: "Best purchase I've ever made! Everything works perfectly."
Analysis: {"overall": "positive", "aspects": {"general": "positive"}}

Now analyze:
Review: {text}
Analysis:
```

---

## 8. Structured Output Prompting

**Description:** Force output into specific formats.

**Implementation:**
```python
class StructuredOutputStrategy(BaseStrategy):
    async def execute(self, prompt: str, schema: dict, **kwargs) -> dict:
        structured_prompt = f"""
{prompt}

Respond with valid JSON matching this schema:
{json.dumps(schema, indent=2)}

JSON Response:
"""
        response = await self.llm.generate(
            structured_prompt,
            response_format={"type": "json_object"}  # If supported
        )
        return json.loads(response)
```

**With Pydantic Validation:**
```python
from pydantic import BaseModel

class SentimentOutput(BaseModel):
    label: str
    confidence: float
    aspects: List[AspectSentiment]
    reasoning: str

result = await strategy.execute(
    prompt,
    output_model=SentimentOutput
)
```

---

## Strategy Selection Matrix

| Task Complexity | Time Budget | Accuracy Need | Recommended |
|-----------------|-------------|---------------|-------------|
| Low | Low | Medium | Direct |
| Low | Medium | High | Few-Shot |
| Medium | Medium | High | CoT |
| Medium | High | Very High | CoT + Self-Consistency |
| High | High | Very High | ReAct |
| Very High | Very High | Critical | ToT + Reflection |

---

## Configuration

```yaml
prompting:
  default_strategy: "chain_of_thought"

  strategies:
    direct:
      enabled: true
      temperature: 0.3

    chain_of_thought:
      enabled: true
      temperature: 0.5
      include_examples: true

    react:
      enabled: true
      max_iterations: 10
      available_tools:
        - scraper
        - sentiment
        - summarizer

    tree_of_thoughts:
      enabled: true
      branches: 3
      depth: 3
      pruning_threshold: 0.6

    self_consistency:
      enabled: true
      samples: 5
      aggregation: "majority_vote"

    reflection:
      enabled: true
      max_iterations: 3
      quality_threshold: 0.8
```
