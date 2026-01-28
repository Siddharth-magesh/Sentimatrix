# Sentimatrix V2 - Agentic Patterns

## Overview

Agentic patterns enable autonomous, multi-step task execution with dynamic decision-making. Sentimatrix V2 implements several agentic architectures for complex sentiment analysis workflows.

---

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         AGENT CONTROLLER                             │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │ Planning │ │ Execution│ │ Memory   │ │ Learning │               │   │
│  │  │ Module   │ │ Engine   │ │ Manager  │ │ Module   │               │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           TOOL LAYER                                 │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            │   │
│  │  │Scraper │ │Sentiment│ │Summarize│ │Compare │ │ Search │            │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         LLM PROVIDERS                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Single Agent Pattern

**Description:** One agent with access to multiple tools.

**Use Case:** Standard sentiment analysis workflows

**Implementation:**
```python
class SentimentAgent:
    def __init__(self, llm: BaseLLMProvider, tools: List[Tool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.memory = ConversationMemory()

    async def run(self, task: str) -> AgentResult:
        self.memory.add("user", task)

        while True:
            # Think and decide action
            response = await self._think()

            if response.is_final:
                return AgentResult(
                    answer=response.content,
                    steps=self.memory.get_steps()
                )

            # Execute tool
            observation = await self._execute_tool(
                response.tool,
                response.tool_input
            )
            self.memory.add("observation", observation)

    async def _think(self) -> AgentResponse:
        prompt = self._build_prompt()
        response = await self.llm.generate(prompt)
        return self._parse_response(response)
```

**Tool Definition:**
```python
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable

    async def execute(self, **kwargs) -> str:
        return await self.func(**kwargs)

# Example tools
scraper_tool = Tool(
    name="scrape_reviews",
    description="Scrape product reviews from a URL",
    parameters={
        "url": {"type": "string", "description": "URL to scrape"},
        "limit": {"type": "integer", "description": "Max reviews"}
    },
    func=scrape_reviews
)

sentiment_tool = Tool(
    name="analyze_sentiment",
    description="Analyze sentiment of text",
    parameters={
        "text": {"type": "string", "description": "Text to analyze"}
    },
    func=analyze_sentiment
)
```

---

## 2. Multi-Agent Pattern

**Description:** Multiple specialized agents collaborate on complex tasks.

**Use Case:** Comprehensive product analysis, competitive analysis

**Architecture:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Scraper   │     │  Analyzer   │     │  Reporter   │
│    Agent    │────▶│    Agent    │────▶│    Agent    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   Raw Data          Insights            Report
```

**Implementation:**
```python
class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {
            "scraper": ScraperAgent(),
            "analyzer": AnalyzerAgent(),
            "reporter": ReporterAgent()
        }
        self.message_bus = MessageBus()

    async def run(self, task: str) -> dict:
        # Plan execution
        plan = await self._create_plan(task)

        # Execute agents in order
        context = {}
        for step in plan.steps:
            agent = self.agents[step.agent]
            result = await agent.run(
                task=step.task,
                context=context
            )
            context[step.output_key] = result

        return context

class ScraperAgent:
    """Specialized agent for data collection"""
    tools = [scraper_tool, search_tool]

class AnalyzerAgent:
    """Specialized agent for sentiment analysis"""
    tools = [sentiment_tool, emotion_tool, aspect_tool]

class ReporterAgent:
    """Specialized agent for report generation"""
    tools = [summarize_tool, visualize_tool, format_tool]
```

---

## 3. Hierarchical Agent Pattern

**Description:** Manager agent delegates to worker agents.

**Use Case:** Large-scale analysis with many sub-tasks

**Architecture:**
```
                    ┌─────────────┐
                    │   Manager   │
                    │    Agent    │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Worker 1 │     │ Worker 2 │     │ Worker 3 │
    │ (Amazon) │     │ (Steam)  │     │ (Reddit) │
    └──────────┘     └──────────┘     └──────────┘
```

**Implementation:**
```python
class ManagerAgent:
    def __init__(self, workers: List[WorkerAgent]):
        self.workers = workers
        self.llm = get_provider("gpt-4o")

    async def run(self, task: str) -> dict:
        # Decompose task
        subtasks = await self._decompose(task)

        # Delegate to workers
        results = await asyncio.gather(*[
            self._delegate(subtask)
            for subtask in subtasks
        ])

        # Synthesize results
        return await self._synthesize(results)

    async def _decompose(self, task: str) -> List[SubTask]:
        prompt = f"""
Decompose this task into subtasks for specialized workers:
Task: {task}

Available workers:
{self._describe_workers()}

Return subtasks as JSON:
"""
        response = await self.llm.generate(prompt)
        return self._parse_subtasks(response)

    async def _delegate(self, subtask: SubTask) -> dict:
        worker = self._select_worker(subtask)
        return await worker.run(subtask.description)
```

---

## 4. Reflexion Pattern

**Description:** Agent reflects on failures and improves.

**Use Case:** Handling edge cases, improving reliability

**Implementation:**
```python
class ReflexionAgent:
    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.memory = EpisodicMemory()
        self.max_attempts = 3

    async def run(self, task: str) -> AgentResult:
        for attempt in range(self.max_attempts):
            # Try to complete task
            result = await self._attempt(task)

            # Evaluate success
            evaluation = await self._evaluate(task, result)

            if evaluation.success:
                return result

            # Reflect on failure
            reflection = await self._reflect(task, result, evaluation)
            self.memory.add_reflection(reflection)

        return AgentResult(
            success=False,
            error="Max attempts exceeded",
            reflections=self.memory.get_reflections()
        )

    async def _reflect(self, task: str, result: Any, evaluation: Evaluation) -> str:
        prompt = f"""
Task: {task}
Attempt result: {result}
Evaluation: {evaluation.feedback}

What went wrong? How can the next attempt be improved?

Reflection:
"""
        return await self.llm.generate(prompt)
```

---

## 5. Plan-and-Execute Pattern

**Description:** Create a plan first, then execute steps.

**Use Case:** Complex multi-step workflows

**Implementation:**
```python
class PlanAndExecuteAgent:
    def __init__(self, planner_llm: BaseLLMProvider, executor_llm: BaseLLMProvider):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = {}

    async def run(self, task: str) -> AgentResult:
        # Planning phase
        plan = await self._plan(task)

        # Execution phase
        results = []
        context = {}

        for step in plan.steps:
            result = await self._execute_step(step, context)
            results.append(result)
            context[step.id] = result

            # Re-plan if needed
            if result.requires_replan:
                remaining = await self._replan(task, plan, results)
                plan.steps = plan.steps[:step.index + 1] + remaining

        return AgentResult(
            answer=results[-1].output,
            plan=plan,
            execution_trace=results
        )

    async def _plan(self, task: str) -> Plan:
        prompt = f"""
Create a step-by-step plan to accomplish this task:
Task: {task}

Available tools:
{self._describe_tools()}

Plan (as numbered steps):
"""
        response = await self.planner.generate(prompt)
        return self._parse_plan(response)
```

---

## 6. Autonomous Research Agent

**Description:** Agent that autonomously researches and analyzes topics.

**Use Case:** Comprehensive market research, competitive analysis

**Implementation:**
```python
class ResearchAgent:
    def __init__(self):
        self.llm = get_provider("gpt-4o")
        self.tools = [
            WebSearchTool(),
            ScraperTool(),
            SentimentTool(),
            SummarizerTool()
        ]
        self.knowledge_base = KnowledgeBase()

    async def research(self, topic: str, depth: int = 3) -> ResearchReport:
        # Generate research questions
        questions = await self._generate_questions(topic)

        # Research each question
        findings = []
        for question in questions:
            finding = await self._investigate(question, depth)
            findings.append(finding)
            self.knowledge_base.add(finding)

        # Synthesize into report
        report = await self._synthesize_report(topic, findings)
        return report

    async def _investigate(self, question: str, depth: int) -> Finding:
        # Search for information
        search_results = await self.tools["search"].execute(question)

        # Deep dive on relevant sources
        detailed_info = []
        for result in search_results[:depth]:
            content = await self.tools["scraper"].execute(result.url)
            analysis = await self.tools["sentiment"].execute(content)
            detailed_info.append({
                "source": result.url,
                "content": content,
                "analysis": analysis
            })

        return Finding(
            question=question,
            sources=detailed_info,
            summary=await self._summarize(detailed_info)
        )
```

---

## 7. Tool Selection Strategies

### Dynamic Tool Selection
```python
class DynamicToolSelector:
    async def select_tool(self, task: str, available_tools: List[Tool]) -> Tool:
        prompt = f"""
Given this task: {task}

Which tool is most appropriate?

Available tools:
{self._format_tools(available_tools)}

Respond with the tool name and brief justification.
"""
        response = await self.llm.generate(prompt)
        return self._parse_selection(response, available_tools)
```

### Tool Chaining
```python
class ToolChain:
    def __init__(self, tools: List[Tool]):
        self.tools = tools

    async def execute(self, initial_input: Any) -> Any:
        current_output = initial_input
        for tool in self.tools:
            current_output = await tool.execute(current_output)
        return current_output

# Example chain: URL → Scrape → Analyze → Summarize
analysis_chain = ToolChain([
    scraper_tool,
    sentiment_tool,
    summarizer_tool
])
```

---

## 8. Memory Systems

### Short-Term Memory (Conversation)
```python
class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.messages = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2:]
```

### Long-Term Memory (Vector Store)
```python
class VectorMemory:
    def __init__(self, embedding_model: str):
        self.embeddings = []
        self.documents = []
        self.model = get_embedding_model(embedding_model)

    async def add(self, document: str, metadata: dict = None):
        embedding = await self.model.embed(document)
        self.embeddings.append(embedding)
        self.documents.append({"content": document, "metadata": metadata})

    async def search(self, query: str, k: int = 5) -> List[dict]:
        query_embedding = await self.model.embed(query)
        similarities = cosine_similarity(query_embedding, self.embeddings)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]
```

### Episodic Memory (Experience)
```python
class EpisodicMemory:
    def __init__(self):
        self.episodes = []

    def add_episode(self, task: str, actions: List, outcome: str, success: bool):
        self.episodes.append({
            "task": task,
            "actions": actions,
            "outcome": outcome,
            "success": success,
            "timestamp": datetime.now()
        })

    def recall_similar(self, task: str, k: int = 3) -> List[dict]:
        # Find similar past experiences
        pass
```

---

## Configuration

```yaml
agents:
  default_type: "single"

  single:
    max_iterations: 15
    tools:
      - scraper
      - sentiment
      - summarizer
      - search

  multi:
    enabled: true
    agents:
      - name: scraper
        role: "Data collection specialist"
      - name: analyzer
        role: "Sentiment analysis specialist"
      - name: reporter
        role: "Report generation specialist"

  hierarchical:
    enabled: true
    max_workers: 5
    parallel: true

  memory:
    conversation:
      max_turns: 20
    vector:
      enabled: true
      model: "all-MiniLM-L6-v2"
      max_documents: 1000
    episodic:
      enabled: true
      max_episodes: 100
```

---

## Best Practices

1. **Start simple** - Use single agent before multi-agent
2. **Define clear tools** - Well-documented tool descriptions improve selection
3. **Limit iterations** - Set max iterations to prevent infinite loops
4. **Use appropriate memory** - Match memory type to task requirements
5. **Monitor costs** - Agent loops can consume many tokens
6. **Implement timeouts** - Prevent runaway agents
7. **Log everything** - Debugging agents requires detailed logs
