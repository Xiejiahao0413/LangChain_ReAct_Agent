# Agent Evaluation

This folder contains a lightweight evaluation set and runner for the
LangChain ReAct Agent.

## What It Measures

- Tool call accuracy: expected tools were used and forbidden tools were not used.
- Expected tool recall: all expected tools appeared in the agent trace.
- Exact tool match rate: the used tool set exactly equals the expected tool set.
- Task completion rate: the answer is non-empty and hits at least one completion keyword.
- Knowledge reference rate: knowledge-base questions used `rag_summarize` and explicitly referenced retrieved material.

## Dataset

`agent_eval_samples.jsonl` contains 40 samples across:

- knowledge-base QA
- weather/tool calls
- report generation
- mixed RAG + user-data tasks
- general chat
- boundary cases

Each sample defines:

```json
{
  "id": "kb_001",
  "category": "knowledge_qa",
  "query": "扫地机器人为什么会漏扫，应该怎么排查？",
  "expected_tools": ["rag_summarize"],
  "forbidden_tools": ["fetch_external_data"],
  "requires_knowledge_reference": true,
  "completion_keywords": ["漏扫", "排查"]
}
```

## Validate The Dataset

```powershell
.\venvreactagent\Scripts\python.exe -m evaluation.agent_evaluator --dry-run
```

## Run A Live Evaluation

Live evaluation calls the LLM and embedding service, so it requires
`DASHSCOPE_API_KEY` and working network access.

```powershell
$env:DASHSCOPE_API_KEY="your-key"
.\venvreactagent\Scripts\python.exe -m evaluation.agent_evaluator --limit 5
```

Full run:

```powershell
.\venvreactagent\Scripts\python.exe -m evaluation.agent_evaluator
```

Reports are written to `evaluation/results/`.

## Evaluate Existing Predictions

If you already have a saved report or prediction JSON:

```powershell
.\venvreactagent\Scripts\python.exe -m evaluation.agent_evaluator --predictions evaluation/results/agent_eval_xxx.json
```
