# Prompt disclosure

Every prompt the system sends lives in this folder, one file per LangGraph node or tool. Filled in during Phase 3.

| File | LangGraph node / tool | Purpose |
|---|---|---|
| `system.md` | all nodes | Shared system prompt: grounding, citation, and safety rules |
| `router.md` | `router` | Classify intent, extract constraints, flag safety concerns |
| `planner.md` | `planner` | Choose rag.search / web.search / both; set fields and comparison criteria |
| `retriever.md` | `retriever` | Tool-call instructions and reconciliation rules (SKU, brand, title) |
| `answerer.md` | `answerer` | Top-3 recommendation with citations, plus ≤15 s spoken summary |
| `critic.md` | `critic` | Grounding and safety check; decide whether to re-retrieve |
| `guardrails.md` | `router`, `critic` | Unsafe requests, chemical mixing, unsupported claims |
| `few_shot.md` | `router`, `planner` | Few-shot examples, if used |
