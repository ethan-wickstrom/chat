# Reasoning graph memory

This prototype records observable agent execution state in Neo4j so future requests can retrieve successful precedents and humans can inspect prior execution paths.

It follows the useful part of Neo4j Agent Memory's reasoning model while staying local to this app:

```text
(Conversation)<-[:INITIATED_BY]-(ReasoningTrace)
  -[:HAS_STEP]->(ReasoningStep)-[:NEXT_STEP]->(ReasoningStep)
  (ReasoningStep)-[:USES_TOOL]->(ToolCall)-[:INSTANCE_OF]->(Tool)
```

`ReasoningTrace` stores the task, model, task embedding, outcome, success/failure state, and timestamps. `ReasoningStep` stores an observable action summary, finish reason, token counts, and visible assistant text. `ToolCall` stores tool input/output snapshots and status. `Tool` gives calls a shared identity so usage can be aggregated across traces.

The app deliberately does not persist provider hidden reasoning or chain-of-thought. The useful durable state is the inspectable execution graph: what task arrived, which action/tool was chosen, what the tool returned, what happened next, and whether the run worked.

## Request loop

For each chat request:

1. Embed the current task with AI Gateway.
2. Retrieve similar successful `ReasoningTrace` nodes from Neo4j's vector index.
3. Add their action/tool paths to the system context as untrusted historical data.
4. Start a new trace linked to the chat's `Conversation` node.
5. Record each AI SDK `onStepFinish` event as a `ReasoningStep`; create `ToolCall` and `Tool` nodes when tools run.
6. Link sequential steps with `NEXT_STEP`.
7. Mark the trace successful or failed when the stream finishes/errors.

If Neo4j or embeddings are unavailable, chat generation continues without memory. Memory writes are bounded and failures do not fail the agent request.

## Configuration

Set:

```bash
NEO4J_QUERY_URL=https://<dbid>.databases.neo4j.io/db/neo4j/query/v2
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_REASONING_EMBEDDING_MODEL=openai/text-embedding-3-small
```

`AI_GATEWAY_API_KEY` is already used by the app and is also used for task embeddings.

## Inspection API

`GET /api/memory/reasoning` lists the signed-in user's traces. Optional query parameters:

```text
?success=true
?success=false
?limit=50
```

`GET /api/memory/reasoning/:id` returns one trace with its ordered steps and tool calls.

## Useful Cypher

Which tools were used on successful runs:

```cypher
MATCH (trace:ReasoningTrace {success: true})-[:HAS_STEP]->(:ReasoningStep)
  -[:USES_TOOL]->(:ToolCall)-[:INSTANCE_OF]->(tool:Tool)
RETURN tool.name, count(*) AS calls
ORDER BY calls DESC
```

Which paths succeeded:

```cypher
MATCH (trace:ReasoningTrace {success: true})-[:HAS_STEP]->(first:ReasoningStep)
WHERE NOT (:ReasoningStep)-[:NEXT_STEP]->(first)
MATCH path = (first)-[:NEXT_STEP*0..]->(last:ReasoningStep)
WHERE NOT (last)-[:NEXT_STEP]->()
RETURN trace.task, [step IN nodes(path) | step.action] AS actions, trace.outcome
ORDER BY trace.completedAt DESC
```

What failed previously:

```cypher
MATCH (trace:ReasoningTrace {success: false})
OPTIONAL MATCH (trace)-[:HAS_STEP]->(step:ReasoningStep)
RETURN trace.task,
       trace.errorKind,
       trace.outcome,
       collect(step.summary) AS path
ORDER BY trace.completedAt DESC
```

Which tool sequence preceded failure:

```cypher
MATCH (trace:ReasoningTrace {success: false})-[:HAS_STEP]->(step:ReasoningStep)
OPTIONAL MATCH (step)-[:USES_TOOL]->(:ToolCall)-[:INSTANCE_OF]->(tool:Tool)
WITH trace, step, collect(tool.name) AS tools
ORDER BY step.index
RETURN trace.task, collect({step: step.index, action: step.action, tools: tools}) AS path
```

## Next cuts

The next high-value additions are step-level embeddings for case-based retrieval, explicit `TOUCHED` edges from steps to domain entities, richer tool failure status/duration instrumentation, and evaluation queries that compare success rate by path/tool combination.
