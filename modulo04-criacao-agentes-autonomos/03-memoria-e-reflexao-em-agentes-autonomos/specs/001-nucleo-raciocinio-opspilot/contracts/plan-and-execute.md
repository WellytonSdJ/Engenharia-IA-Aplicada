# Plan-and-Execute Contract

## Graph State

The graph carries:

- `input`: original operator request.
- `plan`: ordered tool steps, including the completed prefix and active suffix.
- `done`: ordered `[step, serialized result]` pairs appended by the executor.
- `answer`: empty until the planner or replanner completes the request.

## Nodes

- `planner` returns up to eight short steps, or no steps plus a direct answer.
- `executor` selects one available tool (`list_alerts`, `list_incidents`, `open_incident`, or `resolve_incident`) for the next plan step, executes it, and appends its step/result pair to `done`.
- `replanner` returns one explicit decision:
  - `continue`: retain the existing remaining plan, with no replacement steps or final answer.
  - `adjust`: replace the remaining plan with a non-empty ordered list, with no final answer.
  - `finish`: set a non-empty user-facing answer, with no replacement steps.

## Limits and Termination

The total of completed and active steps must remain at most eight:
`done.length + (plan.length - done.length) <= 8`.
Any adjusted plan is checked against the completed count before it is accepted. The configured iteration limit is checked before each tool execution. A finished graph must return a non-empty answer.

## Replanner Recovery

Each replanner response must match the fields allowed for its decision. If structured output parses but combines incompatible fields, the agent asks for one corrected response using the same request, remaining plan, and completed results. It does not execute a tool during this correction. If the corrected response is still contradictory, the graph fails with a validation error. Provider/network errors are not retried.

## Persistence

Tools use the injected `OpsStore`. The runtime default is the JSON-backed store at `src/data/ops-store.json`; tests use a fresh in-memory store or a temporary JSON path.
