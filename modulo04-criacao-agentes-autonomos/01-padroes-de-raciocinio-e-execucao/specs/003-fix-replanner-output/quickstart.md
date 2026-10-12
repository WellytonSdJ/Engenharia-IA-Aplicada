# Quickstart: Validate Robust Replanning

## Prerequisites

- Node.js 22 LTS and project dependencies installed.
- No OpenRouter credentials are needed for these tests; they use an injected deterministic model and in-memory store.

## Focused Validation

From the project directory, run:

```powershell
node --import tsx --test src/agents/plan-and-execute.test.ts
npm run typecheck
npm test
```

Expected outcomes:

- A contradictory first replanner response followed by a valid correction completes without re-executing the prior tool step.
- A second contradictory response returns a validation error.
- `continue`, `adjust`, and `finish` preserve their documented behavior, including action order and the eight-step limit.

See [the response contract](./contracts/replanner.md) for valid decision shapes.
