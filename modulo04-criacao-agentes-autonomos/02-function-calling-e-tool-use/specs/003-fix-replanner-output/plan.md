# Implementation Plan: Robust Plan-and-Execute Replanning

**Branch**: `003-fix-replanner-output` | **Date**: 2026-10-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-fix-replanner-output/spec.md`

## Summary

Make replanner responses unambiguous and recover once from a contradictory structured decision. The replanner will be prompted with the exact fields allowed for each decision. If the response conflicts with the selected decision, it will receive one bounded correction attempt; a second contradictory response remains an explicit validation failure. This preserves decision semantics without silently discarding model output or rerunning completed tool actions.

## Technical Context

**Language/Version**: TypeScript ESM, strict mode, Node.js 22 LTS

**Primary Dependencies**: Existing `@langchain/core`, `@langchain/langgraph`, `@langchain/openai`, and Zod; no new dependencies

**Storage**: Existing injected `OpsStore`; production default is JSON-backed

**Testing**: `node:test` with `tsx`; targeted Plan-and-Execute tests, followed by `npm run typecheck` and `npm test`

**Target Platform**: Node.js CLI

**Project Type**: TypeScript application with CLI and service layers

**Performance Goals**: Valid replanner responses incur no additional model request; a contradictory response incurs at most one additional replanner request.

**Constraints**: Preserve decision invariants, completed step/result pairs, side-effect ordering, eight-step total limit, and surfaced errors. Never repeat a tool execution as part of replanner recovery.

**Scale/Scope**: One planner/replanner strategy and its co-located tests and contract documentation.

## Constitution Check

- TypeScript ESM strict typing: pass; use existing types and schemas.
- Layering and existing architecture: pass; keep the behavior inside the existing Plan-and-Execute agent boundary.
- Zod contracts and explicit errors: pass; validate responses and retain a domain validation error if bounded correction does not produce a valid decision.
- Tests and quality gates: pass; update tests and run targeted tests, typecheck, and full test suite.
- Safe/simple integration: pass; use existing model injection, no packages, secrets, or provider-specific assumptions.

## Project Structure

### Documentation (this feature)

```text
specs/003-fix-replanner-output/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── replanner.md
└── tasks.md
```

### Source Code

```text
src/agents/
├── plan-and-execute.ts
└── plan-and-execute.test.ts

specs/001-nucleo-raciocinio-opspilot/contracts/
└── plan-and-execute.md
```

**Structure Decision**: Keep the fix in the current Plan-and-Execute agent and its unit tests. Update the existing strategy contract to describe the bounded recovery behavior and add a feature-specific contract and offline validation guide.

## Design

1. Keep the existing `continue`, `adjust`, and `finish` semantics and validations. Strengthen the replanner instructions with exact response shapes and explicit rules excluding fields belonging to other decisions.
2. Validate the decision after structured output parsing. If a decision is semantically contradictory, make exactly one additional replanner call with the original request, remaining plan, completed results, and a concise correction instruction. Do not execute tools during recovery.
3. If the corrected response is still contradictory, surface a `ValidationError`; do not choose a decision based on incidental fields or discard conflicting content silently.
4. Test the one-retry recovery, repeated-invalid failure, valid continue/adjust/finish paths, step ordering, and preservation of completed results with injected models and an in-memory store.

## Error Handling

- Provider/network errors are not retried by this change and continue to propagate.
- Only a parsed response that violates the replanner's decision-specific semantic contract triggers the single correction attempt.
- A second semantic contract violation surfaces as a domain validation error with no success-shaped fallback.

## Risks

- Some providers may return the same conflicting fields on the correction attempt. The retry is bounded, and the second failure remains explicit.
- The correction call adds latency and model cost only for invalid responses.
- The first tool action may already have produced a persistent effect before replanning fails. Recovery must not re-run that action; callers should inspect current incidents before manually replaying a failed multi-action request.

## Constitution Re-check

All gates remain satisfied after design. No new package, environment variable, external API, storage entity, or layer is introduced.
