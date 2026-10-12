# Research: Robust Plan-and-Execute Replanning

## Decision 1: Keep decision semantics strict

- **Decision**: `continue` keeps the current remaining steps and has no replacement steps or final answer; `adjust` requires a non-empty replacement list and no final answer; `finish` requires a final answer and no replacement steps.
- **Rationale**: The existing contract explicitly distinguishes these outcomes. Using a mixed response by guessing which fields are authoritative could discard intent or skip requested work.
- **Alternatives considered**: Silently ignore fields that conflict with `decision`; rejected because it masks inconsistent model output. Change the meaning of `continue` to allow replacements or an answer; rejected because it breaks the current graph contract.

## Decision 2: Use a single bounded correction attempt for semantic conflicts

- **Decision**: When structured output parses but violates the decision-specific contract, make at most one additional replanner call with corrective instructions and the same original request, remaining plan, and completed results. If it is still invalid, surface a validation error.
- **Rationale**: The current failure happens after the executor has already run a tool. A bounded retry addresses transient model-format inconsistency without repeating the side-effecting executor step, while persistent invalid output remains visible.
- **Alternatives considered**: Retry indefinitely; rejected because of unbounded cost and loops. Retry provider/network errors; rejected because transport failures are unrelated to the semantic invariant and should remain explicit. Rerun the entire strategy; rejected because completed actions may persist.

## Decision 3: Reuse injected model and store; add no dependencies

- **Decision**: Implement recovery in the existing Plan-and-Execute graph using its injected chat model, Zod validation, and in-memory test store.
- **Rationale**: The current architecture already centralizes these behaviors, making deterministic offline tests possible without credentials or network access.
- **Alternatives considered**: Add a provider-specific retry or a new orchestration dependency; rejected as unnecessary and less portable.
