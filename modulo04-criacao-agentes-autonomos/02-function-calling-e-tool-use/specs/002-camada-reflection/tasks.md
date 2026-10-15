# Tasks: Camada de Reflection

**Input**: Design documents from `/specs/002-camada-reflection/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/reflection.md`, `quickstart.md`

**Organization**: Tasks are grouped by user story to support independent verification.

## Phase 1: Setup

**Purpose**: Reuse the existing TypeScript project setup.

No setup tasks are required; dependencies, test runner, strategy contracts, and arena already exist.

## Phase 2: Foundational

**Purpose**: Blocking prerequisites for user stories.

No foundational tasks are required; the implementation uses the existing `ReasoningStrategy`, `ReasoningResult`, model factory, and trace event types.

## Phase 3: User Story 1 - Reforçar uma estratégia com reflection (Priority: P1) 🎯 MVP

**Goal**: Decorate any strategy so its answer is evaluated, and rejected answers are regenerated with actionable feedback.

**Independent Test**: Use a deterministic strategy and structured-output model to verify approval without regeneration, rejection followed by a contextualized retry and approval, the default limit, limit zero, and explicit validation of options.

### Tests for User Story 1

- [X] T001 [US1] Add deterministic tests in `src/agents/reflection.test.ts` for first-pass approval, rejection followed by regeneration with feedback, default two-regeneration cap, `maxReflections: 0`, invalid option values, and propagated critic/strategy errors; assert the same model instance is passed to each strategy execution and critic.

### Implementation for User Story 1

- [X] T002 [US1] Implement `withReflection(strategy, opts)` in `src/agents/reflection.ts` with a Zod-validated non-negative `maxReflections` defaulting to 2, the structured `{ aproved, feedback }` critic using the shared model, evaluation of the original request and observation-only trace contents, and regeneration input that preserves the original request and appends the previous actionable feedback.

**Checkpoint**: The decorated strategy can approve or regenerate any base strategy without changes to that base strategy.

## Phase 4: User Story 2 - Comparar variantes refletidas na arena (Priority: P1)

**Goal**: Select and run reflected ReAct and Plan-and-Execute variants independently in the arena.

**Independent Test**: Select `reflect:react` and `reflect:plan-and-execute` using deterministic strategies and verify each result retains its own strategy name, trace, and metrics while the unwrapped names remain selectable.

### Tests for User Story 2

- [X] T003 [US2] Add coverage in `src/arena.test.ts` for registry entries and selection of `reflect:react` and `reflect:plan-and-execute`, asserting both base identifiers remain present and arena results stay separate.

### Implementation for User Story 2

- [X] T004 [US2] Export an `arenaStrategies` registry from `src/arena.ts` containing the existing `react` and `plan-and-execute` strategies plus `withReflection` wrappers named `reflect:react` and `reflect:plan-and-execute`, and have `main` pass that registry to `runArena`.

**Checkpoint**: All four strategy identifiers can be selected, and each selected run is reported independently.

## Phase 5: User Story 3 - Inspecionar críticas e custo adicional (Priority: P2)

**Goal**: Make each critique and all additional model work auditable through the trace and execution metrics.

**Independent Test**: Run a fixed sequence of initial execution, critiques, and regenerations; verify one ordered critique event per evaluation, preserved base events, exact LLM-call totals, accumulated latency, and the last rejection at the cap.

### Tests for User Story 3

- [X] T005 [US3] Add trace and metric assertions in `src/agents/reflection.test.ts` for serialized `aproved`/`feedback` critique events after each attempt, preservation and ordering of attempt traces, exact base-plus-critic `llmCalls`, accumulated latency, original-request/observation-only critic inputs, and rejection recorded after the default cap.

### Implementation for User Story 3

- [X] T006 [US3] Complete trace and metric aggregation in `src/agents/reflection.ts` by appending each attempt trace followed by its `critique` event, summing all base-attempt calls and latencies plus one call and measured duration per critique, and returning the latest answer with an explicit final rejection when the regeneration limit is exhausted.

**Checkpoint**: A caller can inspect why a response was regenerated and account for all calls and latency.

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Keep published contracts and quality gates aligned with the implementation.

- [X] T007 [P] Update `specs/001-nucleo-raciocinio-opspilot/contracts/strategy.md`, `specs/001-nucleo-raciocinio-opspilot/contracts/arena-cli.md`, and `specs/002-camada-reflection/quickstart.md` to document the structured critic, evidence restriction, feedback retry, limits, reflected identifiers, trace events, metrics, and offline validation commands.
- [X] T008 Run the targeted reflection and arena tests, `npm run typecheck`, and `npm test`; resolve failures caused by this feature and mark completed tasks in `specs/002-camada-reflection/tasks.md`.

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No tasks; existing project setup is reused.
- **Foundational (Phase 2)**: No tasks; existing contracts and helpers are reused.
- **User Stories (Phases 3-5)**: Complete in priority order; User Story 2 depends on the reflection wrapper from User Story 1, and User Story 3 extends the wrapper behavior introduced in User Story 1.
- **Polish (Phase 6)**: Depends on all three user stories.

### User Story Dependencies

- **User Story 1 (P1)**: Independent; delivers the generic decorator MVP.
- **User Story 2 (P1)**: Depends on User Story 1's decorator to register the two reflected variants.
- **User Story 3 (P2)**: Depends on User Story 1's reflection cycle; verifies and completes its trace and metric reporting.

### Task Dependencies

- T001 must be written and shown to fail before T002.
- T002 must complete before T003/T004 can exercise/register reflected variants.
- T003 must be written before T004 and should fail until the arena registry exposes both variants.
- T005 must be written before T006 and should fail until trace and metric aggregation is complete.
- T007 can proceed independently of source changes because it updates documentation only.
- T008 follows all implementation and documentation tasks.

### Parallel Opportunities

- T007 can proceed in parallel with T001-T006 because it modifies only documentation.
- T001 and T003 touch separate test files, but arena integration depends on the decorator; execute T003 after T002 for a runnable test.

## Parallel Example: Documentation and Core Tests

```text
Task: T001 Add deterministic decorator tests in src/agents/reflection.test.ts
Task: T007 Update strategy/arena contracts and quickstart documentation
```

## Implementation Strategy

### MVP First (User Story 1)

1. Add deterministic tests for the reflection loop and verify they fail.
2. Implement the generic wrapper and pass User Story 1 tests.
3. Register both reflected variants and validate arena selection.
4. Verify audit trace and metrics behavior, then update contracts and quickstart.
5. Run focused tests, `npm run typecheck`, and `npm test`.
