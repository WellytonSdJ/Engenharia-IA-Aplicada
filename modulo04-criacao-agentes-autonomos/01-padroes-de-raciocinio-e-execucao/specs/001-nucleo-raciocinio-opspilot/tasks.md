---
description: "Task list for the OpsPilot reasoning core"
---

# Tasks: Núcleo de raciocínio do OpsPilot

**Input**: Design documents from `specs/001-nucleo-raciocinio-opspilot/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Included because the specification requires deterministic tests without network access.

**Path Conventions**: Single TypeScript project with source and tests under `src/`, matching `npm test` (`src/**/*.test.ts`).

## Dependency Graph

```text
Setup -> Foundation -> US1 -> US2 -> US3 -> US4 -> Polish
                         -> US4 deterministic tests can begin after Foundation
```

- Foundation blocks all strategy and tool work.
- US1 establishes the shared result, trace and metrics contract.
- US2 depends on domain/store contracts and provides the tools consumed by US3.
- US3 depends on US1 and US2 to execute both strategies and the arena.
- US4 validates the deterministic store/trace slice and can be developed in parallel with US2 after Foundation.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare scripts and source layout without changing runtime behavior.

- [ ] T001 Add the `seed` script to `package.json` pointing to the seed entrypoint and preserve all existing `dev`, `arena`, `bench`, `test` and `typecheck` scripts.
- [ ] T002 [P] Create the planned source directories and empty module boundaries under `src/agents/`, `src/controllers/`, `src/models/` and `src/services/` without adding placeholder exports.
- [ ] T003 [P] Add the implementation test convention to `specs/001-nucleo-raciocinio-opspilot/quickstart.md` and keep commands aligned with `package.json`.

## Phase 2: Foundation (Blocking Prerequisites)

**Purpose**: Establish domain contracts, validation, errors, persistence boundaries and shared strategy primitives before user stories.

- [ ] T004 Define domain entities, enums and Zod schemas in `src/models/domain.ts`, including exactly the statuses `firing`/`resolved` for alerts and `open`/`resolved` for incidents.
- [ ] T005 Define typed domain errors and boundary-safe error categories in `src/services/errors.ts` for validation, missing resources, invalid transitions, configuration and iteration limits.
- [ ] T006 Define the injected store interface in `src/models/in-memory-store.ts`, including list alerts, create incident, resolve incident, seed/reset and read summary operations.
- [ ] T007 Implement `InMemoryOpsStore` in `src/models/in-memory-store.ts` with 5 services and exactly 6 seeded alerts, split into 3 `firing` and 3 `resolved`, with idempotent reset behavior.
- [ ] T008 Implement the Sequelize/MySQL persistence boundary in `src/models/sequelize-store.ts`, mapping services, alerts and incidents to Sequelize models without making deterministic tests depend on a live database.
- [ ] T009 Define `TraceEvent`, `ReasoningMetrics`, `ReasoningResult`, validation schemas and ordered trace formatting in `src/agents/trace.ts`; reject unknown event types and require `tool`/`args` on `action` events.
- [ ] T010 Define `ReasoningStrategy`, `ReasoningOptions` and the shared strategy result contract in `src/agents/strategy.ts`, including `maxIterations`, `llmCalls`, `latencyMs` and structured failure behavior.
- [ ] T011 Implement the single model factory in `src/agents/model.ts`, reading only `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`, configuring OpenRouter base URL and `temperature: 0`, and never logging credentials.
- [ ] T012 [P] Add deterministic trace schema/formatter tests in `src/agents/trace.test.ts` covering all six event types, action payload requirements, ordering and rejection of invalid events.

## Phase 3: User Story 1 - Executar uma estratégia de raciocínio observável (Priority: P1)

**Goal**: Provide the common observable strategy result and reusable execution accounting.

**Independent Test**: Invoke a strategy double with a deterministic input and verify `answer`, ordered valid `trace`, non-negative `llmCalls` and non-negative `latencyMs`.

- [ ] T013 [US1] Implement shared execution accounting in `src/agents/strategy.ts`, measuring monotonic latency, counting every model invocation and enforcing `maxIterations` before the next iteration.
- [ ] T014 [P] [US1] Add deterministic strategy contract tests in `src/agents/strategy.test.ts` covering successful results, limit exhaustion and structured failures without network access.
- [ ] T015 [US1] Update `specs/001-nucleo-raciocinio-opspilot/contracts/strategy.md` if implementation-level fields differ while preserving required `answer`, `trace`, `metrics`, `llmCalls` and `latencyMs`.

## Phase 4: User Story 2 - Investigar alertas e incidentes com ferramentas controladas (Priority: P1)

**Goal**: Expose validated alert and incident tools backed by the injected store and a reproducible seed.

**Independent Test**: Reset the in-memory store, verify 5 services and 6 alerts with a 3/3 status split, open an incident, resolve it, and verify invalid inputs do not mutate state.

- [ ] T016 [US2] Implement `list_alerts`, `open_incident` and `resolve_incident` as Zod-validated tools in `src/services/tools.ts`, injecting the store and translating domain errors at the tool boundary.
- [ ] T017 [P] [US2] Add deterministic store tests in `src/models/in-memory-store.test.ts` covering the exact seed counts, status filtering, incident creation, open-to-resolved transition, idempotent reset and invalid transition protection.
- [ ] T018 [P] [US2] Add deterministic tool contract tests in `src/services/tools.test.ts` covering input/output schemas, invalid status/severity/service/id and no partial mutation on failure.
- [ ] T019 [US2] Implement the explicit idempotent seed entrypoint in `src/services/seed.ts`, using the in-memory store by default and printing a summary of 5 services, 6 alerts, 3 `firing` and 3 `resolved`.
- [ ] T020 [US2] Wire the seed command in `package.json` and document its expected output in `specs/001-nucleo-raciocinio-opspilot/quickstart.md` without reading or printing `.env` values.

## Phase 5: User Story 3 - Comparar ReAct e Plan-and-Execute (Priority: P1)

**Goal**: Implement both strategies, their traces and the arena comparison flow over the same input.

**Independent Test**: Run both strategy adapters with a controlled model/store double and verify separate answers, traces, metrics and iteration limits for one input.

- [ ] T021 [US3] Implement the ReAct strategy in `src/agents/react.ts` using LangGraph `createReactAgent`, the model factory and injected tools; map graph messages/tool calls to the complete typed trace.
- [ ] T022 [US3] Implement the Plan-and-Execute graph in `src/agents/plan-and-execute.ts` with typed `planner`, `executor` and `replanner` nodes, one tool step per executor cycle, a maximum of 8 plan steps and conditional termination when no steps remain.
- [ ] T023 [US3] Add explicit iteration guards and LLM-call accounting to both strategy implementations in `src/agents/react.ts` and `src/agents/plan-and-execute.ts`, including a final `answer` or structured limit failure.
- [ ] T024 [P] [US3] Add controlled-model strategy tests in `src/agents/react.test.ts` covering tool actions, observations, final answer, complete trace and iteration accounting without network access.
- [ ] T025 [P] [US3] Add controlled-model graph tests in `src/agents/plan-and-execute.test.ts` covering planner output validation, one-step execution, replanning, zero remaining steps, 8-step cap and iteration limit.
- [ ] T026 [US3] Implement strategy selection and CLI argument validation in `src/controllers/arena-controller.ts`, supporting `--strategies` and positive `--max-iterations` without starting LLM calls for invalid input.
- [ ] T027 [US3] Implement the minimum arena entrypoint in `src/arena.ts`, running one or more selected strategies over the same input and printing separated answers, traces and metrics.
- [ ] T028 [P] [US3] Add arena CLI tests in `src/arena.test.ts` for strategy selection, shared input, output separation and invalid flags using model/store doubles.

## Phase 6: User Story 4 - Validar o núcleo sem rede (Priority: P2)

**Goal**: Make the deterministic validation slice explicit and repeatable for contributors.

**Independent Test**: Run the focused store and trace tests with provider credentials absent and confirm no network client is invoked.

- [ ] T029 [US4] Add a no-network test double/factory seam in `src/agents/model.test.ts`, proving deterministic tests do not construct a real OpenRouter request.
- [ ] T030 [US4] Add a deterministic integration check in `src/core-deterministic.test.ts` that resets the store, formats a known trace and compares stable counts/content across repeated runs.
- [ ] T031 [US4] Update `specs/001-nucleo-raciocinio-opspilot/quickstart.md` with the exact focused test command and the expected no-network behavior.

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T032 Run `npm run typecheck` and fix only feature-related TypeScript errors in the files listed by the plan.
- [ ] T033 Run `npm test` and fix only feature-related failures, preserving deterministic/no-network coverage.
- [ ] T034 Run `npm run seed` and verify the printed seed summary matches 5 services, 6 alerts, 3 `firing` and 3 `resolved`; do not print secrets.
- [ ] T035 Run the arena quickstart with configured provider variables, inspect trace formatting and metrics, and update `specs/001-nucleo-raciocinio-opspilot/quickstart.md` only for verified behavior.
- [ ] T036 Review the implementation against `specs/001-nucleo-raciocinio-opspilot/spec.md`, `plan.md`, `data-model.md` and `contracts/`, then mark no task complete until its validation command passes.

## Parallel Execution Examples

### After Foundation

```text
T012 [trace tests] || T017 [store tests] || T018 [tool tests]
```

### After User Story 2

```text
T024 [ReAct tests] || T025 [Plan-and-Execute tests] || T028 [arena tests]
```

### Before Polish

```text
T031 [quickstart docs] || T032 [typecheck preparation]
```

## Implementation Strategy

1. **MVP**: Complete Foundation, US1 and US2, then run deterministic store/trace/tool tests and `npm run typecheck`.
2. **Strategy comparison**: Complete US3 with controlled-model tests before requiring live OpenRouter credentials.
3. **Operational validation**: Complete US4 and Polish, execute the seed, then run the live arena only when provider variables are configured.

## Completion Criteria

- All tasks follow the required checklist format with sequential IDs and exact file paths.
- User stories can be tested independently according to their stated criteria.
- `npm run typecheck` and `npm test` pass before implementation is reported complete.
- The seed is executed only during implementation, after this task list is reviewed and approved.
