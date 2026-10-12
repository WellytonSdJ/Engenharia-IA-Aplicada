# Tasks: Robust Plan-and-Execute Replanning

**Input**: Design documents from `/specs/003-fix-replanner-output/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/replanner.md`, `quickstart.md`

**Organization**: Tasks are grouped by user story. This feature has one independently testable P1 story.

## Phase 1: Setup

**Purpose**: Project initialization and basic structure.

No setup tasks are required; the TypeScript project, scripts, model injection, and in-memory store already exist.

## Phase 2: Foundational

**Purpose**: Blocking prerequisites for user stories.

No foundational tasks are required; the feature uses existing contracts, graph state, and validation infrastructure.

## Phase 3: User Story 1 - Continue a planned execution (Priority: P1) 🎯 MVP

**Goal**: A contradictory replanner response receives one corrective attempt so that the remaining plan can complete without repeating already executed actions.

**Independent Test**: Use a deterministic model and in-memory store to run the three incident creations and first-incident resolution; provide a contradictory replanner response once, then a valid correction. Verify action order, one execution per requested action, retained results, and final answer. Also verify repeated contradictory responses still fail explicitly.

### Tests for User Story 1

- [X] T001 [US1] Add regression tests in `src/agents/plan-and-execute.test.ts` for a contradictory `continue` response followed by a valid correction, asserting the three incident creations and first resolution complete in order without repeating any tool call; add a second case where a second contradictory response remains an explicit validation failure.

### Implementation for User Story 1

- [X] T002 [US1] Update replanner instructions and decision validation in `src/agents/plan-and-execute.ts` to state the exact allowed response fields and make at most one corrective replanner call for semantic conflicts, preserving request, remaining steps, and completed results without rerunning tools or retrying provider/network errors.
- [X] T003 [P] [US1] Update `specs/001-nucleo-raciocinio-opspilot/contracts/plan-and-execute.md` to document the one bounded correction attempt, preserved completed work, and explicit failure after a second contradictory response.

**Checkpoint**: User Story 1 is independently functional when its focused tests pass and the documented recovery contract matches the implementation.

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: Validate the implementation against project-wide quality gates.

- [X] T004 Run `npm run typecheck` and `npm test` using the project's `package.json`; resolve any failures caused by this feature.

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No tasks; existing project setup is reused.
- **Foundational (Phase 2)**: No tasks; existing validation/model infrastructure is reused.
- **User Story 1 (Phase 3)**: Independent of other user stories.
- **Polish (Phase 4)**: Depends on T001, T002, and T003.

### User Story Dependencies

- **User Story 1 (P1)**: No dependencies on another story.

### Task Dependencies

- T001 must be written and shown to fail before T002 is implemented.
- T002 depends on T001.
- T003 is documentation-only and can proceed in parallel with T001 or T002.
- T004 follows T001, T002, and T003.

## Parallel Opportunities

- T003 can proceed in parallel with T001 or T002 because it changes only the contract document.

## Parallel Example: User Story 1

```text
Task: T001 Add regression tests in src/agents/plan-and-execute.test.ts
Task: T003 Update specs/001-nucleo-raciocinio-opspilot/contracts/plan-and-execute.md
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Add and run T001 regression tests; verify the mixed-output recovery case fails before implementation.
2. Implement T002 and run the focused Plan-and-Execute tests.
3. Complete T003 so the existing strategy contract matches the behavior.
4. Run T004 to validate type safety and the full test suite.

## Notes

- Every task uses the required checkbox, sequential task ID, optional parallel marker, and story label where applicable.
- Test fixtures must inject the model and store; do not read `.env` or make network calls.
- Recovery must not replay side-effecting tool calls. If a manually replayed prior run might have partially persisted actions, inspect the store before replaying it.
