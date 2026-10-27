# Data Model: Robust Plan-and-Execute Replanning

This change does not add persisted entities or alter graph state. It clarifies how existing Plan-and-Execute state is preserved when replanning is corrected.

## Replanner Decision

- `decision`: `continue`, `adjust`, or `finish`.
- `steps`: optional replacement step list; required and non-empty only for `adjust`, forbidden for `continue` and `finish`.
- `answer`: optional final response; required and non-empty only for `finish`, forbidden for `continue` and `adjust`.
- **Validation**: A parsed response must satisfy the decision-specific field combination. A semantic conflict gets at most one correction attempt; a second conflict is an error.

## Plan Execution State

- `input`: original operator request; unchanged through correction.
- `plan`: ordered steps, including completed prefix and remaining suffix; unchanged when continuing or correcting the replanner.
- `done`: ordered completed step/result pairs; preserved across the correction attempt and never replayed.
- `answer`: empty until the planner or replanner finishes the request.

## State Transition on Replanner Correction

1. Executor records its step and result in `done`.
2. Replanner response violates the selected decision's field contract.
3. A single new replanner response is requested with the same request, remaining plan, and completed results.
4. A valid correction is applied through the existing continue/adjust/finish transition; an invalid correction fails explicitly.
