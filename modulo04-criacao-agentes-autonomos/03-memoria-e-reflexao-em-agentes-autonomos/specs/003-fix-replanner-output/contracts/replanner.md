# Replanner Response Contract

The replanner must return exactly one coherent decision shape:

| Decision | Allowed response content | Behavior |
| --- | --- | --- |
| `continue` | `decision` only | Keep the remaining plan unchanged; at least one step must remain. |
| `adjust` | `decision` and a non-empty `steps` list | Replace only remaining steps, preserving completed steps and results. No `answer`. |
| `finish` | `decision` and a non-empty `answer` | End execution. No replacement `steps`. |

The total completed plus planned steps remains at most eight.

If a structured response parses but combines a decision with incompatible fields, the agent asks the replanner once to correct its response. The correction receives the same request, remaining steps, and completed tool results. No tool is re-executed for this correction. A second incompatible response fails explicitly with a validation error.

Provider/network errors are not retried under this contract.
