import assert from "node:assert/strict";
import test from "node:test";
import { formatArenaRuns, parseArenaArgs, runArena } from "./controllers/arena-controller.js";
import type { ReasoningStrategy } from "./agents/strategy.js";

function strategy(name: string, answer: string, seenInputs: string[]): ReasoningStrategy {
  return {
    name,
    async run(input, options) {
      seenInputs.push(input);
      return {
        answer,
        trace: [{ type: "answer", content: answer }],
        metrics: { llmCalls: 1, latencyMs: 1, iterations: options?.maxIterations },
      };
    },
  };
}

test("arena validates flags before execution and shares input across selected strategies", async () => {
  const seenInputs: string[] = [];
  const strategies = {
    first: strategy("first", "Result one", seenInputs),
    second: strategy("second", "Result two", seenInputs),
  };
  assert.throws(() => parseArenaArgs(["--max-iterations", "0", "check"]));
  assert.throws(() => parseArenaArgs(["--unknown", "check"]));
  await assert.rejects(runArena(["--strategies", "missing", "check"], strategies));
  assert.deepEqual(seenInputs, []);

  const runs = await runArena([
    "--strategies=first,second",
    "--max-iterations=4",
    "check firing alerts",
  ], strategies);
  assert.deepEqual(seenInputs, ["check firing alerts", "check firing alerts"]);
  assert.deepEqual(runs.map((run) => run.strategy), ["first", "second"]);
  assert.match(formatArenaRuns(runs), /=== first ===[\s\S]*=== second ===/);
  assert.ok(runs.every((run) => run.metrics.llmCalls === 1));
});
