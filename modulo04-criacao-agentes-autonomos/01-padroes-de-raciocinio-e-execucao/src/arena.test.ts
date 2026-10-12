import assert from "node:assert/strict";
import test from "node:test";
import { formatArenaRuns, parseArenaArgs, runArena } from "./controllers/arena-controller.js";
import type { ReasoningStrategy } from "./agents/strategy.js";
import { arenaStrategies } from "./arena.js";

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

test("arena registers reflected variants without removing base strategies", () => {
  assert.deepEqual(Object.keys(arenaStrategies), [
    "react",
    "plan-and-execute",
    "reflect:react",
    "reflect:plan-and-execute",
  ]);
  assert.equal(arenaStrategies.react.name, "react");
  assert.equal(arenaStrategies["plan-and-execute"].name, "plan-and-execute");
  assert.equal(arenaStrategies["reflect:react"].name, "reflect:react");
  assert.equal(arenaStrategies["reflect:plan-and-execute"].name, "reflect:plan-and-execute");
});

test("arena keeps reflected strategy results separate", async () => {
  const seenInputs: string[] = [];
  const registry = {
    "reflect:react": strategy("reflect:react", "Reflected ReAct", seenInputs),
    "reflect:plan-and-execute": strategy("reflect:plan-and-execute", "Reflected plan", seenInputs),
  };

  const runs = await runArena([
    "--strategies=reflect:react,reflect:plan-and-execute",
    "inspect firing alerts",
  ], registry);

  assert.deepEqual(runs.map((run) => run.strategy), ["reflect:react", "reflect:plan-and-execute"]);
  assert.deepEqual(runs.map((run) => run.answer), ["Reflected ReAct", "Reflected plan"]);
  assert.deepEqual(seenInputs, ["inspect firing alerts", "inspect firing alerts"]);
});
