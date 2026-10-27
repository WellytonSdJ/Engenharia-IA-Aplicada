import assert from "node:assert/strict";
import test from "node:test";
import { IterationLimitError } from "../services/errors.js";
import { createExecutionContext, ReasoningOptionsSchema, type ReasoningStrategy } from "./strategy.js";
import { ReasoningResultSchema } from "./trace.js";

test("strategy contract returns validated response and metrics using a deterministic double", async () => {
  const strategy: ReasoningStrategy = {
    name: "double",
    async run() {
      const execution = createExecutionContext(2);
      execution.recordLlmCall();
      execution.beginIteration();
      execution.trace.push({ type: "answer", content: "No active incident." });
      return execution.result("No active incident.");
    },
  };
  const result = ReasoningResultSchema.parse(await strategy.run("check incidents"));
  assert.equal(result.answer, "No active incident.");
  assert.equal(result.metrics.llmCalls, 1);
  assert.equal(result.metrics.iterations, 1);
  assert.ok(result.metrics.latencyMs >= 0);
});

test("execution context raises a structured iteration-limit error before the next cycle", () => {
  const execution = createExecutionContext(1);
  execution.beginIteration();
  assert.throws(() => execution.beginIteration(), IterationLimitError);
  assert.equal(ReasoningOptionsSchema.safeParse({ maxIterations: 0 }).success, false);
});
