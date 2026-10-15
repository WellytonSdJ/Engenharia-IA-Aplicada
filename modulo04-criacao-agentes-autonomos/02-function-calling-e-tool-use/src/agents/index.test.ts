import assert from "node:assert/strict";
import test from "node:test";
import { resolveStrategy, strategyRegistry, type StrategyRegistry } from "./index.js";
import type { ReasoningStrategy } from "./strategy.js";
import { UnknownStrategyError } from "../services/errors.js";

function fakeStrategy(name: string): ReasoningStrategy {
  return {
    name,
    async run(input) {
      return {
        answer: `${name}: ${input}`,
        trace: [{ type: "answer", content: `${name}: ${input}` }],
        metrics: { llmCalls: 1, latencyMs: 1 },
      };
    },
  };
}

const fakeRegistry: StrategyRegistry = {
  react: fakeStrategy("react"),
  "plan-and-execute": fakeStrategy("plan-and-execute"),
};

test("resolveStrategy defaults to react when name is undefined", () => {
  const strategy = resolveStrategy(undefined, { reflect: false }, fakeRegistry);
  assert.equal(strategy, fakeRegistry.react);
});

test("resolveStrategy returns the same instance for a known base name", () => {
  const strategy = resolveStrategy("plan-and-execute", { reflect: false }, fakeRegistry);
  assert.equal(strategy, fakeRegistry["plan-and-execute"]);
});

test("resolveStrategy applies withReflection when reflect is true", () => {
  const strategy = resolveStrategy("react", { reflect: true }, fakeRegistry);
  assert.equal(strategy.name, "reflect:react");
  assert.notEqual(strategy, fakeRegistry.react);
});

test("resolveStrategy rejects unknown, prototype and reflected names", () => {
  for (const name of ["missing", "constructor", "reflect:react"]) {
    assert.throws(
      () => resolveStrategy(name, { reflect: false }, fakeRegistry),
      (error: unknown) =>
        error instanceof UnknownStrategyError &&
        error.strategy === name &&
        error.available.length === Object.keys(fakeRegistry).length &&
        error.available.every((key) => key in fakeRegistry),
    );
  }
});

test("production strategyRegistry contains exactly react and plan-and-execute", () => {
  assert.deepEqual(Object.keys(strategyRegistry), ["react", "plan-and-execute"]);
});
