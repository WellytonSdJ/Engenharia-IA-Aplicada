import assert from "node:assert/strict";
import test from "node:test";
import { RunnableLambda } from "@langchain/core/runnables";
import { SimpleChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseMessage } from "@langchain/core/messages";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { Runnable } from "@langchain/core/runnables";
import { z } from "zod";
import { IterationLimitError } from "../services/errors.js";
import { InMemoryOpsStore } from "../models/in-memory-store.js";
import { createPlanAndExecuteGraph, PlanSchema, ReplanSchema, ToolExecutionSchema } from "./plan-and-execute.js";

class ControlledStructuredModel extends SimpleChatModel {
  constructor(private readonly outputs: Record<string, unknown>[]) {
    super({});
  }

  _llmType(): string {
    return "controlled-structured-model";
  }

  async _call(
    _messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<string> {
    throw new Error("The structured-output test double should not make text model calls.");
  }

  withStructuredOutput<RunOutput extends Record<string, unknown>>(
    outputSchema: z.ZodType<RunOutput> | Record<string, unknown>,
  ): Runnable<BaseLanguageModelInput, RunOutput> {
    return RunnableLambda.from(async () => {
      const next = this.outputs.shift();
      if (next === undefined) {
        throw new Error("No controlled model response remains.");
      }
      if (outputSchema instanceof z.ZodType) {
        return outputSchema.parse(next) as RunOutput;
      }
      return next as RunOutput;
    });
  }
}

test("planner and replanner validate bounded steps and completion", () => {
  assert.equal(PlanSchema.parse({ steps: [], answer: "No action is needed." }).steps.length, 0);
  assert.equal(PlanSchema.safeParse({ steps: Array(9).fill("inspect") }).success, false);
  assert.equal(ReplanSchema.parse({ decision: "finish", answer: "Done." }).decision, "finish");
  assert.equal(ReplanSchema.safeParse({ decision: "adjust", steps: [""] }).success, false);
});

test("executor output is constrained to known tool names and structured arguments", () => {
  assert.deepEqual(ToolExecutionSchema.parse({ tool: "list_alerts", args: { status: "firing" } }), {
    tool: "list_alerts",
    args: { status: "firing" },
  });
  assert.equal(ToolExecutionSchema.safeParse({ tool: "drop_database", args: {} }).success, false);
  assert.equal(ToolExecutionSchema.safeParse({ tool: "list_incidents", args: { status: "all" } }).success, true);
});

test("plan graph stores executed step/result pairs and honors replanner finish", async () => {
  const store = new InMemoryOpsStore();
  const model = new ControlledStructuredModel([
    { steps: ["List active alerts"] },
    { tool: "list_alerts", args: { status: "firing" } },
    { decision: "finish", answer: "Three alerts are firing." },
  ]);
  let calls = 0;
  let iterations = 0;
  const graph = createPlanAndExecuteGraph({
    model,
    recordLlmCall: () => { calls += 1; },
    beginIteration: () => { iterations += 1; },
  }, store);

  const result = await graph.invoke({ input: "List firing alerts." });
  assert.equal(result.answer, "Three alerts are firing.");
  assert.equal(calls, 3);
  assert.equal(iterations, 1);
  assert.equal(result.done.length, 1);
  assert.equal(result.done[0][0], "List active alerts");
  assert.match(result.done[0][1], /"tool":"list_alerts"/);
});

test("plan graph finishes immediately when the planner has no steps", async () => {
  const zeroStepGraph = createPlanAndExecuteGraph({
    model: new ControlledStructuredModel([{ steps: [], answer: "No action is needed." }]),
    recordLlmCall: () => {},
    beginIteration: () => assert.fail("No execution cycle should start for an empty plan."),
  }, new InMemoryOpsStore());
  const completed = await zeroStepGraph.invoke({ input: "Is anything firing?" });
  assert.equal(completed.answer, "No action is needed.");
  assert.deepEqual(completed.plan, []);
  assert.deepEqual(completed.done, []);
});

test("replanner may continue current plan or replace its remaining steps", async () => {
  const model = new ControlledStructuredModel([
    { steps: ["Initial", "Obsolete"] },
    { tool: "list_alerts", args: { status: "firing" } },
    { decision: "adjust", steps: ["Replacement", "Final check"] },
    { tool: "list_alerts", args: { status: "resolved" } },
    { decision: "continue" },
    { tool: "list_alerts", args: { status: "all" } },
    { decision: "finish", answer: "All alert groups checked." },
  ]);
  let iterations = 0;
  const graph = createPlanAndExecuteGraph({
    model,
    recordLlmCall: () => {},
    beginIteration: () => { iterations += 1; },
  }, new InMemoryOpsStore());

  const result = await graph.invoke({ input: "Check alert groups." });
  assert.equal(result.answer, "All alert groups checked.");
  assert.deepEqual(result.done.map(([step]) => step), ["Initial", "Replacement", "Final check"]);
  assert.deepEqual(result.plan, ["Initial", "Replacement", "Final check"]);
  assert.equal(iterations, 3);
});

test("replanner cannot increase total planned-plus-done work above eight", async () => {
  const guardedGraph = createPlanAndExecuteGraph({
    model: new ControlledStructuredModel([
      { steps: Array.from({ length: 8 }, (_, index) => `Step ${index + 1}`) },
      { tool: "list_alerts", args: { status: "firing" } },
      { decision: "adjust", steps: Array.from({ length: 8 }, (_, index) => `Replacement ${index + 1}`) },
    ]),
    recordLlmCall: () => {},
    beginIteration: () => {},
  }, new InMemoryOpsStore());
  await assert.rejects(guardedGraph.invoke({ input: "Check alerts." }), /eight-step total limit/);
});

test("graph checks configured iteration cap before tool execution", async () => {
  const guardedGraph = createPlanAndExecuteGraph({
    model: new ControlledStructuredModel([{ steps: ["Inspect alerts"] }]),
    recordLlmCall: () => {},
    beginIteration: () => { throw new IterationLimitError(1); },
  }, new InMemoryOpsStore());
  await assert.rejects(guardedGraph.invoke({ input: "Check alerts." }), IterationLimitError);
});

test("replanner rejects a finish decision without a final answer", async () => {
  const graph = createPlanAndExecuteGraph({
    model: new ControlledStructuredModel([
      { steps: ["Inspect alerts"] },
      { tool: "list_alerts", args: { status: "firing" } },
      { decision: "finish" },
    ]),
    recordLlmCall: () => {},
    beginIteration: () => {},
  }, new InMemoryOpsStore());
  await assert.rejects(graph.invoke({ input: "Check alerts." }), /finish decision requires an answer/);
});
