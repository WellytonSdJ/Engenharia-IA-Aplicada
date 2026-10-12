import assert from "node:assert/strict";
import test from "node:test";
import { RunnableLambda } from "@langchain/core/runnables";
import { SimpleChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseMessage } from "@langchain/core/messages";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { Runnable } from "@langchain/core/runnables";
import { z } from "zod";
import type { AlertWithService, Incident, IncidentWithService, Severity, StoreSummary } from "../models/domain.js";
import { IterationLimitError } from "../services/errors.js";
import { InMemoryOpsStore, type OpsStore } from "../models/in-memory-store.js";
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

class RecordingOpsStore implements OpsStore {
  readonly operations: string[] = [];
  private incidents: Incident[] = [];

  async listAlerts(): Promise<AlertWithService[]> {
    return [];
  }

  async listIncidents(): Promise<IncidentWithService[]> {
    return [];
  }

  async createIncident(input: { title: string; service: string; severity: Severity }): Promise<Incident> {
    const incident: Incident = {
      id: `incident-${this.incidents.length + 1}`,
      title: input.title,
      serviceId: `service-${input.service}`,
      severity: input.severity,
      status: "open",
      createdAt: new Date(),
      resolvedAt: null,
    };
    this.incidents.push(incident);
    this.operations.push(`open:${input.service}`);
    return incident;
  }

  async resolveIncident(id: string): Promise<Incident> {
    const incident = this.incidents.find((item) => item.id === id);
    assert.ok(incident, `Expected incident ${id} to exist.`);
    const resolved = { ...incident, status: "resolved" as const, resolvedAt: new Date() };
    this.incidents = this.incidents.map((item) => item.id === id ? resolved : item);
    this.operations.push(`resolve:${id}`);
    return resolved;
  }

  async reset(): Promise<void> {
    this.incidents = [];
    this.operations.length = 0;
  }

  async readSummary(): Promise<StoreSummary> {
    return { services: 3, alerts: 0, firing: 0, resolved: 0, incidents: this.incidents.length };
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

test("without replanner the graph runs every planned step in order and finishes with one final call", async () => {
  const store = new RecordingOpsStore();
  // Sem respostas de replanner na fila: se ele fosse chamado, o FinalAnswerSchema/ReplanSchema quebraria.
  const model = new ControlledStructuredModel([
    { steps: ["Open checkout", "Open payments"] },
    { tool: "open_incident", args: { title: "Checkout", service: "checkout", severity: "high" } },
    { tool: "open_incident", args: { title: "Payments", service: "payments", severity: "high" } },
    { answer: "Opened two incidents." },
  ]);
  let calls = 0;
  const graph = createPlanAndExecuteGraph({
    model,
    recordLlmCall: () => { calls += 1; },
    beginIteration: () => {},
    replanner: false,
  }, store);

  const result = await graph.invoke({ input: "Open incidents for checkout and payments." });
  assert.equal(result.answer, "Opened two incidents.");
  assert.deepEqual(store.operations, ["open:checkout", "open:payments"]);
  // planner + 2 executores + finalizer = 4 (com replanner seriam 5: uma revisão por passo).
  assert.equal(calls, 4);
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

test("replanner corrects one contradictory response without repeating completed incident actions", async () => {
  const store = new RecordingOpsStore();
  const model = new ControlledStructuredModel([
    { steps: ["Open checkout", "Open payments", "Open catalog", "Resolve first incident"] },
    { tool: "open_incident", args: { title: "Checkout issue", service: "checkout", severity: "high" } },
    { decision: "continue", steps: ["Wrong replacement"], answer: "Premature completion." },
    { decision: "continue" },
    { tool: "open_incident", args: { title: "Payments issue", service: "payments", severity: "high" } },
    { decision: "continue" },
    { tool: "open_incident", args: { title: "Catalog issue", service: "catalog", severity: "high" } },
    { decision: "continue" },
    { tool: "resolve_incident", args: { id: "incident-1" } },
    { decision: "finish", answer: "Three incidents opened; the first was resolved." },
  ]);
  let calls = 0;
  const graph = createPlanAndExecuteGraph({
    model,
    recordLlmCall: () => { calls += 1; },
    beginIteration: () => {},
  }, store);

  const result = await graph.invoke({ input: "Open incidents for checkout, payments, and catalog, then resolve the first." });

  assert.equal(result.answer, "Three incidents opened; the first was resolved.");
  assert.deepEqual(store.operations, [
    "open:checkout",
    "open:payments",
    "open:catalog",
    "resolve:incident-1",
  ]);
  assert.equal(result.done.length, 4);
  assert.equal(calls, 10);
});

test("replanner fails explicitly after a second contradictory response", async () => {
  const store = new RecordingOpsStore();
  const graph = createPlanAndExecuteGraph({
    model: new ControlledStructuredModel([
      { steps: ["Open checkout"] },
      { tool: "open_incident", args: { title: "Checkout issue", service: "checkout", severity: "high" } },
      { decision: "continue", answer: "Contradictory answer." },
      { decision: "continue", steps: ["Contradictory replacement"] },
    ]),
    recordLlmCall: () => {},
    beginIteration: () => {},
  }, store);

  await assert.rejects(
    graph.invoke({ input: "Open an incident for checkout." }),
    /A continue decision cannot include replacement steps or an answer/,
  );
  assert.deepEqual(store.operations, ["open:checkout"]);
});

test("replanner cannot increase total planned-plus-done work above eight", async () => {
  const guardedGraph = createPlanAndExecuteGraph({
    model: new ControlledStructuredModel([
      { steps: Array.from({ length: 8 }, (_, index) => `Step ${index + 1}`) },
      { tool: "list_alerts", args: { status: "firing" } },
      { decision: "adjust", steps: Array.from({ length: 8 }, (_, index) => `Replacement ${index + 1}`) },
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
      { decision: "finish" },
    ]),
    recordLlmCall: () => {},
    beginIteration: () => {},
  }, new InMemoryOpsStore());
  await assert.rejects(graph.invoke({ input: "Check alerts." }), /finish decision requires an answer/);
});
