import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { z } from "zod";
import {
  defaultOpsStore,
  createOpsTools,
  ListAlertsInputSchema,
  ListIncidentsInputSchema,
  OpenIncidentInputSchema,
  ResolveIncidentInputSchema,
  type OpsTools,
} from "../services/tools.js";
import { IterationLimitError, ValidationError } from "../services/errors.js";
import type { OpsStore } from "../models/in-memory-store.js";
import { resolveModel } from "./model.js";
import {
  createExecutionContext,
  ReasoningOptionsSchema,
  type ReasoningOptions,
  type ReasoningStrategy,
} from "./strategy.js";
import type { TraceEvent } from "./trace.js";

export const PlanSchema = z.object({
  steps: z.array(z.string().trim().min(1)).max(8),
  answer: z.string().trim().min(1).optional(),
});

export const ToolExecutionSchema = z.object({
  tool: z.enum(["list_alerts", "list_incidents", "open_incident", "resolve_incident"]),
  args: z.record(z.unknown()),
});

export const ReplanSchema = z.object({
  decision: z.enum(["continue", "adjust", "finish"]),
  steps: z.array(z.string().trim().min(1)).max(8).optional(),
  answer: z.string().trim().min(1).optional(),
});

const PEState = Annotation.Root({
  input: Annotation<string>(),
  plan: Annotation<string[]>({
    value: (_current, update) => update,
    default: () => [],
  }),
  done: Annotation<Array<[string, string]>>({
    value: (current, update) => current.concat(update),
    default: () => [],
  }),
  answer: Annotation<string>({
    value: (_current, update) => update,
    default: () => "",
  }),
});

export type PEStateType = typeof PEState.State;

export interface PlanGraphOptions {
  model: BaseChatModel;
  beginIteration(): void;
  recordLlmCall(): void;
}

export function createPlanAndExecuteGraph(
  options: PlanGraphOptions,
  store: OpsStore = defaultOpsStore,
) {
  const [listAlerts, openIncident, resolveIncident, listIncidents] = createOpsTools(store);

  const planner = async (state: PEStateType) => {
    options.recordLlmCall();
    const output = await options.model.withStructuredOutput(PlanSchema).invoke([
      new SystemMessage(
        "Create an ordered operational plan as short executable tool steps. Return at most eight steps. If no tool is needed, return no steps and a concise answer.",
      ),
      new HumanMessage(state.input),
    ]);
    const parsed = PlanSchema.parse(output);
    if (parsed.steps.length === 0 && !parsed.answer) {
      throw new ValidationError("Planner must return steps or a final answer.");
    }
    return {
      plan: parsed.steps,
      answer: parsed.steps.length === 0 ? parsed.answer ?? "" : "",
    };
  };

  const executor = async (state: PEStateType) => {
    options.beginIteration();
    const step = state.plan[state.done.length];
    if (!step) {
      throw new ValidationError("Executor was called without a remaining plan step.");
    }

    options.recordLlmCall();
    const output = await options.model.withStructuredOutput(ToolExecutionSchema).invoke([
      new SystemMessage(
        "Resolve this plan step using exactly one available tool. Return the tool name and its JSON arguments. Never invent tool names.",
      ),
      new HumanMessage(
        `Request: ${state.input}\nStep: ${step}\nCompleted steps and tool results:\n${formatDone(state.done)}`,
      ),
    ]);
    const execution = ToolExecutionSchema.parse(output);
    const result = await invokeTool(execution.tool, execution.args, {
      listAlerts,
      openIncident,
      resolveIncident,
      listIncidents,
    });
    const serialized = JSON.stringify({
      tool: execution.tool,
      args: execution.args,
      output: typeof result === "string" ? result : JSON.stringify(result),
    });
    return { done: [[step, serialized] as [string, string]] };
  };

  const replanner = async (state: PEStateType) => {
    options.recordLlmCall();
    const currentSteps = state.plan.slice(state.done.length);
    const output = await options.model.withStructuredOutput(ReplanSchema).invoke([
      new SystemMessage(
        "Review the request, current remaining plan, and completed tool results. Decide explicitly: continue the current remaining steps, adjust them with a replacement list, or finish with a user-facing answer. Keep completed work in mind. The total count of completed steps plus replacement/remaining steps must never exceed eight. If no more work remains, finish.",
      ),
      new HumanMessage([
        `Request: ${state.input}`,
        `Remaining plan: ${JSON.stringify(currentSteps)}`,
        `Done (step/result pairs): ${JSON.stringify(state.done)}`,
      ].join("\n")),
    ]);
    const decision = ReplanSchema.parse(output);

    if (decision.decision === "continue") {
      if (decision.steps !== undefined || decision.answer !== undefined) {
        throw new ValidationError("A continue decision cannot include replacement steps or an answer.");
      }
      if (currentSteps.length === 0) {
        throw new ValidationError("Replanner cannot continue when no plan steps remain.");
      }
      return {};
    }
    if (decision.decision === "adjust") {
      if (!decision.steps?.length || decision.answer !== undefined) {
        throw new ValidationError("An adjust decision requires replacement steps and cannot include a final answer.");
      }
      if (state.done.length + decision.steps.length > 8) {
        throw new ValidationError("Replanner adjustment exceeds the eight-step total limit.");
      }
      return { plan: [...state.plan.slice(0, state.done.length), ...decision.steps] };
    }
    if (!decision.answer || decision.steps !== undefined) {
      throw new ValidationError("A finish decision requires an answer and cannot include replacement steps.");
    }
    return { answer: decision.answer };
  };

  return new StateGraph(PEState)
    .addNode("planner", planner)
    .addNode("executor", executor)
    .addNode("replanner", replanner)
    .addEdge(START, "planner")
    .addConditionalEdges("planner", (state) => state.answer || state.plan.length === 0 ? END : "executor")
    .addEdge("executor", "replanner")
    .addConditionalEdges("replanner", (state) => {
      if (state.answer) return END;
      if (state.done.length < state.plan.length) return "executor";
      return END;
    })
    .compile();
}

function formatDone(done: Array<[string, string]>): string {
  return done.map(([step, result]) => `${step}: ${result}`).join("\n") || "(none)";
}

async function invokeTool(
  name: z.infer<typeof ToolExecutionSchema>["tool"],
  args: Record<string, unknown>,
  tools: {
    listAlerts: OpsTools[0];
    openIncident: OpsTools[1];
    resolveIncident: OpsTools[2];
    listIncidents: OpsTools[3];
  },
): Promise<unknown> {
  switch (name) {
    case "list_alerts":
      return tools.listAlerts.invoke(ListAlertsInputSchema.parse(args));
    case "list_incidents":
      return tools.listIncidents.invoke(ListIncidentsInputSchema.parse(args));
    case "open_incident":
      return tools.openIncident.invoke(OpenIncidentInputSchema.parse(args));
    case "resolve_incident":
      return tools.resolveIncident.invoke(ResolveIncidentInputSchema.parse(args));
  }
}

function traceFromState(state: PEStateType): TraceEvent[] {
  const trace: TraceEvent[] = [
    { type: "plan", content: "Plan executed and revised as needed.", steps: state.plan },
  ];
  for (const [step, serialized] of state.done) {
    let details: unknown;
    try {
      details = JSON.parse(serialized) as unknown;
    } catch (error) {
      throw new ValidationError(`Plan step "${step}" returned invalid serialized output.`, { cause: error });
    }
    const parsed = z.object({
      tool: z.string().min(1),
      args: z.record(z.unknown()),
      output: z.string(),
    }).parse(details);
    trace.push(
      { type: "action", content: step, tool: parsed.tool, args: parsed.args },
      { type: "observation", content: parsed.output },
    );
  }
  if (state.answer) {
    trace.push({ type: "answer", content: state.answer });
  }
  return trace;
}

export const planAndExecuteStrategy: ReasoningStrategy = {
  name: "plan-and-execute",
  async run(input: string, options?: ReasoningOptions) {
    const { maxIterations } = ReasoningOptionsSchema.parse({ maxIterations: options?.maxIterations });
    if (!input.trim()) {
      throw new ValidationError("Input must not be empty.");
    }
    const context = createExecutionContext(maxIterations);
    const store = options?.store ?? defaultOpsStore;
    const model = resolveModel(options?.model);
    const graph = createPlanAndExecuteGraph({
      model,
      beginIteration: () => {
        if (context.iterations >= context.maxIterations) {
          throw new IterationLimitError(context.maxIterations);
        }
        context.iterations += 1;
      },
      recordLlmCall: () => context.recordLlmCall(),
    }, store);
    const result = await graph.invoke({ input });
    const answer = result.answer.trim();
    if (!answer) {
      throw new ValidationError("Plan-and-Execute completed without a final answer.");
    }
    context.trace.push(...traceFromState(result));
    return context.result(answer);
  },
};
