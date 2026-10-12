import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { z } from "zod";
import { ValidationError } from "../services/errors.js";
import { resolveModel } from "./model.js";
import type { ReasoningOptions, ReasoningStrategy } from "./strategy.js";
import { ReasoningResultSchema, type ReasoningResult, type TraceEvent } from "./trace.js";

export const ReflectionOptionsSchema = z.object({
  maxReflections: z.number().int().nonnegative().default(2),
});

export const ReflectionCritiqueSchema = z.object({
  aproved: z.boolean(),
  feedback: z.string().describe(
    "If rejected, provide specific and actionable corrections. If approved, briefly explain why the answer is supported.",
  ),
});

const CRITIC_PROMPT = [
  "You are a strict answer critic.",
  "Evaluate the answer ONLY against the original request and the observations supplied in the trace.",
  "Treat the observations as the only factual evidence; do not rely on external knowledge or other trace events.",
  "Approve only when the answer addresses the request and is supported by the observations.",
  "If rejected, provide specific, actionable corrections in feedback. If approved, briefly state why it is supported.",
].join(" ");

function observationsOf(trace: readonly TraceEvent[]): string {
  const observations = trace
    .filter((event) => event.type === "observation")
    .map((event, index) => `${index + 1}. ${event.content}`);
  return observations.length > 0 ? observations.join("\n") : "(none)";
}

function revisionInput(request: string, feedback: string): string {
  return `${request}\n\nCritic feedback for revising the previous answer:\n${feedback}`;
}

async function critique(
  request: string,
  result: ReasoningResult,
  model: BaseChatModel,
): Promise<{ decision: z.infer<typeof ReflectionCritiqueSchema>; latencyMs: number }> {
  const startedAt = performance.now();
  const output = await model.withStructuredOutput(ReflectionCritiqueSchema).invoke([
    new SystemMessage(CRITIC_PROMPT),
    new HumanMessage(
      `Request: ${request}\nObservations:\n${observationsOf(result.trace)}\nAnswer: ${result.answer}`,
    ),
  ]);
  const decision = ReflectionCritiqueSchema.parse(output);
  return { decision, latencyMs: performance.now() - startedAt };
}

function parseAttempt(result: ReasoningResult): ReasoningResult {
  const parsed = ReasoningResultSchema.parse(result);
  if (!parsed.answer.trim()) {
    throw new ValidationError("Reflection cannot evaluate an empty strategy answer.");
  }
  return parsed;
}

export function withReflection(
  strategy: ReasoningStrategy,
  opts: z.input<typeof ReflectionOptionsSchema> = {},
): ReasoningStrategy {
  const { maxReflections } = ReflectionOptionsSchema.parse(opts);
  return {
    name: `reflect:${strategy.name}`,
    async run(input: string, options?: ReasoningOptions): Promise<ReasoningResult> {
      if (maxReflections === 0) {
        return parseAttempt(await strategy.run(input, options));
      }

      const model = resolveModel(options?.model);
      const strategyOptions: ReasoningOptions = { ...(options ?? {}), model };
      const trace: TraceEvent[] = [];
      let llmCalls = 0;
      let latencyMs = 0;
      let iterations = 0;
      let hasIterations = false;
      let result = parseAttempt(await strategy.run(input, strategyOptions));
      let reflectionCount = 0;
      let feedback = "";

      while (true) {
        trace.push(...result.trace);
        llmCalls += result.metrics.llmCalls;
        latencyMs += result.metrics.latencyMs;
        if (result.metrics.iterations !== undefined) {
          iterations += result.metrics.iterations;
          hasIterations = true;
        }

        const { decision, latencyMs: critiqueLatencyMs } = await critique(input, result, model);
        llmCalls += 1;
        latencyMs += critiqueLatencyMs;
        trace.push({
          type: "critique",
          content: JSON.stringify({ aproved: decision.aproved, feedback: decision.feedback }),
        });

        if (decision.aproved || reflectionCount >= maxReflections) {
          break;
        }

        reflectionCount += 1;
        feedback = decision.feedback;
        result = parseAttempt(await strategy.run(revisionInput(input, feedback), strategyOptions));
      }

      return ReasoningResultSchema.parse({
        answer: result.answer,
        trace,
        metrics: {
          llmCalls,
          latencyMs,
          ...(hasIterations ? { iterations } : {}),
        },
      });
    },
  };
}
