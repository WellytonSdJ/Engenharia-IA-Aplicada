import { z } from "zod";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { OpsStore } from "../models/in-memory-store.js";
import { IterationLimitError } from "../services/errors.js";
import { ReasoningResultSchema, type ReasoningResult, type TraceEvent } from "./trace.js";

export const ReasoningOptionsSchema = z.object({
  maxIterations: z.number().int().positive().default(8),
});

export type ReasoningOptions = z.input<typeof ReasoningOptionsSchema> & {
  model?: BaseChatModel;
  store?: OpsStore;
};

export interface ReasoningStrategy {
  readonly name: string;
  run(input: string, options?: ReasoningOptions): Promise<ReasoningResult>;
}

export interface ExecutionContext {
  readonly maxIterations: number;
  readonly trace: TraceEvent[];
  llmCalls: number;
  iterations: number;
  recordLlmCall(): void;
  beginIteration(): void;
  result(answer: string): ReasoningResult;
}

export function createExecutionContext(maxIterations = 8): ExecutionContext {
  const parsed = ReasoningOptionsSchema.parse({ maxIterations });
  const startedAt = performance.now();
  const trace: TraceEvent[] = [];
  let llmCalls = 0;
  let iterations = 0;

  return {
    maxIterations: parsed.maxIterations,
    trace,
    get llmCalls() {
      return llmCalls;
    },
    set llmCalls(value: number) {
      llmCalls = value;
    },
    get iterations() {
      return iterations;
    },
    set iterations(value: number) {
      iterations = value;
    },
    recordLlmCall() {
      llmCalls += 1;
    },
    beginIteration() {
      if (iterations >= parsed.maxIterations) {
        throw new IterationLimitError(parsed.maxIterations);
      }
      iterations += 1;
    },
    result(answer: string) {
      const result = ReasoningResultSchema.parse({
        answer,
        trace,
        metrics: {
          llmCalls,
          latencyMs: performance.now() - startedAt,
          iterations,
        },
      });
      return result;
    },
  };
}
