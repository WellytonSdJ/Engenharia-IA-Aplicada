import { z } from "zod";
import { ReasoningResultSchema } from "../agents/trace.js";
import type { ReasoningOptions, ReasoningStrategy } from "../agents/strategy.js";

const ArenaInputSchema = z.object({
  strategies: z.array(z.string().min(1)).min(1),
  maxIterations: z.number().int().positive(),
  input: z.string().trim().min(1),
});

export interface ArenaArguments {
  strategies: string[];
  maxIterations: number;
  input: string;
}

export interface ArenaRun {
  strategy: string;
  answer: string;
  trace: ReturnType<typeof ReasoningResultSchema.parse>["trace"];
  metrics: ReturnType<typeof ReasoningResultSchema.parse>["metrics"];
}

export function parseArenaArgs(args: readonly string[]): ArenaArguments {
  let strategyList = "react,plan-and-execute";
  let maxIterations = 8;
  const positional: string[] = [];

  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];
    if (value === "--strategies" || value === "--max-iterations") {
      const next = args[index + 1];
      if (!next || next.startsWith("--")) {
        throw new Error(`Missing value for ${value}.`);
      }
      if (value === "--strategies") {
        strategyList = next;
      } else {
        const parsed = Number(next);
        if (!Number.isInteger(parsed) || parsed <= 0) {
          throw new Error("--max-iterations must be a positive integer.");
        }
        maxIterations = parsed;
      }
      index += 1;
    } else if (value.startsWith("--strategies=")) {
      strategyList = value.slice("--strategies=".length);
    } else if (value.startsWith("--max-iterations=")) {
      const parsed = Number(value.slice("--max-iterations=".length));
      if (!Number.isInteger(parsed) || parsed <= 0) {
        throw new Error("--max-iterations must be a positive integer.");
      }
      maxIterations = parsed;
    } else if (value.startsWith("--")) {
      throw new Error(`Unknown option "${value}".`);
    } else {
      positional.push(value);
    }
  }

  const parsed = ArenaInputSchema.parse({
    strategies: strategyList.split(",").map((name) => name.trim()),
    maxIterations,
    input: positional.join(" "),
  });
  return {
    strategies: parsed.strategies,
    maxIterations: parsed.maxIterations,
    input: parsed.input,
  };
}

export async function runArena(
  args: readonly string[],
  strategyRegistry: Readonly<Record<string, ReasoningStrategy>>,
): Promise<ArenaRun[]> {
  const parsed = parseArenaArgs(args);
  const selected = parsed.strategies.map((name) => {
    const strategy = strategyRegistry[name];
    if (!strategy) {
      throw new Error(`Unknown strategy "${name}". Available strategies: ${Object.keys(strategyRegistry).join(", ")}.`);
    }
    return strategy;
  });
  const results: ArenaRun[] = [];
  const options: ReasoningOptions = { maxIterations: parsed.maxIterations };
  for (const strategy of selected) {
    const result = ReasoningResultSchema.parse(await strategy.run(parsed.input, options));
    results.push({ strategy: strategy.name, ...result });
  }
  return results;
}

export function formatArenaRuns(runs: readonly ArenaRun[]): string {
  return runs.map((run) => [
    `=== ${run.strategy} ===`,
    `Answer: ${run.answer}`,
    `Trace:\n${run.trace.map((event, index) => `  ${index + 1}. ${event.type}: ${event.content}`).join("\n")}`,
    `Metrics: ${JSON.stringify(run.metrics)}`,
  ].join("\n")).join("\n\n");
}
