import { z } from "zod";

export const TraceEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("thought"), content: z.string() }),
  z.object({
    type: z.literal("action"),
    content: z.string(),
    tool: z.string().min(1),
    args: z.unknown().refine((value) => value !== undefined, "Action arguments are required."),
  }),
  z.object({ type: z.literal("observation"), content: z.string() }),
  z.object({ type: z.literal("plan"), content: z.string(), steps: z.array(z.string()).max(8) }),
  z.object({ type: z.literal("critique"), content: z.string() }),
  z.object({ type: z.literal("answer"), content: z.string() }),
]);

export const ReasoningMetricsSchema = z.object({
  llmCalls: z.number().nonnegative(),
  latencyMs: z.number().nonnegative(),
  iterations: z.number().int().nonnegative().optional(),
});

export const ReasoningResultSchema = z.object({
  answer: z.string(),
  trace: z.array(TraceEventSchema),
  metrics: ReasoningMetricsSchema,
});

export type TraceEvent = z.infer<typeof TraceEventSchema>;
export type ReasoningMetrics = z.infer<typeof ReasoningMetricsSchema>;
export type ReasoningResult = z.infer<typeof ReasoningResultSchema>;

export function parseTrace(events: readonly unknown[]): TraceEvent[] {
  return z.array(TraceEventSchema).parse(events);
}

export function formatTrace(events: readonly unknown[]): string {
  return parseTrace(events)
    .map((event, index) => `${index + 1}. ${event.type}: ${event.content}`)
    .join("\n");
}
