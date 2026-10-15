import { z } from "zod";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { resolveStrategy, strategyRegistry, type StrategyRegistry } from "../agents/index.js";
import { ReasoningResultSchema } from "../agents/trace.js";
import { ConversationStore } from "../models/conversation-store.js";
import type { Message } from "../models/domain.js";
import { ExecutionTimeoutError } from "./errors.js";

export interface ChatDependencies {
  registry: StrategyRegistry;
  timeoutMs: number;
  model?: BaseChatModel;
  conversationStore: ConversationStore;
}

export const DEFAULT_CHAT_TIMEOUT_MS = 60_000;
export const MAX_HISTORY_MESSAGES = 12;

const TimeoutMsSchema = z.number().int().positive();

export function resolveChatDependencies(partial?: Partial<ChatDependencies>): ChatDependencies {
  return {
    registry: partial?.registry ?? strategyRegistry,
    timeoutMs: TimeoutMsSchema.parse(partial?.timeoutMs ?? DEFAULT_CHAT_TIMEOUT_MS),
    model: partial?.model,
    conversationStore: partial?.conversationStore ?? new ConversationStore(),
  };
}

export interface ChatRequest {
  message: string;
  strategy?: string;
  reflect: boolean;
  conversationId?: string;
}

export const ChatResultSchema = ReasoningResultSchema.extend({
  conversationId: z.string().min(1),
  metrics: ReasoningResultSchema.shape.metrics.extend({
    historyMessages: z.number().int().nonnegative(),
  }),
});

export type ChatResult = z.infer<typeof ChatResultSchema>;

function formatHistory(history: Message[]): string {
  return history
    .map((message) => `${message.role === "user" ? "User" : "Assistant"}: ${message.content}`)
    .join("\n");
}

function composeInput(message: string, history: Message[]): string {
  if (history.length === 0) {
    return message;
  }
  return `Conversation so far:\n${formatHistory(history)}\n\nUser: ${message}`;
}

export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  let timer: NodeJS.Timeout;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(new ExecutionTimeoutError(ms)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => {
    clearTimeout(timer);
    promise.catch(() => {});
  });
}

export async function runChat(request: ChatRequest, deps: ChatDependencies): Promise<ChatResult> {
  const strategy = resolveStrategy(request.strategy, { reflect: request.reflect }, deps.registry);
  const conversation = request.conversationId
    ? { id: request.conversationId }
    : await deps.conversationStore.create();
  const history = await deps.conversationStore.lastMessages(conversation.id, MAX_HISTORY_MESSAGES);
  const input = composeInput(request.message, history);

  const result = await withTimeout(
    strategy.run(input, deps.model ? { model: deps.model } : undefined),
    deps.timeoutMs,
  );
  const parsed = ReasoningResultSchema.parse(result);

  await deps.conversationStore.append(conversation.id, [
    { role: "user", content: request.message },
    { role: "assistant", content: parsed.answer },
  ]);

  return ChatResultSchema.parse({
    ...parsed,
    conversationId: conversation.id,
    metrics: { ...parsed.metrics, historyMessages: history.length },
  });
}
