import assert from "node:assert/strict";
import test from "node:test";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { StrategyRegistry } from "../agents/index.js";
import type { ReasoningOptions, ReasoningStrategy } from "../agents/strategy.js";
import { ConversationStore } from "../models/conversation-store.js";
import { ExecutionTimeoutError, UnknownConversationError } from "./errors.js";
import { resolveChatDependencies, runChat, withTimeout } from "./chat.js";

function fakeStrategy(
  name: string,
  seenInputs: string[],
  seenOptions: Array<ReasoningOptions | undefined>,
): ReasoningStrategy {
  return {
    name,
    async run(input, options) {
      seenInputs.push(input);
      seenOptions.push(options);
      return {
        answer: `${name}: ${input}`,
        trace: [{ type: "answer", content: `${name}: ${input}` }],
        metrics: { llmCalls: 1, latencyMs: 1 },
      };
    },
  };
}

test("withTimeout resolves with the value when the promise settles before the deadline", async () => {
  const result = await withTimeout(Promise.resolve("done"), 1000);
  assert.equal(result, "done");
});

test("withTimeout rejects with ExecutionTimeoutError when the deadline is exceeded", async () => {
  const pending = new Promise<never>(() => {});
  await assert.rejects(
    withTimeout(pending, 20),
    (error: unknown) => error instanceof ExecutionTimeoutError && error.timeoutMs === 20,
  );
});

test("withTimeout does not produce an unhandled rejection for a promise that rejects after the deadline", async () => {
  let unhandledCount = 0;
  const onUnhandled = () => {
    unhandledCount += 1;
  };
  process.on("unhandledRejection", onUnhandled);
  try {
    let rejectLate: (error: Error) => void;
    const late = new Promise<never>((_resolve, reject) => {
      rejectLate = reject;
    });
    await assert.rejects(withTimeout(late, 10), ExecutionTimeoutError);
    rejectLate!(new Error("late failure"));
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(unhandledCount, 0);
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }
});

test("withTimeout propagates the original error when it rejects before the deadline", async () => {
  const originalError = new Error("original failure");
  await assert.rejects(
    withTimeout(Promise.reject(originalError), 1000),
    (error: unknown) => error === originalError,
  );
});

function memoryDeps(partial?: Parameters<typeof resolveChatDependencies>[0]) {
  return resolveChatDependencies({ conversationStore: new ConversationStore(":memory:"), ...partial });
}

test("runChat passes the message through and omits options.model when no model is injected", async () => {
  const seenInputs: string[] = [];
  const seenOptions: Array<ReasoningOptions | undefined> = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", seenInputs, seenOptions) };
  const deps = memoryDeps({ registry });

  const result = await runChat({ message: "oi", reflect: false }, deps);

  assert.equal(result.answer, "react: oi");
  assert.deepEqual(seenInputs, ["oi"]);
  assert.equal(seenOptions[0], undefined);
});

test("runChat forwards deps.model as options.model", async () => {
  const seenInputs: string[] = [];
  const seenOptions: Array<ReasoningOptions | undefined> = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", seenInputs, seenOptions) };
  const model = {} as BaseChatModel;
  const deps = memoryDeps({ registry, model });

  await runChat({ message: "oi", reflect: false }, deps);

  assert.equal(seenOptions[0]?.model, model);
});

test("runChat without conversationId creates a new conversation and returns its id with historyMessages 0", async () => {
  const seenInputs: string[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", seenInputs, []) };
  const deps = memoryDeps({ registry });

  const result = await runChat({ message: "oi", reflect: false }, deps);

  assert.ok(result.conversationId.length > 0);
  assert.equal(result.metrics.historyMessages, 0);
  assert.deepEqual(seenInputs, ["oi"]);
});

test("runChat with an existing conversationId reuses the conversation and injects its history into the strategy input", async () => {
  const seenInputs: string[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", seenInputs, []) };
  const conversationStore = new ConversationStore(":memory:");
  const deps = memoryDeps({ registry, conversationStore });
  const conversation = await conversationStore.create();
  await conversationStore.append(conversation.id, [
    { role: "user", content: "pergunta anterior" },
    { role: "assistant", content: "resposta anterior" },
  ]);

  const result = await runChat({ message: "nova pergunta", reflect: false, conversationId: conversation.id }, deps);

  assert.equal(result.conversationId, conversation.id);
  assert.equal(result.metrics.historyMessages, 2);
  assert.match(seenInputs[0], /pergunta anterior/);
  assert.match(seenInputs[0], /resposta anterior/);
  assert.match(seenInputs[0], /nova pergunta/);
});

test("runChat with a conversation of 13 prior messages injects only the 12 most recent", async () => {
  const seenInputs: string[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", seenInputs, []) };
  const conversationStore = new ConversationStore(":memory:");
  const deps = memoryDeps({ registry, conversationStore });
  const conversation = await conversationStore.create();
  for (let i = 0; i < 13; i += 1) {
    await conversationStore.append(conversation.id, [{ role: "user", content: `msg-${i}` }]);
  }

  const result = await runChat({ message: "atual", reflect: false, conversationId: conversation.id }, deps);

  assert.equal(result.metrics.historyMessages, 12);
  assert.doesNotMatch(seenInputs[0], /msg-0(?!\d)/);
  assert.match(seenInputs[0], /msg-12/);
});

test("runChat with an unknown conversationId throws UnknownConversationError without invoking the strategy", async () => {
  const seenInputs: string[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", seenInputs, []) };
  const deps = memoryDeps({ registry });

  await assert.rejects(
    runChat({ message: "oi", reflect: false, conversationId: "conversation-missing" }, deps),
    (error: unknown) => error instanceof UnknownConversationError,
  );
  assert.deepEqual(seenInputs, []);
});

test("runChat does not persist the turn when the strategy fails", async () => {
  const registry: StrategyRegistry = {
    react: {
      name: "react",
      async run() {
        throw new Error("strategy failed");
      },
    },
  };
  const conversationStore = new ConversationStore(":memory:");
  const deps = memoryDeps({ registry, conversationStore });
  const conversation = await conversationStore.create();

  await assert.rejects(
    runChat({ message: "oi", reflect: false, conversationId: conversation.id }, deps),
    /strategy failed/,
  );

  const messages = await conversationStore.lastMessages(conversation.id, 12);
  assert.deepEqual(messages, []);
});
