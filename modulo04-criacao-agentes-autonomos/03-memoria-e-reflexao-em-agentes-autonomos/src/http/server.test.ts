import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import test, { after, afterEach } from "node:test";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import { SimpleChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseMessage } from "@langchain/core/messages";
import { RunnableLambda, type Runnable } from "@langchain/core/runnables";
import { z } from "zod";
import type { StrategyRegistry } from "../agents/index.js";
import type { ReasoningOptions, ReasoningStrategy } from "../agents/strategy.js";
import { ReasoningResultSchema, type ReasoningResult } from "../agents/trace.js";
import type { ChatDependencies } from "../services/chat.js";
import { ConversationStore } from "../models/conversation-store.js";
import { SqliteMemoryStore, type MemoryStore } from "../models/memory-store.js";
import { EmbeddingError } from "../services/errors.js";
import { FakeEmbedder } from "../services/fake-embedder.js";
import { createServer } from "./server.js";

class ControlledCriticModel extends SimpleChatModel {
  readonly inputs: BaseLanguageModelInput[] = [];

  constructor(private readonly outputs: Array<unknown | Error>) {
    super({});
  }

  _llmType(): string {
    return "controlled-critic-model";
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
    return RunnableLambda.from(async (input: BaseLanguageModelInput) => {
      this.inputs.push(input);
      const next = this.outputs.shift();
      if (next instanceof Error) {
        throw next;
      }
      if (next === undefined) {
        throw new Error("No controlled critic response remains.");
      }
      if (outputSchema instanceof z.ZodType) {
        return outputSchema.parse(next) as RunOutput;
      }
      return next as RunOutput;
    });
  }
}

interface FakeCall {
  input: string;
  options: ReasoningOptions | undefined;
}

function fakeStrategy(name: string, calls: FakeCall[]): ReasoningStrategy {
  return {
    name,
    async run(input, options) {
      calls.push({ input, options });
      const result: ReasoningResult = {
        answer: `${name}: ${input}`,
        trace: [
          { type: "observation", content: "obs" },
          { type: "answer", content: `${name}: ${input}` },
        ],
        metrics: { llmCalls: 1, latencyMs: 1 },
      };
      return result;
    },
  };
}

function hangingStrategy(name: string): ReasoningStrategy {
  return {
    name,
    run() {
      return new Promise<ReasoningResult>(() => {});
    },
  };
}

function throwingStrategy(name: string, message: string): ReasoningStrategy {
  return {
    name,
    async run() {
      throw new Error(message);
    },
  };
}

async function startTestServer(deps?: Partial<ChatDependencies>): Promise<{ url: string; close: () => Promise<void> }> {
  const resolved: Partial<ChatDependencies> = {
    conversationStore: new ConversationStore(":memory:"),
    // Sem isso o servidor usaria o embedder real e tentaria baixar o modelo durante os testes.
    memoryStore: new SqliteMemoryStore(":memory:", new FakeEmbedder()),
    ...deps,
  };
  const server = createServer(resolved).listen(0, "127.0.0.1");
  await new Promise<void>((resolve) => server.once("listening", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

async function postChat(url: string, body: unknown | string): Promise<{ status: number; body: unknown }> {
  const response = await fetch(`${url}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
  const text = await response.text();
  return { status: response.status, body: text ? JSON.parse(text) : undefined };
}

const openServers: Array<() => Promise<void>> = [];

afterEach(async () => {
  while (openServers.length > 0) {
    const close = openServers.pop();
    await close?.();
  }
});

async function withServer(deps?: Partial<ChatDependencies>) {
  const { url, close } = await startTestServer(deps);
  openServers.push(close);
  return url;
}

test("200 with the default react strategy, calling only the fake react strategy", async () => {
  const calls: FakeCall[] = [];
  const planCalls: FakeCall[] = [];
  const registry: StrategyRegistry = {
    react: fakeStrategy("react", calls),
    "plan-and-execute": fakeStrategy("plan-and-execute", planCalls),
  };
  const url = await withServer({ registry });

  const { status, body } = await postChat(url, { userId: "ana", message: "oi" });

  assert.equal(status, 200);
  assert.equal((body as { answer: string }).answer, "react: oi");
  assert.equal(calls.length, 1);
  assert.equal(planCalls.length, 0);
  assert.doesNotThrow(() => ReasoningResultSchema.parse(body));
});

test("200 with a named strategy, calling only that fake strategy", async () => {
  const reactCalls: FakeCall[] = [];
  const planCalls: FakeCall[] = [];
  const registry: StrategyRegistry = {
    react: fakeStrategy("react", reactCalls),
    "plan-and-execute": fakeStrategy("plan-and-execute", planCalls),
  };
  const url = await withServer({ registry });

  const { status, body } = await postChat(url, { userId: "ana", message: "oi", strategy: "plan-and-execute" });

  assert.equal(status, 200);
  assert.equal((body as { answer: string }).answer, "plan-and-execute: oi");
  assert.equal(reactCalls.length, 0);
  assert.equal(planCalls.length, 1);
});

test("extra fields in the body are ignored", async () => {
  const calls: FakeCall[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", calls) };
  const url = await withServer({ registry });

  const { status } = await postChat(url, { userId: "ana", message: "oi", extra: 1 });

  assert.equal(status, 200);
});

test("reflect: true wraps the strategy with reflection and produces a critique event", async () => {
  const calls: FakeCall[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", calls) };
  const model = new ControlledCriticModel([{ aproved: true, feedback: "ok" }]);
  const url = await withServer({ registry, model });

  const { status, body } = await postChat(url, { userId: "ana", message: "oi", reflect: true });

  assert.equal(status, 200);
  const trace = (body as ReasoningResult).trace;
  assert.equal(trace.filter((event) => event.type === "critique").length, 1);
  assert.ok((body as ReasoningResult).metrics.llmCalls > 1);
  assert.equal(calls.length, 1);
  assert.equal(model.inputs.length, 1);
});

test("reflect absent or false runs without reflection and without consuming the critic", async () => {
  const calls: FakeCall[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", calls) };
  const model = new ControlledCriticModel([{ aproved: true, feedback: "ok" }]);
  const url = await withServer({ registry, model });

  const first = await postChat(url, { userId: "ana", message: "oi" });
  const second = await postChat(url, { userId: "ana", message: "oi", reflect: false });

  for (const { status, body } of [first, second]) {
    assert.equal(status, 200);
    const trace = (body as ReasoningResult).trace;
    assert.equal(trace.filter((event) => event.type === "critique").length, 0);
  }
  assert.equal(model.inputs.length, 0);
});

test("400 for malformed JSON, with a synthetic issue at path []", async () => {
  const registry: StrategyRegistry = { react: fakeStrategy("react", []) };
  const url = await withServer({ registry });

  const { status, body } = await postChat(url, "{message");

  assert.equal(status, 400);
  const parsed = body as { error: string; issues: Array<{ path: unknown[]; message: string }> };
  assert.equal(parsed.error, "INVALID_BODY");
  assert.deepEqual(parsed.issues[0].path, []);
  assert.equal(parsed.issues[0].message, "Malformed JSON body.");
});

test("400 for invalid bodies, without ever calling the fake strategy", async () => {
  const calls: FakeCall[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", calls) };
  const url = await withServer({ registry });

  const cases: Array<{ body: unknown; field?: string }> = [
    { body: {} , field: "message" },
    { body: { userId: "ana", message: "" }, field: "message" },
    { body: { userId: "ana", message: "   " }, field: "message" },
    { body: { userId: "ana", message: 1 }, field: "message" },
    { body: { message: "oi" }, field: "userId" },
    { body: { userId: "  ", message: "oi" }, field: "userId" },
    { body: { userId: "ana", message: "oi", strategy: "" }, field: "strategy" },
    { body: { userId: "ana", message: "oi", reflect: "true" }, field: "reflect" },
  ];

  for (const testCase of cases) {
    const { status, body } = await postChat(url, testCase.body);
    assert.equal(status, 400, JSON.stringify(testCase.body));
    const parsed = body as { error: string; issues: Array<{ path: unknown[] }> };
    assert.equal(parsed.error, "INVALID_BODY");
    assert.ok(parsed.issues.length > 0);
    if (testCase.field) {
      assert.equal(parsed.issues[0].path[0], testCase.field);
    }
  }
  assert.equal(calls.length, 0);
});

test("422 for an unknown strategy, including reflected-variant names", async () => {
  const registry: StrategyRegistry = { react: fakeStrategy("react", []) };
  const url = await withServer({ registry });

  const missing = await postChat(url, { userId: "ana", message: "oi", strategy: "missing" });
  assert.equal(missing.status, 422);
  const missingBody = missing.body as { error: string; strategy: string; available: string[] };
  assert.equal(missingBody.error, "UNKNOWN_STRATEGY");
  assert.equal(missingBody.strategy, "missing");
  assert.deepEqual(missingBody.available, Object.keys(registry));

  const reflected = await postChat(url, { userId: "ana", message: "oi", strategy: "reflect:react" });
  assert.equal(reflected.status, 422);
});

test("400 wins over 422 when both the body and the strategy are invalid", async () => {
  const registry: StrategyRegistry = { react: fakeStrategy("react", []) };
  const url = await withServer({ registry });

  const { status, body } = await postChat(url, { userId: "ana", message: "", strategy: "missing" });

  assert.equal(status, 400);
  assert.equal((body as { error: string }).error, "INVALID_BODY");
});

test("504 when the strategy never resolves within timeoutMs", async () => {
  const registry: StrategyRegistry = { react: hangingStrategy("react") };
  const url = await withServer({ registry, timeoutMs: 50 });

  const startedAt = Date.now();
  const { status, body } = await postChat(url, { userId: "ana", message: "oi" });

  assert.equal(status, 504);
  assert.equal((body as { error: string }).error, "TIMEOUT");
  assert.ok(Date.now() - startedAt < 1000);
});

test("500 without leaking the original error message", async () => {
  const registry: StrategyRegistry = { react: throwingStrategy("react", "secret detail") };
  const url = await withServer({ registry });

  const response = await fetch(`${url}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ userId: "ana", message: "oi" }),
  });
  const text = await response.text();

  assert.equal(response.status, 500);
  assert.doesNotMatch(text, /secret detail/);
  assert.equal(JSON.parse(text).message, "Internal server error.");
});

test("200 without conversationId creates a conversation and returns it; reusing it on a second call keeps the same id", async () => {
  const calls: FakeCall[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", calls) };
  const url = await withServer({ registry });

  const first = await postChat(url, { userId: "ana", message: "primeira mensagem" });
  assert.equal(first.status, 200);
  const firstBody = first.body as { conversationId: string; metrics: { historyMessages: number } };
  assert.ok(firstBody.conversationId.length > 0);
  assert.equal(firstBody.metrics.historyMessages, 0);

  const second = await postChat(url, { userId: "ana", message: "segunda mensagem", conversationId: firstBody.conversationId });
  assert.equal(second.status, 200);
  const secondBody = second.body as { conversationId: string; metrics: { historyMessages: number } };
  assert.equal(secondBody.conversationId, firstBody.conversationId);
  assert.equal(secondBody.metrics.historyMessages, 2);
  assert.equal(calls.length, 2);
  assert.match(calls[1].input, /primeira mensagem/);
  assert.match(calls[1].input, /segunda mensagem/);
});

test("422 for a well-formed but nonexistent conversationId, without ever calling the fake strategy", async () => {
  const calls: FakeCall[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", calls) };
  const url = await withServer({ registry });

  const { status, body } = await postChat(url, { userId: "ana", message: "oi", conversationId: "conversation-does-not-exist" });

  assert.equal(status, 422);
  const parsed = body as { error: string; conversationId: string };
  assert.equal(parsed.error, "UNKNOWN_CONVERSATION");
  assert.equal(parsed.conversationId, "conversation-does-not-exist");
  assert.equal(calls.length, 0);
});

test("400 for an empty-string conversationId", async () => {
  const registry: StrategyRegistry = { react: fakeStrategy("react", []) };
  const url = await withServer({ registry });

  const { status, body } = await postChat(url, { userId: "ana", message: "oi", conversationId: "" });

  assert.equal(status, 400);
  const parsed = body as { error: string; issues: Array<{ path: unknown[] }> };
  assert.equal(parsed.error, "INVALID_BODY");
  assert.equal(parsed.issues[0].path[0], "conversationId");
});

test("historyMessages caps at 12 even when the conversation has more prior messages", async () => {
  const calls: FakeCall[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", calls) };
  const conversationStore = new ConversationStore(":memory:");
  const url = await withServer({ registry, conversationStore });

  const created = await conversationStore.create();
  for (let i = 0; i < 13; i += 1) {
    await conversationStore.append(created.id, [{ role: "user", content: `turno-${i}` }]);
  }

  const { status, body } = await postChat(url, { userId: "ana", message: "mensagem atual", conversationId: created.id });

  assert.equal(status, 200);
  assert.equal((body as { metrics: { historyMessages: number } }).metrics.historyMessages, 12);
});

test("a failing strategy does not persist the turn to the conversation", async () => {
  const registry: StrategyRegistry = { react: throwingStrategy("react", "boom") };
  const conversationStore = new ConversationStore(":memory:");
  const url = await withServer({ registry, conversationStore });

  const created = await conversationStore.create();
  const { status } = await postChat(url, { userId: "ana", message: "oi", conversationId: created.id });

  assert.equal(status, 500);
  const messages = await conversationStore.lastMessages(created.id, 12);
  assert.deepEqual(messages, []);
});

async function postJson(url: string, path: string, body: unknown): Promise<{ status: number; body: unknown }> {
  const response = await fetch(`${url}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: response.status, body: await response.json() };
}

test("POST /memories stores a fact (201) and a near-duplicate returns the existing id (200)", async () => {
  const embedder = new FakeEmbedder().setAxis("checkout primeiro", 0).setAxis("checkout primeiro!", 0);
  const url = await withServer({ memoryStore: new SqliteMemoryStore(":memory:", embedder) });

  const created = await postJson(url, "/memories", { userId: "ana", fact: "checkout primeiro" });
  const duplicate = await postJson(url, "/memories", { userId: "ana", fact: "checkout primeiro!" });

  assert.equal(created.status, 201);
  const createdBody = created.body as { id: string; stored: boolean; userId: string; fact: string };
  assert.equal(createdBody.stored, true);
  assert.equal(createdBody.userId, "ana");
  assert.equal(duplicate.status, 200);
  assert.deepEqual(duplicate.body, { id: createdBody.id, stored: false, userId: "ana", fact: "checkout primeiro!" });
});

test("POST /memories rejects an invalid body with 400", async () => {
  const url = await withServer();

  for (const body of [{}, { userId: "ana" }, { userId: "ana", fact: "  " }, { userId: "", fact: "x" }]) {
    const { status, body: response } = await postJson(url, "/memories", body);
    assert.equal(status, 400, JSON.stringify(body));
    assert.equal((response as { error: string }).error, "INVALID_BODY");
  }
});

test("/chat injects recalled memories of that user into the prompt and reports recalledMemories", async () => {
  const calls: FakeCall[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", calls) };
  const embedder = new FakeEmbedder()
    .setAxis("sempre priorize o checkout", 0)
    .setAxis("o que eu ataco primeiro?", 0);
  const memoryStore = new SqliteMemoryStore(":memory:", embedder);
  await memoryStore.remember("ana", "sempre priorize o checkout");
  const url = await withServer({ registry, memoryStore });

  const ana = await postChat(url, { userId: "ana", message: "o que eu ataco primeiro?" });
  const bia = await postChat(url, { userId: "bia", message: "o que eu ataco primeiro?" });

  assert.equal(ana.status, 200);
  assert.equal((ana.body as { metrics: { recalledMemories: number } }).metrics.recalledMemories, 1);
  assert.match(calls[0].input, /Relevant memories:\n- sempre priorize o checkout/);
  assert.equal((bia.body as { metrics: { recalledMemories: number } }).metrics.recalledMemories, 0);
  assert.equal(calls[1].input, "o que eu ataco primeiro?");
});

test("503 when the embedding model is unavailable during recall", async () => {
  const calls: FakeCall[] = [];
  const registry: StrategyRegistry = { react: fakeStrategy("react", calls) };
  const brokenMemory: MemoryStore = {
    async remember() {
      throw new EmbeddingError("model download failed");
    },
    async recall() {
      throw new EmbeddingError("model download failed");
    },
    async forget() {
      return false;
    },
  };
  const url = await withServer({ registry, memoryStore: brokenMemory });

  const { status, body } = await postChat(url, { userId: "ana", message: "oi" });

  assert.equal(status, 503);
  assert.equal((body as { error: string }).error, "EMBEDDING_UNAVAILABLE");
  assert.equal(calls.length, 0);
});

after(async () => {
  while (openServers.length > 0) {
    const close = openServers.pop();
    await close?.();
  }
});
