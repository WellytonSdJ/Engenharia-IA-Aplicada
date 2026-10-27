import assert from "node:assert/strict";
import test from "node:test";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { RecalledMemory, RememberResult } from "../models/domain.js";
import { SqliteMemoryStore, type MemoryStore } from "../models/memory-store.js";
import { FakeEmbedder } from "./fake-embedder.js";
import {
  createLLMLearningReflector,
  LEARNING_REFLECTOR_PROMPT,
  scheduleLearning,
  type LearningReflection,
} from "./learning-reflector.js";

function spyMemoryStore(): MemoryStore & { remembered: Array<{ userId: string; fact: string }> } {
  const remembered: Array<{ userId: string; fact: string }> = [];
  return {
    remembered,
    async remember(userId, fact): Promise<RememberResult> {
      remembered.push({ userId, fact });
      return { id: "memory-spy", stored: true };
    },
    async recall(): Promise<RecalledMemory[]> {
      return [];
    },
    async forget() {
      return false;
    },
  };
}

/** Modelo fake que só implementa withStructuredOutput — o único método que o refletor usa. */
function structuredModel(reply: LearningReflection | Error, seen: BaseLanguageModelInput[] = []): BaseChatModel {
  return {
    withStructuredOutput() {
      return {
        async invoke(input: BaseLanguageModelInput) {
          seen.push(input);
          if (reply instanceof Error) throw reply;
          return reply;
        },
      };
    },
  } as unknown as BaseChatModel;
}

test("a durable preference is remembered once, with the fact trimmed", async () => {
  const memoryStore = spyMemoryStore();

  await scheduleLearning({
    reflector: async () => ({ hasLearning: true, fact: "  O plantonista prioriza o checkout  " }),
    memoryStore,
    userId: "ana",
    userMessage: "sempre priorize o checkout",
  });

  assert.deepEqual(memoryStore.remembered, [{ userId: "ana", fact: "O plantonista prioriza o checkout" }]);
});

test("punctual requests, secrets and hasLearning:true with an empty fact are not remembered", async () => {
  const cases: Array<{ message: string; reflection: LearningReflection }> = [
    { message: "liste os alertas agora", reflection: { hasLearning: false, fact: "" } },
    { message: "minha senha do grafana é hunter2", reflection: { hasLearning: false, fact: "" } },
    { message: "resposta contraditória do modelo", reflection: { hasLearning: true, fact: "   " } },
  ];
  for (const { message, reflection } of cases) {
    const memoryStore = spyMemoryStore();
    await scheduleLearning({ reflector: async () => reflection, memoryStore, userId: "ana", userMessage: message });
    assert.deepEqual(memoryStore.remembered, [], message);
  }
});

test("a failing reflector or a failing remember never throws", async () => {
  const originalError = console.error;
  console.error = () => {};
  try {
    await scheduleLearning({
      reflector: async () => {
        throw new Error("llm down");
      },
      memoryStore: spyMemoryStore(),
      userId: "ana",
      userMessage: "oi",
    });
    const brokenStore: MemoryStore = {
      ...spyMemoryStore(),
      async remember() {
        throw new Error("db down");
      },
    };
    await scheduleLearning({
      reflector: async () => ({ hasLearning: true, fact: "fato durável" }),
      memoryStore: brokenStore,
      userId: "ana",
      userMessage: "oi",
    });
  } finally {
    console.error = originalError;
  }
});

test("the learned fact can be recalled later through the real SqliteMemoryStore", async () => {
  const embedder = new FakeEmbedder()
    .setAxis("O plantonista prioriza o checkout", 4)
    .setAxis("qual serviço vem primeiro?", 4);
  const memoryStore = new SqliteMemoryStore(":memory:", embedder);

  await scheduleLearning({
    reflector: async () => ({ hasLearning: true, fact: "O plantonista prioriza o checkout" }),
    memoryStore,
    userId: "ana",
    userMessage: "sempre priorize o checkout",
  });

  const recalled = await memoryStore.recall("ana", "qual serviço vem primeiro?");
  assert.deepEqual(recalled.map((memory) => memory.fact), ["O plantonista prioriza o checkout"]);
});

test("createLLMLearningReflector sends the reflector prompt plus the user message and parses the reply", async () => {
  const seen: BaseLanguageModelInput[] = [];
  const reflector = createLLMLearningReflector(() =>
    structuredModel({ hasLearning: true, fact: "Prefere resumos curtos" }, seen),
  );

  const reflection = await reflector("prefiro resumos curtos");

  assert.deepEqual(reflection, { hasLearning: true, fact: "Prefere resumos curtos" });
  const messages = seen[0] as Array<{ content: string }>;
  assert.equal(messages[0].content, LEARNING_REFLECTOR_PROMPT);
  assert.equal(messages[1].content, "prefiro resumos curtos");
});

test("createLLMLearningReflector falls back to no learning when the model fails", async () => {
  const reflector = createLLMLearningReflector(() => structuredModel(new Error("rate limited")));

  assert.deepEqual(await reflector("sempre priorize o checkout"), { hasLearning: false, fact: "" });
});

test("the reflector prompt covers durable facts, punctual requests and secrets", () => {
  assert.match(LEARNING_REFLECTOR_PROMPT, /DURÁVEIS/);
  assert.match(LEARNING_REFLECTOR_PROMPT, /PONTUAIS/);
  assert.match(LEARNING_REFLECTOR_PROMPT, /SEGREDOS/);
  assert.match(LEARNING_REFLECTOR_PROMPT, /liste alertas/);
});
