// Testes com o modelo REAL de embeddings (all-MiniLM-L6-v2). Ficam fora do `npm test` de
// propósito: a primeira execução baixa ~23 MB do Hugging Face, e o `npm test` do projeto roda
// sem rede. Rode com `npm run test:semantic` (depois do 1º download, o modelo vem do cache local).
import assert from "node:assert/strict";
import test from "node:test";
import { EMBEDDING_DIM, getDefaultEmbedder } from "../services/embeddings.js";
import { SqliteMemoryStore } from "./memory-store.js";

const TIMEOUT = { timeout: 180_000 };

test("the default embedder returns a 384-d vector with norm ~1", TIMEOUT, async () => {
  const vector = await getDefaultEmbedder().embed("OpsPilot embedding smoke");

  assert.equal(vector.length, EMBEDDING_DIM);
  assert.ok(Math.abs(Math.hypot(...vector) - 1) < 0.01);
});

test("semantic recall finds a fact that shares no words with the question", TIMEOUT, async () => {
  const store = new SqliteMemoryStore(":memory:", getDefaultEmbedder());
  await store.remember("plantonista", "The checkout payment queue has elevated latency during peak hours");
  await store.remember("plantonista", "The on-call engineer prefers short incident summaries");

  // Nenhuma palavra em comum com o primeiro fato: "card charges" ≈ "payment", "rush time" ≈ "peak hours".
  const recalled = await store.recall("plantonista", "Are card charges slow at rush time?");

  assert.match(recalled[0]?.fact ?? "", /checkout payment queue/);
  assert.ok((recalled[0]?.score ?? 0) >= 0.3);
});

test("paraphrases cross the 0.92 dedup line only when they really say the same thing", TIMEOUT, async () => {
  const store = new SqliteMemoryStore(":memory:", getDefaultEmbedder());

  const first = await store.remember("plantonista", "Always prioritize the checkout service");
  const same = await store.remember("plantonista", "Always prioritize the checkout service.");
  const different = await store.remember("plantonista", "Payments alerts can wait until morning");

  assert.equal(same.stored, false);
  assert.equal(same.id, first.id);
  assert.equal(different.stored, true);
});
