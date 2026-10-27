import assert from "node:assert/strict";
import test from "node:test";
import { EMBEDDING_DIM, type Embedder } from "../services/embeddings.js";
import { InvalidMemoryInputError } from "../services/errors.js";
import { FakeEmbedder } from "../services/fake-embedder.js";
import { bufferToEmbedding, dot, embeddingToBuffer, SqliteMemoryStore } from "./memory-store.js";

/** Vetor com componente `main` no eixo 0 e `side` num eixo próprio: cosseno com o eixo 0 ≈ main. */
function vectorWith(main: number, sideAxis: number, side: number): number[] {
  return Array.from({ length: EMBEDDING_DIM }, (_, index) => {
    if (index === 0) return main;
    if (index === sideAxis) return side;
    return 0;
  });
}

test("BLOB round-trip preserves the 384-d embedding", () => {
  const original = Float32Array.from({ length: EMBEDDING_DIM }, (_, index) => index / EMBEDDING_DIM);
  assert.deepEqual(bufferToEmbedding(embeddingToBuffer(original)), original);
});

test("remember stores a new fact and recall returns it with its score", async () => {
  const embedder = new FakeEmbedder().setAxis("checkout tem prioridade", 0);
  const store = new SqliteMemoryStore(":memory:", embedder);

  const saved = await store.remember("ana", "checkout tem prioridade");
  const recalled = await store.recall("ana", "checkout tem prioridade");

  assert.equal(saved.stored, true);
  assert.match(saved.id, /^memory-/);
  assert.deepEqual(recalled, [{ id: saved.id, fact: "checkout tem prioridade", score: 1 }]);
});

test("a near-duplicate fact (similarity > 0.92) is not stored twice and returns the existing id", async () => {
  const embedder = new FakeEmbedder()
    .set("prefiro resumos curtos", vectorWith(1, 1, 0.01))
    .set("gosto de resumos curtos", vectorWith(1, 1, 0.05));
  assert.ok(dot(await embedder.embed("prefiro resumos curtos"), await embedder.embed("gosto de resumos curtos")) > 0.92);
  const store = new SqliteMemoryStore(":memory:", embedder);

  const first = await store.remember("ana", "prefiro resumos curtos");
  const second = await store.remember("ana", "gosto de resumos curtos");

  assert.deepEqual(second, { id: first.id, stored: false });
  embedder.setAxis("resumos?", 0);
  assert.equal((await store.recall("ana", "resumos?")).length, 1);
});

test("dedup is per user: the same fact for another user is stored", async () => {
  const embedder = new FakeEmbedder().setAxis("checkout primeiro", 0);
  const store = new SqliteMemoryStore(":memory:", embedder);

  await store.remember("ana", "checkout primeiro");
  const other = await store.remember("bia", "checkout primeiro");

  assert.equal(other.stored, true);
});

test("recall returns at most 3 facts, all with score >= 0.3, ordered by score desc", async () => {
  const embedder = new FakeEmbedder()
    .setAxis("consulta", 0)
    .set("alto", vectorWith(1, 1, 0))
    .set("medio", vectorWith(0.8, 2, 0.6))
    .set("baixo-mas-ok", vectorWith(0.5, 3, 0.866))
    .set("tambem-ok", vectorWith(0.4, 4, 0.917))
    .set("abaixo-do-piso", vectorWith(0.1, 5, 0.995));
  const store = new SqliteMemoryStore(":memory:", embedder);
  for (const fact of ["abaixo-do-piso", "tambem-ok", "baixo-mas-ok", "medio", "alto"]) {
    await store.remember("ana", fact);
  }

  const recalled = await store.recall("ana", "consulta");

  assert.deepEqual(recalled.map((memory) => memory.fact), ["alto", "medio", "baixo-mas-ok"]);
  assert.ok(recalled.every((memory) => memory.score >= 0.3));
});

// Testamos logo acima e logo abaixo do piso (e não exatamente 0.3): Float32 arredonda, e um
// "0.3" pode virar 0.29999998 — exatamente o tipo de teste que fica instável sem motivo.
test("the 0.3 floor keeps a fact just above it and drops one just below it", async () => {
  const embedder = new FakeEmbedder()
    .setAxis("consulta", 0)
    .set("acima", vectorWith(0.31, 1, Math.sqrt(1 - 0.31 ** 2)))
    .set("abaixo", vectorWith(0.29, 2, Math.sqrt(1 - 0.29 ** 2)));
  const store = new SqliteMemoryStore(":memory:", embedder);
  await store.remember("ana", "acima");
  await store.remember("ana", "abaixo");

  const recalled = await store.recall("ana", "consulta");

  assert.deepEqual(recalled.map((memory) => memory.fact), ["acima"]);
});

test("memories are isolated by userId", async () => {
  const embedder = new FakeEmbedder().setAxis("fato da ana", 1).setAxis("fato da bia", 1);
  const store = new SqliteMemoryStore(":memory:", embedder);
  await store.remember("ana", "fato da ana");

  assert.deepEqual(await store.recall("bia", "fato da bia"), []);
  assert.equal((await store.recall("ana", "fato da bia")).length, 1);
});

test("recall for a user without memories does not even call the embedder", async () => {
  let calls = 0;
  const countingEmbedder: Embedder = {
    async embed() {
      calls += 1;
      return new Float32Array(EMBEDDING_DIM);
    },
  };
  const store = new SqliteMemoryStore(":memory:", countingEmbedder);

  assert.deepEqual(await store.recall("ninguem", "qualquer coisa"), []);
  assert.equal(calls, 0);
});

test("forget removes the user's memory; unknown id or another user's id is a safe no-op", async () => {
  const embedder = new FakeEmbedder().setAxis("esquecer isto", 7);
  const store = new SqliteMemoryStore(":memory:", embedder);
  const { id } = await store.remember("ana", "esquecer isto");

  assert.equal(await store.forget("bia", id), false);
  assert.equal((await store.recall("ana", "esquecer isto")).length, 1);
  assert.equal(await store.forget("ana", id), true);
  assert.deepEqual(await store.recall("ana", "esquecer isto"), []);
  assert.equal(await store.forget("ana", id), false);
});

test("empty fact, query or userId is rejected with InvalidMemoryInputError", async () => {
  const store = new SqliteMemoryStore(":memory:", new FakeEmbedder());

  await assert.rejects(store.remember("ana", "   "), InvalidMemoryInputError);
  await assert.rejects(store.remember("", "fato"), InvalidMemoryInputError);
  await assert.rejects(store.recall("ana", ""), InvalidMemoryInputError);
});
