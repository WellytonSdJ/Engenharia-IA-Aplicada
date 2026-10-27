import { DatabaseSync, type StatementSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { RecalledMemorySchema, RememberResultSchema, type RecalledMemory, type RememberResult } from "./domain.js";
import { EMBEDDING_DIM, getDefaultEmbedder, type Embedder } from "../services/embeddings.js";
import { InvalidMemoryInputError } from "../services/errors.js";

// Limiares fixos da spec 009. Os dois comparam a mesma coisa (similaridade de cosseno, de -1 a 1):
// - acima de 0.92 dois fatos são "o mesmo fato com outras palavras" → não grava duplicata;
// - abaixo de 0.3 o fato é ruído para aquela consulta → não entra no prompt.
export const DEDUP_THRESHOLD = 0.92;
export const RECALL_MIN_SCORE = 0.3;
export const RECALL_TOP_K = 3;

/** Contrato da memória semântica de longo prazo. Toda operação é escopada por `userId`. */
export interface MemoryStore {
  remember(userId: string, fact: string): Promise<RememberResult>;
  recall(userId: string, query: string): Promise<RecalledMemory[]>;
  forget(userId: string, id: string): Promise<boolean>;
}

interface MemoryRow {
  id: string;
  fact: string;
  embedding: Uint8Array;
}

function requireNonEmpty(label: string, value: string): string {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new InvalidMemoryInputError(`${label} must not be empty.`);
  }
  return trimmed;
}

/** Produto escalar. Com vetores de norma 1 ele é exatamente a similaridade de cosseno. */
export function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    sum += a[index] * b[index];
  }
  return sum;
}

// SQLite não tem tipo "vetor": guardamos os 384 floats como bytes crus num BLOB
// (384 × 4 bytes = 1536 bytes por fato) e reconstruímos o Float32Array na leitura.
export function embeddingToBuffer(vector: Float32Array): Buffer {
  return Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);
}

export function bufferToEmbedding(bytes: Uint8Array): Float32Array {
  // Copia antes de reinterpretar: o Float32Array exige offset múltiplo de 4, e o buffer
  // devolvido pelo SQLite não garante isso.
  const copy = Uint8Array.from(bytes);
  const vector = new Float32Array(copy.buffer);
  if (vector.length !== EMBEDDING_DIM) {
    throw new InvalidMemoryInputError(`Stored embedding has ${vector.length} dimensions; expected ${EMBEDDING_DIM}.`);
  }
  return vector;
}

/**
 * Vector store "na mão": SQLite guarda fato + vetor, e a busca é força bruta em TypeScript
 * (calcula a similaridade com cada fato do usuário). Para dezenas/centenas de fatos por usuário
 * isso é instantâneo; um vector DB dedicado (índice ANN) só compensa com volumes muito maiores.
 */
export class SqliteMemoryStore implements MemoryStore {
  private readonly db: DatabaseSync;
  private readonly insertMemory: StatementSync;
  private readonly selectByUser: StatementSync;
  private readonly deleteMemory: StatementSync;

  constructor(
    path: string = process.env.OPSPILOT_DB ?? "./data/opspilot.db",
    private readonly embedder: Embedder = getDefaultEmbedder(),
  ) {
    if (path !== ":memory:") {
      mkdirSync(dirname(path), { recursive: true });
    }
    this.db = new DatabaseSync(path);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memories (
        id         TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        fact       TEXT NOT NULL,
        embedding  BLOB NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
    `);
    this.insertMemory = this.db.prepare(
      "INSERT INTO memories (id, user_id, fact, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
    );
    this.selectByUser = this.db.prepare("SELECT id, fact, embedding FROM memories WHERE user_id = ?");
    this.deleteMemory = this.db.prepare("DELETE FROM memories WHERE id = ? AND user_id = ?");
  }

  async remember(userId: string, fact: string): Promise<RememberResult> {
    const owner = requireNonEmpty("userId", userId);
    const text = requireNonEmpty("fact", fact);
    const embedding = await this.embedder.embed(text);

    // Dedup semântico: "prefiro resumos curtos" e "gosto de resumos curtos" não viram dois fatos.
    for (const existing of this.rowsFor(owner)) {
      if (dot(embedding, existing.embedding) > DEDUP_THRESHOLD) {
        return RememberResultSchema.parse({ id: existing.id, stored: false });
      }
    }

    const id = `memory-${randomUUID()}`;
    this.insertMemory.run(id, owner, text, embeddingToBuffer(embedding), new Date().toISOString());
    return RememberResultSchema.parse({ id, stored: true });
  }

  async recall(userId: string, query: string, k = RECALL_TOP_K): Promise<RecalledMemory[]> {
    const owner = requireNonEmpty("userId", userId);
    const text = requireNonEmpty("query", query);
    const rows = this.rowsFor(owner);
    // Usuário sem memórias: nem gera o embedding da consulta. Assim o primeiro /chat de quem
    // nunca guardou nada não paga o carregamento do modelo.
    if (rows.length === 0) {
      return [];
    }
    const queryEmbedding = await this.embedder.embed(text);

    // Ordem importa: pontua tudo → filtra pelo piso → ordena → corta no top-k.
    // Filtrar antes de cortar garante que o top-k só contenha fatos relevantes.
    return rows
      .map((row) => ({ id: row.id, fact: row.fact, score: dot(queryEmbedding, row.embedding) }))
      .filter((memory) => memory.score >= RECALL_MIN_SCORE)
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((memory) => RecalledMemorySchema.parse(memory));
  }

  async forget(userId: string, id: string): Promise<boolean> {
    const owner = requireNonEmpty("userId", userId);
    const memoryId = requireNonEmpty("id", id);
    // `AND user_id = ?` é o isolamento: mesmo conhecendo o id, ninguém apaga memória de outro usuário.
    const result = this.deleteMemory.run(memoryId, owner);
    return Number(result.changes) === 1;
  }

  private rowsFor(userId: string): Array<{ id: string; fact: string; embedding: Float32Array }> {
    const rows = this.selectByUser.all(userId) as unknown as MemoryRow[];
    return rows.map((row) => ({ id: row.id, fact: row.fact, embedding: bufferToEmbedding(row.embedding) }));
  }
}
