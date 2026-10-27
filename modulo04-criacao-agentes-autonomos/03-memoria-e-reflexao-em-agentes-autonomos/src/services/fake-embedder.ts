import { EMBEDDING_DIM, type Embedder } from "./embeddings.js";

function normalize(values: Float32Array): Float32Array {
  const norm = Math.hypot(...values);
  if (norm === 0) {
    throw new Error("Fake embedding cannot be the zero vector.");
  }
  return values.map((value) => value / norm);
}

/**
 * Double determinístico de Embedder para testes: nada de download, nada de rede.
 * Você escolhe o vetor de cada texto, então controla exatamente a similaridade entre eles.
 */
export class FakeEmbedder implements Embedder {
  private readonly table = new Map<string, Float32Array>();

  set(text: string, values: number[]): this {
    if (values.length !== EMBEDDING_DIM) {
      throw new Error(`Fake vector must have ${EMBEDDING_DIM} dimensions.`);
    }
    this.table.set(text, normalize(Float32Array.from(values)));
    return this;
  }

  /** Vetor unitário num único eixo: textos no mesmo eixo têm similaridade 1, em eixos diferentes 0. */
  setAxis(text: string, axis: number): this {
    return this.set(text, Array.from({ length: EMBEDDING_DIM }, (_, index) => (index === axis ? 1 : 0)));
  }

  async embed(text: string): Promise<Float32Array> {
    const registered = this.table.get(text);
    if (registered) {
      return new Float32Array(registered);
    }
    // Texto não registrado: hash FNV-1a → um eixo "aleatório" porém estável, para que testes
    // que não ligam para memória (ex.: /chat comum) funcionem sem cadastrar vetores.
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    const vector = new Float32Array(EMBEDDING_DIM);
    vector[Math.abs(hash) % EMBEDDING_DIM] = 1;
    return vector;
  }
}
