import { EmbeddingError } from "./errors.js";

// all-MiniLM-L6-v2 transforma qualquer texto num vetor de 384 números. Textos com significado
// parecido viram vetores que apontam para a mesma direção — é isso que permite a busca semântica.
export const EMBEDDING_DIM = 384;
export const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

/** Contrato mínimo que o MemoryStore precisa: texto → vetor L2-normalizado (norma 1). */
export interface Embedder {
  embed(text: string): Promise<Float32Array>;
}

type FeatureExtractionPipeline = (
  text: string,
  options: { pooling: "mean"; normalize: true },
) => Promise<{ data: Float32Array | number[] }>;

let pipelinePromise: Promise<FeatureExtractionPipeline> | null = null;

async function loadPipeline(): Promise<FeatureExtractionPipeline> {
  try {
    // Import dinâmico: o pacote (e o runtime ONNX) só carrega quando alguém realmente pede um
    // embedding. Testes com FakeEmbedder e as rotas que não usam memória nunca pagam esse custo.
    const { pipeline } = await import("@huggingface/transformers");
    return (await pipeline("feature-extraction", MODEL_ID)) as unknown as FeatureExtractionPipeline;
  } catch (cause) {
    // Zera a promise para a próxima chamada tentar de novo (ex.: rede voltou) em vez de
    // ficar presa para sempre numa promise rejeitada.
    pipelinePromise = null;
    throw new EmbeddingError(`Failed to load embedding model ${MODEL_ID}.`, { cause });
  }
}

// Lazy singleton: o modelo (~23 MB, baixado na 1ª execução e depois lido do cache local) é
// carregado uma única vez por processo. Guardar a *promise* (e não o resultado) evita que duas
// requisições simultâneas disparem dois carregamentos.
function getPipeline(): Promise<FeatureExtractionPipeline> {
  pipelinePromise ??= loadPipeline();
  return pipelinePromise;
}

class TransformersEmbedder implements Embedder {
  async embed(text: string): Promise<Float32Array> {
    const extractor = await getPipeline();
    try {
      // pooling "mean": o modelo gera um vetor por token; a média vira um vetor da frase inteira.
      // normalize: true deixa a norma em 1, então produto escalar == similaridade de cosseno.
      const output = await extractor(text, { pooling: "mean", normalize: true });
      const data = Float32Array.from(output.data);
      if (data.length !== EMBEDDING_DIM) {
        throw new EmbeddingError(`Unexpected embedding dimension ${data.length}; expected ${EMBEDDING_DIM}.`);
      }
      return data;
    } catch (cause) {
      if (cause instanceof EmbeddingError) {
        throw cause;
      }
      throw new EmbeddingError("Failed to embed text.", { cause });
    }
  }
}

let defaultEmbedder: Embedder | null = null;

/** Embedder padrão (all-MiniLM-L6-v2 local). Criar a instância não carrega o modelo. */
export function getDefaultEmbedder(): Embedder {
  defaultEmbedder ??= new TransformersEmbedder();
  return defaultEmbedder;
}
