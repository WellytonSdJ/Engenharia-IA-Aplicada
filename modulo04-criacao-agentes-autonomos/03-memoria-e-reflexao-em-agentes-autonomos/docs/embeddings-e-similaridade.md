# Embeddings e similaridade — transformar texto em "endereço de significado"

> Este doc cobre a base da **spec 009** ([`009-semantic-memory`](../specs/009-semantic-memory/spec.md)): o que é um embedding, como o comparamos e como o modelo roda localmente em [`src/services/embeddings.ts`](../src/services/embeddings.ts). O uso disso como memória fica em [memoria-semantica.md](./memoria-semantica.md).
>
> **Em 30 segundos**
>
> - Um **embedding** é uma lista de números (aqui, 384) que representa o *significado* de um texto.
> - Textos parecidos viram vetores que **apontam para a mesma direção**.
> - Comparamos direções com o **cosseno**. Com vetores normalizados, o cosseno é só um **produto escalar**.
> - O modelo (`all-MiniLM-L6-v2`) roda **dentro do seu processo Node**: sem API key e sem custo.

---

## O que é

Você já viu "RAG com vetor" citado de passagem em [`rag-com-grafos.md`](../../../modulo02-integracao-apis-llms/06-rag-neo4j-students/docs/rag-com-grafos.md), na tabela que compara RAG vetorial com RAG em grafo: "Embedding + similaridade cosseno". Aqui é onde a gente finalmente abre essa caixa.

**Analogia: o GPS do significado.** Imagine que cada frase recebe uma coordenada num mapa gigante. Mas o mapa não é de cidades, é de *assuntos*. Frases sobre "pagamento lento" ficam num bairro, frases sobre "resumos curtos" ficam em outro. Duas frases podem não ter **nenhuma palavra em comum** e mesmo assim morar na mesma rua:

```text
"The checkout payment queue has elevated latency during peak hours"
"Are card charges slow at rush time?"
          ↑ zero palavras iguais, mas o mesmo bairro do mapa
```

Esse mapa tem 384 dimensões em vez de 2, mas a ideia é a mesma. O **modelo de embeddings** é quem sabe dar a coordenada de qualquer frase.

### Por que isso importa para um agente

Até a spec 008, toda "memória" do OpsPilot era **busca exata**:

| Onde | Como busca | Problema |
| --- | --- | --- |
| `consultar_runbook` (spec 005) | `WHERE service_id = ?` | precisa saber o nome exato do serviço |
| Histórico (spec 008) | `WHERE conversation_id = ?` | só lembra a conversa atual |

Com embeddings, a busca passa a ser **por significado**. O plantonista pode perguntar de um jeito diferente do que foi guardado e ainda assim o agente acha.

> ✅ **Checkpoint**: embedding = coordenada de significado. Busca semântica = achar as coordenadas mais próximas.

---

## Como comparar dois embeddings: cosseno e produto escalar

"Próximo" aqui quer dizer **mesma direção**, não mesma distância. A medida padrão é a **similaridade de cosseno**:

```text
          mesma direção  →  cosseno ≈ 1     (mesmo assunto)
          perpendicular  →  cosseno ≈ 0     (nada a ver)
          direção oposta →  cosseno ≈ -1    (raro com texto)
```

A fórmula completa divide pelo tamanho (norma) de cada vetor. Mas existe um atalho: se **todo vetor tiver norma 1** (normalizado), o cosseno vira só o **produto escalar** (multiplica posição a posição e soma). É exatamente o que o projeto faz:

```typescript
// src/models/memory-store.ts
/** Produto escalar. Com vetores de norma 1 ele é exatamente a similaridade de cosseno. */
export function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    sum += a[index] * b[index];
  }
  return sum;
}
```

Quem garante a norma 1 é a opção `normalize: true` na hora de gerar o embedding (próxima seção). **Sem ela, o `dot` deixa de ser cosseno**, e os limiares 0,92 e 0,3 perdem o sentido.

> ✅ **Checkpoint**: vetor normalizado + produto escalar = cosseno. Uma linha de matemática, zero bibliotecas.

---

## Como está sendo usado neste projeto

### O modelo: `all-MiniLM-L6-v2`, rodando local

```typescript
// src/services/embeddings.ts
export const EMBEDDING_DIM = 384;
export const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

/** Contrato mínimo que o MemoryStore precisa: texto → vetor L2-normalizado (norma 1). */
export interface Embedder {
  embed(text: string): Promise<Float32Array>;
}
```

- **`all-MiniLM-L6-v2`**: um modelo pequeno (~23 MB) feito só para gerar embeddings de frases. Ele não conversa nem gera texto.
- **`Xenova/...`**: a versão convertida para ONNX, o formato que o `@huggingface/transformers` (Transformers.js) consegue rodar em Node.
- **Local**: na primeira execução o pacote baixa os pesos do Hugging Face e guarda em cache (em `node_modules/@huggingface/transformers/.cache`). Depois disso, tudo roda offline. Não existe `OPENROUTER_API_KEY` para embeddings.

A interface `Embedder` é o detalhe que torna tudo testável: o `MemoryStore` só conhece `embed(text)`. Em produção entra o modelo real; nos testes entra o `FakeEmbedder` (ver [testando-memoria.md](./testando-memoria.md)).

### `pooling: "mean"` e `normalize: true`

```typescript
// src/services/embeddings.ts
      // pooling "mean": o modelo gera um vetor por token; a média vira um vetor da frase inteira.
      // normalize: true deixa a norma em 1, então produto escalar == similaridade de cosseno.
      const output = await extractor(text, { pooling: "mean", normalize: true });
```

O modelo cospe **um vetor por token** (pedaço de palavra). Para ter um vetor só da frase, tiramos a **média** de todos (`mean pooling`). Depois normalizamos para norma 1.

```text
"checkout está lento"
   ↓ tokenização
[check] [out] [está] [lento]
   ↓ modelo
 v1     v2     v3     v4      (4 vetores de 384)
   ↓ mean pooling
        v_frase               (1 vetor de 384)
   ↓ normalize
        v_frase / |v_frase|   (norma 1)
```

### Lazy singleton: carregar uma vez, só quando precisar

Carregar o modelo leva alguns segundos. Fazer isso a cada requisição seria absurdo, e fazer na inicialização deixaria todo `npm run dev` lento, mesmo para quem nunca usa memória. A solução é o **lazy singleton**:

```typescript
// src/services/embeddings.ts
// Lazy singleton: o modelo (~23 MB, baixado na 1ª execução e depois lido do cache local) é
// carregado uma única vez por processo. Guardar a *promise* (e não o resultado) evita que duas
// requisições simultâneas disparem dois carregamentos.
function getPipeline(): Promise<FeatureExtractionPipeline> {
  pipelinePromise ??= loadPipeline();
  return pipelinePromise;
}
```

Três detalhes que valem ouro:

1. **Guarda a promise, não o resultado.** Se duas requisições chegam juntas no cold start, as duas recebem a *mesma* promise e esperam o mesmo carregamento.
2. **Import dinâmico.** `await import("@huggingface/transformers")` só acontece dentro do `loadPipeline`. Quem nunca gera embedding nunca carrega o runtime ONNX, e é por isso que o `npm test` continua rápido.
3. **Falhou? Tenta de novo depois.** Em erro, `pipelinePromise = null`. Sem isso, uma queda de rede no primeiro download deixaria o processo preso para sempre numa promise rejeitada.

```typescript
// src/services/embeddings.ts
  } catch (cause) {
    // Zera a promise para a próxima chamada tentar de novo (ex.: rede voltou) em vez de
    // ficar presa para sempre numa promise rejeitada.
    pipelinePromise = null;
    throw new EmbeddingError(`Failed to load embedding model ${MODEL_ID}.`, { cause });
  }
```

O `EmbeddingError` é um erro de domínio (padrão de [`contrato-de-estrategia-e-trace.md`](../../01-padroes-de-raciocinio-e-execucao/docs/contrato-de-estrategia-e-trace.md) e dos erros do projeto 01). Na borda HTTP ele vira **503 `EMBEDDING_UNAVAILABLE`**: é uma dependência indisponível, não um erro de quem chamou.

> ✅ **Checkpoint**: o modelo carrega uma vez, na primeira necessidade, e um cold start que falhou pode tentar de novo.

---

## Limitações e cuidados

| Cuidado | Por quê |
| --- | --- |
| **Modelo treinado em inglês** | Funciona com português, mas os scores tendem a ser menores. Os testes semânticos usam frases em inglês justamente para provar o mecanismo sem ruído. |
| **Trocar de modelo invalida o banco** | Vetores de modelos diferentes vivem em "mapas" diferentes. Se trocar o modelo, é preciso regenerar todos os embeddings guardados. |
| **384 é fixo** | `EMBEDDING_DIM` é conferido na geração e na leitura do BLOB. Um vetor de outro tamanho vira erro, não lixo silencioso. |
| **Primeira execução precisa de rede** | Para baixar os pesos. Depois, offline. |

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Contrato `Embedder` | [`src/services/embeddings.ts`](../src/services/embeddings.ts) | Só `embed(text)`, o que permite trocar real ↔ fake |
| Lazy singleton + import dinâmico | [`src/services/embeddings.ts`](../src/services/embeddings.ts) | `getPipeline`, `loadPipeline`, `pipelinePromise = null` no erro |
| Produto escalar | [`src/models/memory-store.ts`](../src/models/memory-store.ts) | `dot(a, b)` |
| Erro → 503 | [`src/controllers/chat-controller.ts`](../src/controllers/chat-controller.ts) | `EmbeddingError` em `toErrorResponse` |
| Modelo real em ação | [`src/models/memory-store.semantic-test.ts`](../src/models/memory-store.semantic-test.ts) | Vetor de 384 com norma ~1; recall sem palavras em comum |
