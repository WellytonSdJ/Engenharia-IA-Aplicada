# Memória semântica — o vector store "na mão" com SQLite

> Este doc cobre a **spec 009** ([`009-semantic-memory`](../specs/009-semantic-memory/spec.md)): o `SqliteMemoryStore` ([`src/models/memory-store.ts`](../src/models/memory-store.ts)), a rota `POST /memories` e a injeção das memórias no `/chat`. Antes, leia [01-embeddings-e-similaridade.md](./01-embeddings-e-similaridade.md): aqui assumimos que você já sabe o que é embedding e cosseno.
>
> **Em 30 segundos**
>
> - Cada fato vira uma linha na tabela `memories`: texto + vetor (BLOB) + dono (`user_id`).
> - `remember` grava, mas **não duplica** (similaridade > 0,92 = mesmo fato).
> - `recall` devolve **até 3** fatos com similaridade **≥ 0,3**.
> - `forget` apaga, **só do próprio usuário**.
> - O `/chat` agora exige `userId` e coloca as memórias no prompt, num bloco `Relevant memories:`.

---

## O que é: os dois tipos de memória do agente

O OpsPilot agora tem **duas memórias diferentes**, e confundir as duas é o erro número 1 desta unidade:

| | Histórico de conversa (spec 008) | Memória semântica (spec 009) |
| --- | --- | --- |
| **Analogia** | O caderno de anotações *desta* reunião | O que você sabe sobre o colega há anos |
| **Escopo** | `conversationId` | `userId` |
| **Vida útil** | Enquanto a conversa existir | Entre conversas, para sempre (até `forget`) |
| **Como busca** | As 12 últimas mensagens, por ordem | Os 3 fatos mais parecidos com a pergunta, por significado |
| **Onde mora** | [`src/models/conversation-store.ts`](../src/models/conversation-store.ts) | [`src/models/memory-store.ts`](../src/models/memory-store.ts) |

Na literatura de agentes, a primeira é **memória de curto prazo** (working memory) e a segunda é **memória de longo prazo** (semântica). O agente recebe as duas no mesmo prompt, e cada uma responde a uma pergunta diferente:

- histórico: "o que acabamos de conversar?"
- memória semântica: "o que eu sei sobre esta pessoa?"

> ✅ **Checkpoint**: conversa = curto prazo, por `conversationId`. Memória semântica = longo prazo, por `userId`.

---

## O "vector store" deste projeto

Um **vector store** é qualquer lugar que guarda vetores e responde "quais são os mais parecidos com este?". Existem bancos dedicados (pgvector, Pinecone, Qdrant), mas o OpsPilot faz isso com o que já tem: **SQLite + um loop em TypeScript**.

```typescript
// src/models/memory-store.ts
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
```

Mesmo estilo do `SqliteOpsStore` e do `ConversationStore` do projeto 02: `node:sqlite`, DDL idempotente, prepared statements, mesmo arquivo `OPSPILOT_DB`. A novidade é a coluna `embedding BLOB`.

### Guardando um vetor num BLOB

SQLite não tem tipo "vetor". Então guardamos os 384 floats como **bytes crus**: 384 × 4 bytes = 1536 bytes por fato.

```typescript
// src/models/memory-store.ts
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
```

A cópia em `bufferToEmbedding` parece desperdício, mas evita um `RangeError` chato: um `Float32Array` só pode começar num byte múltiplo de 4, e o SQLite não promete isso.

### Busca por força bruta (e por que tudo bem)

Não existe índice vetorial. O `recall` lê **todas** as memórias do usuário e calcula o cosseno com cada uma. Parece ingênuo, mas faça a conta: 100 fatos × 384 multiplicações = 38.400 operações. O Node faz isso em microssegundos. Um índice ANN (approximate nearest neighbor) só começa a compensar com centenas de milhares de vetores.

> ✅ **Checkpoint**: vector store = guardar vetor + achar os mais parecidos. Aqui: BLOB no SQLite + loop com `dot`.

---

## As três operações

### `remember` — guardar sem duplicar

```typescript
// src/models/memory-store.ts
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
```

O dedup é **semântico**, não textual. Se a similaridade passar de **0,92**, o fato é considerado "o mesmo com outras palavras" e não é gravado. Quem chamou recebe o id do fato que já existia, com `stored: false`.

Por que isso importa? Na spec 010 o agente vai aprender sozinho, turno após turno. Sem dedup, "priorize o checkout" dito 5 vezes viraria 5 linhas, e o top-3 do recall ficaria entupido com a mesma informação.

### `recall` — os 3 mais relevantes, com piso

```typescript
// src/models/memory-store.ts
    // Ordem importa: pontua tudo → filtra pelo piso → ordena → corta no top-k.
    // Filtrar antes de cortar garante que o top-k só contenha fatos relevantes.
    return rows
      .map((row) => ({ id: row.id, fact: row.fact, score: dot(queryEmbedding, row.embedding) }))
      .filter((memory) => memory.score >= RECALL_MIN_SCORE)
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((memory) => RecalledMemorySchema.parse(memory));
```

Dois parâmetros com papéis diferentes:

| Parâmetro | Valor | Pergunta que responde |
| --- | --- | --- |
| `RECALL_TOP_K` | 3 | "Quantos fatos cabem no prompt sem poluir?" |
| `RECALL_MIN_SCORE` | 0,3 | "A partir de quanto um fato é relevante e não ruído?" |

**Analogia**: o top-k é o tamanho da mochila, o piso é o filtro de qualidade. Sem o piso, a mochila sempre vai cheia, mesmo que seja de coisas inúteis. Um usuário com 3 memórias sobre "resumos curtos" receberia essas 3 numa pergunta sobre alertas do checkout.

Um detalhe esperto antes desse código:

```typescript
// src/models/memory-store.ts
    // Usuário sem memórias: nem gera o embedding da consulta. Assim o primeiro /chat de quem
    // nunca guardou nada não paga o carregamento do modelo.
    if (rows.length === 0) {
      return [];
    }
```

### `forget` — apagar, só o que é seu

```typescript
// src/models/memory-store.ts
    // `AND user_id = ?` é o isolamento: mesmo conhecendo o id, ninguém apaga memória de outro usuário.
    const result = this.deleteMemory.run(memoryId, owner);
    return Number(result.changes) === 1;
```

O isolamento por usuário está em **todas** as consultas (`WHERE user_id = ?`). A memória de uma pessoa nunca aparece para outra.

> ✅ **Checkpoint**: dedup > 0,92 (estrito), recall ≥ 0,3 (inclusivo), top-3, tudo filtrado por `user_id`.

---

## A memória entrando no `/chat`

### `userId` agora é obrigatório

```typescript
// src/controllers/chat-controller.ts
export const ChatRequestSchema = z.object({
  message: z.string().trim().min(1),
  // Spec 009: obrigatório. Escopa a memória semântica; não há autenticação, o cliente declara quem é.
  userId: z.string().trim().min(1),
```

Isso é uma **mudança de contrato**: um `curl` do projeto 02 sem `userId` agora recebe 400. Não há login: o cliente declara quem é. Em produção, o `userId` viria de um token de autenticação. Aqui é estudo, então confiamos no corpo.

### A montagem do prompt

O `runChat` (que você viu nascer na spec 004 e ganhar histórico na 008) agora faz recall antes de chamar a estratégia:

```typescript
// src/services/chat.ts
  const history = await deps.conversationStore.lastMessages(conversation.id, MAX_HISTORY_MESSAGES);
  // Recall com a mensagem atual como consulta: busca por significado, não por palavra igual.
  const recalled = await deps.memoryStore.recall(request.userId, request.message);
  const input = composeInput(request.message, history, recalled);
```

E monta o texto nesta ordem:

```typescript
// src/services/chat.ts
// As estratégias recebem uma única string. Montamos nesta ordem: o que o agente sabe sobre o
// usuário (memória semântica, entre sessões) → o fio desta conversa (histórico, spec 008) →
// a mensagem atual. Sem memória e sem histórico, a mensagem segue intacta.
function composeInput(message: string, history: Message[], recalled: RecalledMemory[]): string {
```

O resultado que a estratégia recebe (é exatamente o que o teste de `chat.test.ts` confere):

```text
Relevant memories:
- checkout primeiro

Conversation so far:
User: pergunta anterior

User: e agora?
```

As estratégias (ReAct, Plan-and-Execute, Reflection) **não mudaram de assinatura**. Elas continuam recebendo uma string, só que mais rica. É o mesmo truque da spec 008 com o histórico.

### Ensinando o agente a ler o bloco

O LLM precisa saber que "Relevant memories:" é contexto, não um pedido. Para isso nasceu [`src/agents/system-prompt.ts`](../src/agents/system-prompt.ts):

```typescript
// src/agents/system-prompt.ts
export const MEMORY_GUIDANCE =
  'Se a mensagem vier com "Relevant memories:", trate esses fatos como preferências do plantonista (ex.: ordem de prioridade), não como pedidos novos, e não repita o bloco na resposta.';
```

- O **ReAct** usa o `OPSPILOT_SYSTEM_PROMPT` completo, que além da memória padroniza o formato da resposta (**Resumo → Achados → Próximos passos**).
- O **Plan-and-Execute** recebe só o `MEMORY_GUIDANCE` no prompt do planner. Sem isso, o planner poderia transformar "checkout primeiro" num passo do plano. O executor dele não usa `createReactAgent` (veja [05-plan-and-execute.md](../../01-padroes-de-raciocinio-e-execucao/docs/05-plan-and-execute.md)), então não há onde encaixar o system prompt inteiro.

A resposta 200 ganha `metrics.recalledMemories`, irmã da `historyMessages` da spec 008.

### `POST /memories` — gravar na mão

```typescript
// src/controllers/memory-controller.ts
        const result = await memoryStore.remember(userId, fact);
        res.status(result.stored ? 201 : 200).json({ ...result, userId, fact });
```

| Status | Quando |
| --- | --- |
| 201 | fato novo gravado |
| 200 | já existia um fato quase idêntico; devolve o id dele (`stored: false`) |
| 400 | `userId` ou `fact` vazio/ausente |
| 503 | modelo de embeddings indisponível |

`GET /memories` e `DELETE /memories` ficaram de fora, como no material de referência. Esquecer acontece pela tool `forget_preference` (ver [04-forget-preference.md](./04-forget-preference.md)).

> ✅ **Checkpoint**: `userId` obrigatório → recall da mensagem → bloco no topo do prompt → `recalledMemories` na resposta.

---

## Por que esses números (0,92 e 0,3)?

São os valores da referência do curso, e fazem sentido para o MiniLM:

```text
1.0 ┤ idêntico
0.92┤ ─── dedup: acima disso é "o mesmo fato" (ex.: só mudou a pontuação)
    │
    │     zona de "assunto relacionado"
    │
0.3 ┤ ─── piso do recall: abaixo disso é ruído
0.0 ┤ nada a ver
```

O teste semântico real ([`memory-store.semantic-test.ts`](../src/models/memory-store.semantic-test.ts)) confirma os dois lados: "Always prioritize the checkout service" e a mesma frase com ponto final passam de 0,92; um fato sobre alertas de payments não passa.

Não são números mágicos universais. Com outro modelo, ou com textos só em português, valeria calibrar.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Tabela `memories` e BLOB | [`src/models/memory-store.ts`](../src/models/memory-store.ts) | DDL, `embeddingToBuffer`, `bufferToEmbedding` |
| Limiares | [`src/models/memory-store.ts`](../src/models/memory-store.ts) | `DEDUP_THRESHOLD`, `RECALL_MIN_SCORE`, `RECALL_TOP_K` |
| Isolamento por usuário | [`src/models/memory-store.ts`](../src/models/memory-store.ts) | `WHERE user_id = ?` e `AND user_id = ?` |
| Schemas de retorno | [`src/models/domain.ts`](../src/models/domain.ts) | `RecalledMemorySchema`, `RememberResultSchema` |
| Montagem do prompt | [`src/services/chat.ts`](../src/services/chat.ts) | `formatMemoriesForPrompt`, `composeInput`, `recalledMemories` |
| `userId` obrigatório | [`src/controllers/chat-controller.ts`](../src/controllers/chat-controller.ts) | `ChatRequestSchema` |
| Rota de gravação | [`src/controllers/memory-controller.ts`](../src/controllers/memory-controller.ts) | 201 × 200 |
| System prompt | [`src/agents/system-prompt.ts`](../src/agents/system-prompt.ts) | `MEMORY_GUIDANCE`, `OPSPILOT_SYSTEM_PROMPT` |

---

⬅️ Anterior: [01 · Embeddings e similaridade](./01-embeddings-e-similaridade.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [03 · Refletor de aprendizado](./03-refletor-de-aprendizado.md)**
