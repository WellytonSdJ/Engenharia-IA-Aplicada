# Persistência

O projeto usa dois sistemas de persistência com propósitos distintos. Confundir os dois é um erro comum ao estudar LangGraph.

---

## Postgres — memória de sessão (LangGraph)

Gerenciado pelo `MemoryService`, usando os pacotes oficiais do LangGraph:

```ts
// memoryService.ts
const checkpointer = PostgresSaver.fromConnString(dbUri)
const store = PostgresStore.fromConnString(dbUri)

await checkpointer.setup() // cria tabela checkpoints + blobs no Postgres
await store.setup()        // cria tabela store no Postgres
```

### Checkpointer

Salva um snapshot completo do `GraphState` após cada nó executar. Funciona como um log de estados da conversa por thread.

**O que persiste:** tudo que está no state — mensagens, flags, resumo atual, preferências pendentes.

**Chave de acesso:** `thread_id` (passado em `config.configurable.thread_id`).

**Quando é usado:** toda vez que `graph.invoke()` é chamado com um `thread_id`, o checkpointer restaura o último state salvo antes de executar os nós.

```ts
// index.ts — cada sessão tem thread único com timestamp
const threadId = `${actualUserId}-${Date.now()}`
const config = { configurable: { thread_id: threadId } }
```

No projeto atual, o `thread_id` inclui timestamp, então cada execução começa uma thread nova. O checkpointer mantém o histórico dentro da sessão (entre invocações da mesma execução), não entre sessões diferentes.

### Store

Key-value persistido independente de thread. Acessível pelos nós via `runtime.store`.

**Status no projeto:** configurado e injetado no grafo, mas não utilizado diretamente pelos nós — os dados de preferências vão para o SQLite. O store estaria disponível para casos de uso como dados compartilhados entre múltiplas threads do mesmo usuário.

---

## SQLite — preferências entre sessões

Gerenciado pelo `PreferencesService` via Knex + better-sqlite3:

```ts
// preferencesService.ts
this.db = knex({
  client: 'better-sqlite3',
  connection: { filename: dbPath } // arquivo local .db
})
```

**Tabela:** `user_preferences`, com uma linha por `user_id`.

**O que persiste:** nome, idade, gêneros, bandas, contexto, resumo da última conversa.

**Chave de acesso:** `userId` (string passada pelo usuário via `--user` na CLI).

### Dois modos de escrita

**`mergePreferences`** — chamado pelo `savePreferencesNode` quando o LLM extrai preferências novas na conversa:

- Acumula gêneros e bandas via `Set` (sem duplicatas)
- Concatena contexto novo ao existente (não substitui)
- Mantém `name` e `age` existentes se a nova extração vier `null`

**`storeSummary`** — chamado pelo `summarizationNode` após sumarização:

- Sobrescreve completamente com o resumo gerado pelo LLM
- O LLM já recebeu o resumo anterior como input, então o output é um merge inteligente

### Como é usado na próxima sessão

```ts
// index.ts — antes do primeiro invoke
const userContext = await preferencesService.getBasicInfo(actualUserId)

// chatNode.ts — fallback se userContext não veio no state
const userContext = state.userContext ?? await preferencesService.getBasicInfo(userId)

// chatResponse.ts — injetado no system prompt
preferencias_previamente_armazenadas: userContext || 'Nenhuma'
```

---

## Comparação

| | Postgres (checkpointer) | SQLite (preferences) |
| --- | --- | --- |
| **Gerenciado por** | LangGraph | Knex / aplicação |
| **Escopo** | Por thread_id | Por userId |
| **Persiste entre sessões?** | Não (thread tem timestamp) | Sim |
| **Formato** | Binário serializado | Tabela relacional |
| **Lido por** | LangGraph automaticamente | Nós e index.ts explicitamente |
| **Escrito por** | LangGraph após cada nó | savePreferencesNode e summarizationNode |
| **Propósito** | Estado da conversa ativa | Conhecimento acumulado do usuário |

---

## Referências no projeto

| Conceito | Arquivo |
| --- | --- |
| Checkpointer + Store | [src/services/memoryService.ts](../src/services/memoryService.ts) |
| SQLite / preferências | [src/services/preferencesService.ts](../src/services/preferencesService.ts) |
| Injeção no grafo | [src/graph/factory.ts](../src/graph/factory.ts) |
| Leitura na inicialização | [src/index.ts](../src/index.ts) |
