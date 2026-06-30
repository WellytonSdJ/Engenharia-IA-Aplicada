# LangGraph

## O que é

LangGraph é um framework para construir fluxos de execução com LLMs onde o estado persiste entre etapas. O nome vem de "grafo" — você define nós (funções) e arestas (transições) que formam um grafo dirigido.

A diferença central em relação a chains simples do LangChain: o LangGraph tem estado mutável compartilhado entre todos os nós, e esse estado pode ser persistido entre invocações (memória de sessão).

---

## Conceitos fundamentais

### State

O state é o objeto central que flui pelo grafo. Todo nó lê do state e retorna um `Partial<State>` com o que quer atualizar.

```ts
// graph.ts
const ChatStateAnnotation = z.object({
  messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta),
  userContext: z.string().optional(),
  extractedPreferences: z.any().optional(),
  needsSummarization: z.boolean().optional(),
  conversationSummary: z.any().optional(),
  userId: z.string().optional(),
})
```

O state é **imutável por convenção** — nós não modificam o objeto recebido, retornam um partial com as mudanças. O LangGraph faz o merge via reducers.

### Reducers

Reducer é a função que define como um campo do state é atualizado quando um nó retorna um novo valor.

Para a maioria dos campos, o reducer padrão é simples substituição. Para `messages`, o LangGraph usa um reducer especial configurado via `MessagesZodMeta`:

```ts
messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta)
```

Com esse reducer:

- Um nó retornando `messages: [new AIMessage("oi")]` **adiciona** ao array existente
- Um nó retornando `messages: [new RemoveMessage({ id: "xyz" })]` **remove** a mensagem com aquele id
- Sem esse reducer, o nó substituiria o array inteiro

Isso é o que permite o `summarizationNode` deletar mensagens antigas sem sobrescrever as novas:

```ts
// summarizationNode.ts
const deleteMessages = state.messages
  .slice(0, -2)
  .map(m => new RemoveMessage({ id: m.id as string }))

return { messages: deleteMessages } // remove tudo exceto as 2 últimas
```

### Nodes

Nó é uma função assíncrona que recebe o state atual e retorna um partial com as mudanças:

```ts
// Assinatura de um nó
async (state: GraphState, runtime?: Runtime): Promise<Partial<GraphState>> => {
  // lê do state, faz algo, retorna o que mudou
  return { messages: [new AIMessage("resposta")] }
}
```

No projeto, os nós são criados via factory functions que recebem dependências por injeção:

```ts
export function createChatNode(llmClient, preferencesService) {
  return async (state, runtime) => { ... }
}
```

Isso evita acoplamento direto e facilita testes — você pode passar mocks.

### Edges

Aresta define para onde o grafo vai após cada nó. Há dois tipos:

**Estáticas** — sempre vão para o mesmo nó:

```ts
.addEdge(START, 'chat')       // entrada sempre vai pro chat
.addEdge('summarize', END)    // summarize sempre encerra
```

**Condicionais** — a função de roteamento decide o destino com base no state:

```ts
.addConditionalEdges('chat', routeAfterChat, {
  savePreferences: 'savePreferences',
  summarize: 'summarize',
  end: END,
})
```

A função de roteamento recebe o state e retorna uma string que mapeia para um destino:

```ts
// edgeConditions.ts
export const routeAfterChat = (state: GraphState): string =>
  state.extractedPreferences ? 'savePreferences' :
  state.needsSummarization ? 'summarize' : 'end'
```

### Fluxo do projeto

```text
START
  └─> chat
        ├─ (tem extractedPreferences?) ──> savePreferences
        │                                       ├─ (needsSummarization?) ──> summarize ──> END
        │                                       └─ END
        ├─ (needsSummarization e sem prefs?) ──> summarize ──> END
        └─ END
```

---

## Checkpointer

O checkpointer é responsável por salvar um snapshot completo do state após cada nó. Isso é o que dá "memória" ao grafo entre invocações.

```ts
// memoryService.ts
const checkpointer = PostgresSaver.fromConnString(dbUri)
await checkpointer.setup() // cria tabelas no Postgres

// graph.ts
graph.compile({ checkpointer: memoryService.checkpointer })
```

Quando você invoca o grafo com o mesmo `thread_id`, ele restaura o state do último checkpoint antes de executar:

```ts
// index.ts
const config = {
  configurable: { thread_id: threadId }, // identifica a sessão
  context: { userId: actualUserId }
}
await graph.invoke({ messages: [...] }, config)
```

O `thread_id` é o identificador da conversa. Threads diferentes = históricos isolados. No projeto, o `thread_id` inclui timestamp (`userId-Date.now()`), então cada execução começa uma nova thread — a persistência entre sessões vem do SQLite, não do checkpointer.

### Por que Postgres e não SQLite para o checkpointer?

O LangGraph salva checkpoints em formato binário serializado com metadados de versionamento. O `PostgresSaver` e `PostgresStore` são implementações oficiais que lidam com isso. Usar SQLite para o checkpointer exigiria uma implementação customizada da interface `BaseCheckpointSaver`.

---

## Store

O store é um key-value persistido separado do checkpointer. Enquanto o checkpointer salva snapshots do state por thread, o store é um namespace global disponível para os nós via `runtime`.

```ts
const store = PostgresStore.fromConnString(dbUri)
graph.compile({ store: memoryService.store })
```

No projeto atual, o store está configurado mas não é usado diretamente pelos nós — as preferências são salvas no SQLite via `PreferencesService`. O store seria usado para dados que precisam ser acessados por múltiplas threads do mesmo usuário via API do LangGraph.

---

## Runtime e Context

`Runtime` é o objeto injetado nos nós pelo LangGraph durante a execução. Ele carrega `context`, que é passado na configuração do invoke:

```ts
// index.ts
const config = { context: { userId: actualUserId } }

// chatNode.ts / summarizationNode.ts
const userId = String(runtime?.context?.userId || state.userId || 'unknown')
```

O `context` é útil quando o mesmo grafo é usado por múltiplos usuários simultâneos — cada invocação tem seu próprio contexto sem interferir no state compartilhado da thread.

---

## LangSmith — observabilidade do grafo

LangSmith é a plataforma de observabilidade do LangChain. Sem ele, você é "cego" ao que acontece dentro do grafo — qual nó executou, qual prompt foi enviado, qual foi a resposta, onde deu erro.

Com LangSmith você vê, para cada invocação:

- Quais nós foram executados e em que ordem
- O state antes e depois de cada nó
- O prompt exato enviado ao modelo
- A resposta recebida e se passou na validação Zod
- Latência por nó e custo de tokens

Para habilitar, basta definir as variáveis de ambiente (configuração opcional no projeto):

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=song-highlights
```

---

## Referências no projeto

| Conceito | Arquivo |
| --- | --- |
| Definição do state e grafo | [src/graph/graph.ts](../src/graph/graph.ts) |
| Nós do grafo | [src/graph/nodes/](../src/graph/nodes/) |
| Edge conditions | [src/graph/nodes/edgeConditions.ts](../src/graph/nodes/edgeConditions.ts) |
| Checkpointer + Store | [src/services/memoryService.ts](../src/services/memoryService.ts) |
| Compilação e configuração | [src/graph/factory.ts](../src/graph/factory.ts) |
| Invoke com thread_id e context | [src/index.ts](../src/index.ts) |
