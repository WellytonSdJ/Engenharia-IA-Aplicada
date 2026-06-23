# Song Highlights — Chatbot Musical com Memória

Chatbot de recomendação musical via CLI construído com **LangGraph**, **LangChain** e **OpenRouter**. O bot conversa naturalmente, extrai preferências musicais em tempo real com saída estruturada (Zod), persiste o histórico em PostgreSQL via LangGraph e armazena preferências estruturadas em SQLite.

## O que o projeto faz

- Inicia a conversa carregando preferências já salvas do banco (nome, gêneros, bandas favoritas)
- Extrai informações do usuário em tempo real usando saída estruturada com Zod
- Salva preferências extraídas em SQLite mesclando com dados anteriores
- Sumariza e poda o histórico automaticamente a cada N mensagens, mantendo apenas as últimas 2
- Persiste checkpoints da conversa em PostgreSQL para retomar sessões

## Dois mecanismos de memória

O projeto usa dois tipos de persistência com responsabilidades distintas:

| Mecanismo              | Tecnologia                   | Responsabilidade                                                                             |
| ---------------------- | ---------------------------- | -------------------------------------------------------------------------------------------- |
| **Checkpointer**       | PostgreSQL (`PostgresSaver`) | Histórico de mensagens da conversa; permite retomar o thread entre sessões                   |
| **Store**              | PostgreSQL (`PostgresStore`) | Memória de longo prazo gerenciada pelo LangGraph                                             |
| **PreferencesService** | SQLite / Knex                | Preferências estruturadas (nome, idade, gêneros, bandas) — legíveis como texto pelo chatNode |

## Arquitetura — Grafo de Estados (LangGraph)

O fluxo conversacional é um `StateGraph` com três nós e roteamento condicional:

```
START
  │
  ▼
chat ──── extractedPreferences? ──► savePreferences ──── needsSummarization? ──► summarize ──► END
  │                                                  └──────────────────────────────────────► END
  │
  ├── needsSummarization? ──► summarize ──► END
  └─────────────────────────────────────► END
```

### Nós

| Nó                | O que faz                                                                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chat`            | Carrega contexto do usuário no SQLite, monta prompts, chama o LLM com saída estruturada (Zod), retorna resposta + preferências extraídas + flag de sumarização |
| `savePreferences` | Persiste as preferências extraídas no SQLite via `mergePreferences` (acumula gêneros e bandas com deduplicação)                                                |
| `summarize`       | Chama o LLM para sumarizar a conversa, salva o sumário no SQLite via `storeSummary` e remove mensagens antigas com `RemoveMessage` (mantém as últimas 2)       |

### Roteamento condicional (`edgeConditions.ts`)

```typescript
// Após chat:
extractedPreferences → savePreferences
needsSummarization   → summarize
(nenhum)             → END

// Após savePreferences:
needsSummarization → summarize
(nenhum)           → END
```

### Gatilho de sumarização

A sumarização é ativada quando `messages.length >= config.maxMessagesToSummary` (atualmente `2`). Após sumarizar, o nó remove todas as mensagens exceto as últimas 2 usando `RemoveMessage` do LangGraph.

### Estado do grafo

```typescript
{
  messages: BaseMessage[]         // histórico (gerenciado pelo LangGraph + checkpointer)
  userContext?: string            // preferências carregadas do SQLite no início
  extractedPreferences?: object   // preferências extraídas nesta mensagem
  needsSummarization?: boolean    // flag que dispara o nó summarize
  conversationSummary?: object    // último sumário gerado
  userId?: string                 // identificador da sessão/usuário
}
```

## Estrutura do projeto

```
src/
├── config.ts                          # Configuração: API key, modelos, banco, maxMessagesToSummary
├── index.ts                           # CLI interativo com readline e flag --user
├── graph/
│   ├── graph.ts                       # StateGraph compilado com checkpointer e store
│   ├── factory.ts                     # Instancia serviços e monta o grafo
│   └── nodes/
│       ├── chatNode.ts                # Nó de conversação com extração de preferências
│       ├── summarizationNode.ts       # Nó de sumarização e poda do histórico
│       ├── savePreferencesNode.ts     # Nó de persistência das preferências no SQLite
│       └── edgeConditions.ts          # Funções de roteamento condicional
├── services/
│   ├── memoryService.ts               # PostgresSaver + PostgresStore (LangGraph)
│   ├── openrouterService.ts           # ChatOpenAI com OpenRouter + saída estruturada
│   └── preferencesService.ts          # CRUD de preferências no SQLite com Knex
└── prompts/
    └── v1/
        ├── chatResponse.ts            # Schema Zod + prompts do nó chat
        └── summarization.ts           # Schema Zod + prompts do nó summarize
tests/
└── chat.e2e.test.ts                   # Testes E2E com node:test
```

## Saída estruturada com Zod

O LLM não retorna texto livre — é instruído a responder em JSON validado por schemas Zod:

```typescript
// Nó chat — prompts/v1/chatResponse.ts
ChatResponseSchema = z.object({
  message: z.string(),                    // resposta conversacional ao usuário
  preferences: UserPreferencesSchema,     // dados extraídos desta mensagem
  shouldSavePreferences: z.boolean(),     // se deve acionar savePreferences
})

UserPreferencesSchema = z.object({
  name, age, favoriteGenres, favoriteBands,
  mood, listeningContext, additionalInfo
})

// Nó summarize — prompts/v1/summarization.ts
SummarySchema = z.object({
  name, age, favoriteGenres, favoriteBands,
  keyPreferences: z.string(),    // sumário em 2-4 frases
  importantContext?: z.string()
})
```

## Dependência entre serviços (injeção)

Os nós recebem os serviços via injeção de dependência. O `factory.ts` é responsável por instanciar tudo:

```typescript
// factory.ts
export async function buildGraph(dbPath = "./preferences.db") {
  const llmClient = new OpenRouterService(config);
  const memoryService = await createMemoryService(); // PostgreSQL
  const preferencesService = new PreferencesService(dbPath); // SQLite

  const graph = buildChatGraph(llmClient, preferencesService, memoryService);
  return { graph, preferencesService };
}
```

O grafo é compilado com `checkpointer` e `store` do LangGraph:

```typescript
graph.compile({
  checkpointer: memoryService.checkpointer,
  store: memoryService.store,
});
```

## Requisitos

### Software

| Requisito          | Versão mínima | Observação                                            |
| ------------------ | ------------- | ----------------------------------------------------- |
| **Node.js**        | 24.10.0+      | Obrigatório — usa `--experimental-strip-types` nativo |
| **npm**            | 10+           | Incluído no Node.js                                   |
| **Docker**         | 20+           | Para subir o PostgreSQL via `docker-compose`          |
| **Docker Compose** | 2+            | Incluído no Docker Desktop                            |

> O projeto declara `"engines": { "node": ">=24.10.0" }` no `package.json`. Versões anteriores não são suportadas.

### Contas e APIs

| Serviço        | Obrigatório | Como obter                                                                                              |
| -------------- | ----------- | ------------------------------------------------------------------------------------------------------- |
| **OpenRouter** | Sim         | Crie uma conta em [openrouter.ai](https://openrouter.ai/) e gere uma API key                            |
| **LangSmith**  | Não         | Opcional — habilita rastreamento de chamadas LLM em [smith.langchain.com](https://smith.langchain.com/) |

### Variáveis de ambiente

Crie `.env` copiando `.env.example`:

```bash
cp .env.example .env
```

| Variável                  | Obrigatória | Descrição                                                         |
| ------------------------- | ----------- | ----------------------------------------------------------------- |
| `OPENROUTER_API_KEY`      | Sim         | Chave de acesso à API OpenRouter                                  |
| `OPENROUTER_HTTP_REFERER` | Não         | Referer enviado nas requisições (padrão: `http://localhost:3000`) |
| `OPENROUTER_X_TITLE`      | Não         | Nome do app enviado ao OpenRouter                                 |
| `LANGSMITH_API_KEY`       | Não         | Habilita rastreamento com LangSmith                               |
| `LANGCHAIN_TRACING_V2`    | Não         | Ativar rastreamento (`true`/`false`)                              |
| `LANGCHAIN_PROJECT`       | Não         | Nome do projeto no LangSmith                                      |

### Banco de dados

O projeto usa **dois bancos** com responsabilidades distintas:

| Banco          | Uso                                                     | Como provisionar                           |
| -------------- | ------------------------------------------------------- | ------------------------------------------ |
| **PostgreSQL** | Checkpointer + Store LangGraph (histórico de mensagens) | `npm run docker:up` (sobe via Docker)      |
| **SQLite**     | Preferências estruturadas do usuário                    | Criado automaticamente em `preferences.db` |

A URI do PostgreSQL está em `src/config.ts`:

```
postgresql://postgres:mysecretpassword@localhost:5432/song_recommender
```

## Configuração

Crie um `.env` na raiz do projeto:

```env
OPENROUTER_API_KEY=sua-chave-aqui
```

Parâmetros em `src/config.ts`:

| Parâmetro              | Valor padrão                          | Descrição                                 |
| ---------------------- | ------------------------------------- | ----------------------------------------- |
| `models`               | `arcee-ai/trinity-large-preview:free` | Modelo LLM usado                          |
| `provider.sort.by`     | `throughput`                          | Critério de roteamento no OpenRouter      |
| `temperature`          | `0.7`                                 | Criatividade das respostas                |
| `maxMessagesToSummary` | `2`                                   | Nº de mensagens que dispara a sumarização |
| `memory.dbUri`         | PostgreSQL local                      | URI de conexão para checkpointer e store  |

## Execução

```bash
npm install

# Subir PostgreSQL via Docker
npm run docker:up

# Iniciar conversa como usuário anônimo
node --experimental-strip-types --env-file .env src/index.ts

# Iniciar como usuário identificado (retoma sessão e preferências)
npm run chat:wellyton
npm run chat:ana
```

## Testes

```bash
npm test           # executa todos os testes E2E
npm run test:watch # modo watch
```

Os testes usam o runner nativo do Node.js (`node:test`) e verificam extração de preferências, múltiplas trocas de mensagem, persistência e manutenção do histórico.

## LangGraph Studio

O projeto inclui `langgraph.json` para visualizar e debugar o grafo no LangGraph Studio:

```bash
npm run langgraph:serve
```
