# Song Highlights — Chatbot Musical com Memória

Chatbot de recomendação musical via CLI construído com **LangGraph**, **LangChain** e **OpenRouter**. O bot conversa naturalmente com o usuário, extrai preferências musicais em tempo real usando saída estruturada (Zod) e as persiste em SQLite para enriquecer sessões futuras.

## O que o projeto faz

- Inicia a conversa saudando o usuário com base em preferências já salvas (ou pedindo que ele se apresente, se for a primeira vez)
- Extrai em tempo real informações como nome, idade, gêneros e bandas favoritas, humor e contexto de escuta
- Salva as preferências extraídas em SQLite após cada mensagem relevante
- Sumariza a conversa periodicamente para consolidar o histórico
- Roda como CLI interativo com `readline`, com suporte a múltiplos usuários via flag `--user`

## Arquitetura — Grafo de Estados (LangGraph)

O fluxo conversacional é modelado como um `StateGraph` com três nós e roteamento condicional:

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

| Nó | Responsabilidade |
|----|-----------------|
| `chat` | Gera resposta conversacional + extrai preferências do usuário usando saída estruturada (Zod) |
| `savePreferences` | Persiste as preferências extraídas no banco SQLite via `PreferencesService` |
| `summarize` | Resume a conversa e atualiza o perfil consolidado do usuário no banco |

### Roteamento condicional

- **`routeAfterChat`:** se `extractedPreferences` estiver preenchido → `savePreferences`; se `needsSummarization` estiver ativo → `summarize`; caso contrário → END
- **`routeAfterSavePreferences`:** se `needsSummarization` → `summarize`; caso contrário → END

### Estado do grafo

```typescript
{
  messages: BaseMessage[]        // histórico da conversa (gerenciado pelo LangGraph)
  userContext?: string           // preferências carregadas do banco no início da sessão
  extractedPreferences?: object  // preferências extraídas pelo nó chat
  needsSummarization?: boolean   // flag que dispara o nó de sumarização
  conversationSummary?: object   // resultado da sumarização
  userId?: string                // identificador da sessão/usuário
}
```

## Estrutura do projeto

```
src/
├── config.ts                          # Configuração: chave de API, modelos, banco de dados
├── index.ts                           # CLI interativo com readline
├── graph/
│   ├── graph.ts                       # Definição do StateGraph e compilação
│   ├── factory.ts                     # Instancia serviços e monta o grafo
│   └── nodes/
│       ├── chatNode.ts                # Nó principal de conversação
│       ├── summarizationNode.ts       # Nó de sumarização da conversa
│       ├── savePreferencesNode.ts     # Nó de persistência das preferências
│       └── edgeConditions.ts          # Funções de roteamento condicional
├── services/
│   ├── openrouterService.ts           # Cliente LLM com ChatOpenAI + OpenRouter
│   └── preferencesService.ts          # CRUD de preferências no SQLite com Knex
└── prompts/
    └── v1/
        ├── chatResponse.ts            # Schema Zod + prompts do nó chat
        └── summarization.ts           # Schema Zod + prompts do nó summarize
tests/
└── chat.e2e.test.ts                   # Testes E2E com node:test
```

## Saída estruturada com Zod

O nó `chat` não retorna texto livre — o LLM é instruído a responder em JSON validado pelos schemas Zod definidos em `prompts/v1/chatResponse.ts`:

```typescript
// Resposta do nó chat
ChatResponseSchema = z.object({
  message: z.string(),                  // resposta conversacional para o usuário
  preferences: UserPreferencesSchema,   // dados extraídos desta mensagem
  shouldSavePreferences: z.boolean(),   // se deve persistir as preferências
})

// Preferências extraídas
UserPreferencesSchema = z.object({
  name, age, favoriteGenres, favoriteBands,
  mood, listeningContext, additionalInfo
})
```

O mesmo padrão é usado no nó `summarize` com `SummarySchema`.

## Persistência — PreferencesService

Usa **Knex** com **better-sqlite3** para armazenar um perfil por usuário na tabela `user_preferences`:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `user_id` | string (unique) | Identificador da sessão |
| `name` | string | Nome do usuário |
| `age` | integer | Idade |
| `favorite_genres` | JSON | Lista de gêneros favoritos |
| `favorite_bands` | JSON | Lista de bandas/artistas favoritos |
| `key_preferences` | text | Sumário das preferências principais |
| `important_context` | text | Outros contextos relevantes |
| `updated_at` | timestamp | Última atualização |

Ao salvar, as preferências são **mescladas** com as existentes — gêneros e bandas são acumulados com deduplicação via `Set`.

## Configuração

Crie um `.env` na raiz do projeto:

```env
OPENROUTER_API_KEY=sua-chave-aqui
```

Parâmetros principais em `src/config.ts`:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `models` | `upstage/solar-pro-3:free` | Modelo LLM usado |
| `provider.sort.by` | `throughput` | Critério de roteamento no OpenRouter |
| `temperature` | `0.7` | Criatividade das respostas |

## Execução

```bash
npm install

# Iniciar conversa como usuário anônimo (novo thread a cada execução)
node --experimental-strip-types --env-file .env src/index.ts

# Iniciar como usuário identificado (retoma preferências salvas)
npm run chat:erickwendel
npm run chat:ana
```

## Testes

```bash
npm test           # executa todos os testes E2E
npm run test:watch # modo watch
```

Os testes usam o runner nativo do Node.js (`node:test`) e verificam: extração e salvamento de preferências, múltiplas trocas de mensagem, recuperação de contexto entre sessões e manutenção do histórico.

## LangGraph Studio

O projeto inclui `langgraph.json` para visualização e debug do grafo no LangGraph Studio:

```bash
npm run langgraph:serve
```
