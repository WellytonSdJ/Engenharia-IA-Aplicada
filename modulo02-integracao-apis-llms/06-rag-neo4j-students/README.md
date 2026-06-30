# RAG Neo4j Students — Análise de Dados com Grafo e LLM

API HTTP de análise de vendas de uma academia online, construída com **LangGraph**, **LangChain**, **Neo4j** e **OpenRouter**. O sistema converte perguntas em linguagem natural em queries Cypher, executa no banco de grafos, e retorna respostas analíticas em prosa — com autocorreção de queries e decomposição automática de perguntas complexas.

## O que o projeto faz

- Recebe perguntas em linguagem natural via `POST /sales`
- Analisa se a pergunta é simples ou complexa (multi-step)
- Gera queries Cypher usando o schema do Neo4j como contexto
- Executa as queries no banco de grafos
- Corrige automaticamente queries com erro de sintaxe (self-correction)
- Retorna uma resposta analítica em prosa com perguntas de follow-up sugeridas

## Domínio

O banco contém dados de uma academia online: cursos, alunos, compras e progresso. Exemplos de perguntas que o sistema responde:

- "Quais cursos são comprados juntos com frequência?"
- "Qual é a distribuição de receita entre os cursos?"
- "Quais alunos compraram mas nunca começaram um curso?"
- "Compare a receita entre cursos com alta e baixa taxa de conclusão"

## Arquitetura — Grafo de Estados (LangGraph)

```
START
  │
  ▼
extractQuestion
  │
  ▼
queryPlanner ──── isMultiStep? ──► divide em subQuestions
  │
  ▼
cypherGenerator ◄──────────────────────────────────────────┐
  │                                                         │
  ▼                                                         │
cypherExecutor ──── erro? ──► cypherCorrection ─────────────┘
  │
  ├── isMultiStep + mais steps? ──► cypherGenerator (próximo step)
  │
  ▼
analyticalResponse
  │
  ▼
END
```

### Nós

| Nó | O que faz |
|---|---|
| `extractQuestion` | Extrai a pergunta da última mensagem do estado |
| `queryPlanner` | Analisa complexidade: simples (uma query) ou complexa (decomposição em sub-perguntas) |
| `cypherGenerator` | Gera a query Cypher usando o schema do Neo4j + contexto de negócio |
| `cypherExecutor` | Valida e executa a query; detecta erros e sinaliza necessidade de correção |
| `cypherCorrection` | Reescreve a query com base no erro retornado pelo Neo4j |
| `analyticalResponse` | Gera resposta em prosa com os resultados; trata erros e resultados vazios |

### Roteamento condicional

```typescript
// Após extractQuestion:
state.error → END
(sem erro)  → queryPlanner

// Após cypherExecutor:
needsCorrection && correctionAttempts < 1 → cypherCorrection
isMultiStep && currentStep < subQuestions.length → cypherGenerator (próximo step)
(demais casos) → analyticalResponse
```

### Estado do grafo

```typescript
{
  messages: BaseMessage[]         // histórico (entrada da API)
  question?: string               // pergunta extraída

  // Geração e execução de query
  query?: string                  // query Cypher gerada
  originalQuery?: string          // query original antes da correção
  dbResults?: any[]               // resultados retornados pelo Neo4j

  // Self-correction
  correctionAttempts?: number     // número de tentativas de correção
  validationError?: string        // erro de validação/execução
  needsCorrection?: boolean       // flag que dispara cypherCorrection

  // Multi-step (perguntas complexas)
  isMultiStep?: boolean           // se a pergunta foi decomposta
  subQuestions?: string[]         // sub-perguntas geradas pelo queryPlanner
  currentStep?: number            // índice da sub-pergunta atual
  subQueries?: string[]           // queries geradas por step
  subResults?: any[][]            // resultados acumulados por step

  // Resposta
  answer?: string                 // resposta analítica final em prosa
  followUpQuestions?: string[]    // perguntas de follow-up sugeridas
  error?: string                  // erro geral, se houver
}
```

## Estrutura do projeto

```
src/
├── config.ts                              # Configuração: API key, modelos, Neo4j, limites
├── index.ts                               # Entry point: sobe o servidor e faz uma requisição de teste
├── server.ts                              # Servidor Fastify com rota POST /sales
├── graph/
│   ├── graph.ts                           # StateGraph com todos os nós e roteamento
│   ├── factory.ts                         # Instancia serviços e monta o grafo
│   └── nodes/
│       ├── extractQuestionNode.ts         # Extrai pergunta do estado
│       ├── queryPlannerNode.ts            # Analisa complexidade e decompõe a pergunta
│       ├── cypherGeneratorNode.ts         # Gera Cypher com schema + contexto
│       ├── cypherExecutorNode.ts          # Executa e valida a query no Neo4j
│       ├── cypherCorrectionNode.ts        # Corrige query com base no erro
│       └── analyticalResponseNode.ts     # Gera resposta analítica em prosa
├── services/
│   ├── neo4jService.ts                    # Wrapper do Neo4jGraph: schema, validação, query, seed
│   └── openrouterService.ts               # ChatOpenAI com OpenRouter + saída estruturada (Zod)
└── prompts/
    └── v1/
        ├── queryAnalyzer.ts               # Schema + prompts para análise de complexidade
        ├── cypherGenerator.ts             # Schema + prompts para geração de Cypher
        ├── cypherCorrection.ts            # Schema + prompts para correção de Cypher
        ├── analyticalResponse.ts          # Schema + prompts para resposta analítica
        ├── nlpResponse.ts                 # Schema para resposta NLP (uso alternativo)
        └── salesContext.ts                # Contexto de negócio injetado nos prompts
data/
├── courses.json                           # Cursos da academia (seed)
├── seed.ts                                # Entry point do seed
└── seedHelpers.ts                         # Geração de dados fictícios com faker + inserção no Neo4j
tests/
└── sales.e2e.test.ts                      # Testes E2E: seed + requisições reais à API
```

## Saída estruturada com Zod

O LLM retorna JSON validado por schemas Zod em todos os nós:

```typescript
// queryAnalyzer.ts — análise de complexidade
QueryAnalysisSchema = z.object({
  complexity: z.enum(['simple', 'complex']),
  requiresDecomposition: z.boolean(),
  subQuestions: z.array(z.string()),
  reasoning: z.string(),
})

// cypherGenerator.ts — geração de query
CypherQuerySchema = z.object({
  query: z.string(),  // query Cypher pura, sem markdown
})

// cypherCorrection.ts — correção de query
CypherCorrectionSchema = z.object({
  correctedQuery: z.string(),
  explanation: z.string(),  // o que foi corrigido
})

// analyticalResponse.ts — resposta final
AnalyticalResponseSchema = z.object({
  answer: z.string(),                      // resposta analítica em prosa
  followUpQuestions: z.array(z.string()),  // 2-3 perguntas sugeridas
})
```

## Schema do banco (Neo4j)

```
(Student)-[:PURCHASED {status, paymentMethod, paymentDate, amount}]->(Course)
(Student)-[:PROGRESS {progress: 0-100}]->(Course)
```

O schema é carregado dinamicamente do Neo4j (`neo4jService.getSchema()`) e injetado nos prompts de geração e correção de Cypher — o LLM sempre trabalha com o schema real do banco.

## Requisitos

### Software

| Requisito | Versão mínima | Observação |
|---|---|---|
| **Node.js** | 24.10.0+ | Usa `--experimental-strip-types` nativo |
| **npm** | 10+ | Incluído no Node.js |
| **Docker** | 20+ | Para subir o Neo4j |
| **Docker Compose** | 2+ | Incluído no Docker Desktop |

### Contas e APIs

| Serviço | Obrigatório | Como obter |
|---|---|---|
| **OpenRouter** | Sim | [openrouter.ai](https://openrouter.ai/) |
| **LangSmith** | Não | Opcional — rastreamento em [smith.langchain.com](https://smith.langchain.com/) |

### Variáveis de ambiente

Crie `.env` copiando `.env.example`:

```bash
cp .env.example .env
```

| Variável | Obrigatória | Descrição |
|---|---|---|
| `OPENROUTER_API_KEY` | Sim | Chave de acesso à API OpenRouter |
| `LANGSMITH_API_KEY` | Não | Habilita rastreamento com LangSmith |
| `LANGCHAIN_TRACING_V2` | Não | Ativar rastreamento (`true`/`false`) |
| `LANGCHAIN_PROJECT` | Não | Nome do projeto no LangSmith |

### Banco de dados

| Banco | Uso | Como provisionar |
|---|---|---|
| **Neo4j** | Armazena grafos de alunos, cursos, compras e progresso | `npm run docker:infra:up` |

URI configurada em `src/config.ts`:
```
neo4j://localhost:7687  (usuário: neo4j / senha: password)
```

## Configuração

Parâmetros em `src/config.ts`:

| Parâmetro | Valor padrão | Descrição |
|---|---|---|
| `models` | `arcee-ai/trinity-large-preview:free` | Modelo LLM usado |
| `temperature` | `0.7` | Criatividade das respostas |
| `maxCorrectionAttempts` | `1` | Tentativas de autocorreção de Cypher |
| `maxSubQuestions` | `3` | Máximo de sub-perguntas na decomposição |
| `neo4j.uri` | `neo4j://localhost:7687` | URI de conexão |

## Execução

```bash
npm install

# Subir Neo4j via Docker
npm run docker:infra:up

# Popular banco com dados fictícios
npm run seed

# Iniciar servidor (porta 4000)
npm run dev
```

A requisição de teste embutida em `src/index.ts` é executada automaticamente na inicialização:

```bash
curl -X POST -H 'Content-type: application/json' \
  --data '{"question": "Which courses are commonly bought together?"}' \
  localhost:4000/sales
```

## Testes

```bash
npm test           # executa testes E2E (faz seed + requisições reais)
npm run test:e2e   # apenas testes E2E
npm run test:dev   # modo watch
```

Os testes usam o runner nativo do Node.js (`node:test`) e cobrem: listagem de cursos, queries por aluno, análise de receita, distribuição percentual, progresso, métodos de pagamento e casos de borda.

## LangGraph Studio

```bash
npm run langgraph:serve
```

## Infraestrutura Docker

```bash
npm run docker:infra:up       # sobe Neo4j
npm run docker:infra:down     # para os containers
npm run docker:infra:cleanup  # remove containers + volumes + pasta storage/
npm run docker:infra:logs     # acompanha logs em tempo real
```
