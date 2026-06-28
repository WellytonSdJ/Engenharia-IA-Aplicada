# LangGraph — Pipeline de RAG

## A arquitetura como grafo

Este projeto usa o LangGraph para orquestrar um pipeline de 6 nós. A escolha do LangGraph sobre uma chain simples vem de duas necessidades:

1. **Roteamento condicional** — a mesma pergunta pode seguir caminhos muito diferentes (uma query simples ou 3 queries sequenciais com correção no meio)
2. **Estado compartilhado** — dados produzidos por um nó (query gerada, resultados, erros) precisam estar disponíveis para nós subsequentes sem passagem explícita de parâmetros

---

## O grafo completo

```
START
  │
  ▼
extractQuestion ──── erro? ──► END
  │
  ▼
queryPlanner
  │
  ▼
cypherGenerator ◄────────────────────────────────────────────────┐
  │                                                              │
  ▼                                                              │
cypherExecutor ──── erro + tentativas restantes? ──► cypherCorrection ──┘
  │
  ├── multi-step + mais steps? ──► cypherGenerator (próximo step)
  │
  ▼
analyticalResponse
  │
  ▼
END
```

---

## O estado do grafo (GraphState)

O `GraphState` é o objeto central que passa de nó em nó. Todo nó lê campos do state e retorna um `Partial<GraphState>` com o que mudou.

```typescript
// graph.ts
type GraphState = {
  // entrada
  messages: BaseMessage[]        // histórico da conversa (entrada da API)
  question?: string              // pergunta extraída pelo extractQuestionNode

  // geração e execução de query
  query?: string                 // query Cypher gerada
  originalQuery?: string         // query antes da correção (para log)
  dbResults?: any[]              // resultados retornados pelo Neo4j

  // self-correction
  correctionAttempts?: number    // quantas vezes o cypherCorrection já rodou (máx 1)
  validationError?: string       // erro retornado pelo EXPLAIN ou pela execução
  needsCorrection?: boolean      // flag que ativa o roteamento para cypherCorrection

  // multi-step (perguntas complexas)
  isMultiStep?: boolean          // se a pergunta foi decomposta em sub-perguntas
  subQuestions?: string[]        // sub-perguntas geradas pelo queryPlanner
  currentStep?: number           // índice da sub-pergunta sendo processada agora
  subQueries?: string[]          // queries Cypher geradas por step
  subResults?: any[][]           // resultados acumulados por step

  // resposta final
  answer?: string                // resposta analítica em prosa
  followUpQuestions?: string[]   // 2-3 perguntas sugeridas
  error?: string                 // erro geral, se houver
}
```

---

## Os nós

### extractQuestionNode

Extrai o texto da última mensagem do array `messages`. Se a mensagem estiver vazia, seta `state.error` e o grafo vai direto para END — nenhum outro nó é executado.

```typescript
// extractQuestionNode.ts
const lastMessage = state.messages.at(-1)
if (!lastMessage?.content) return { error: 'No question found' }
return { question: String(lastMessage.content) }
```

### queryPlannerNode

Faz uma chamada LLM para classificar a complexidade da pergunta.

- **Simples**: uma única query Cypher resolve
- **Complexa**: precisa ser decomposta em até 3 sub-perguntas independentes

```typescript
// Saída via Zod (QueryAnalysisSchema)
{
  complexity: 'simple' | 'complex',
  requiresDecomposition: boolean,
  subQuestions: string[],   // vazio se simples
  reasoning: string,
}
```

Se `requiresDecomposition` for `true`, seta `isMultiStep: true`, `subQuestions`, e `currentStep: 0`.

### cypherGeneratorNode

O nó mais complexo do pipeline. Recebe:
- O schema real do Neo4j (buscado em tempo real)
- As regras de negócio do domínio (`salesContext`)
- A pergunta atual (ou a sub-pergunta do step atual, se multi-step)
- 5 exemplos de queries no prompt

Retorna a query Cypher gerada (schema `CypherQuerySchema`).

Para queries multi-step, acumula no array `subQueries[]` em vez de sobrescrever `query`.

### cypherExecutorNode

Valida e executa a query. Dois passos:

1. **Validação**: roda `EXPLAIN <query>` no Neo4j. Se falhar → sintaxe inválida antes de executar
2. **Execução**: roda a query real e coleta resultados

Se houver erro:
- `correctionAttempts < 1` → seta `needsCorrection: true` e roda `cypherCorrection`
- `correctionAttempts >= 1` → desiste e seta `error` (não entra em loop infinito)

Para multi-step: acumula em `subResults[]` e incrementa `currentStep`.

### cypherCorrectionNode

Recebe a query que falhou + a mensagem de erro do Neo4j + a pergunta original.
Faz uma chamada LLM com contexto de debugging:

- Erros de agregação → usar `WITH` para separar agrupamento
- Variáveis fora de escopo → passar por `WITH`
- `ORDER BY` com NULL → adicionar `NULLS LAST`
- Propriedades incorretas → verificar schema e usar alias correto

Retorna `correctedQuery` e `explanation`. Incrementa `correctionAttempts`.

### analyticalResponseNode

Último nó — gera a resposta em prosa. Três caminhos:

| Situação | Comportamento |
| --- | --- |
| `state.error` existe | Explica o erro de forma amigável + sugere alternativas |
| `dbResults` está vazio | Diz que não encontrou dados + sugere razões |
| Resultados presentes | Gera análise em prosa com cálculos, percentuais, comparações |

Para queries multi-step: recebe todos os steps (pergunta + query + resultados de cada step) e sintetiza em uma narrativa coerente.

Sempre retorna `answer` (prosa) + `followUpQuestions` (array com 2-3 sugestões).

---

## As edges condicionais

```typescript
// Após extractQuestion:
if (state.error) → END
else             → queryPlanner

// Após cypherExecutor:
if (needsCorrection && correctionAttempts < 1) → cypherCorrection
if (isMultiStep && currentStep < subQuestions.length) → cypherGenerator
else → analyticalResponse

// Após cypherCorrection:
→ cypherExecutor (sempre — tenta executar a query corrigida)
```

As edges condicionais são funções que leem o state e retornam uma string:

```typescript
// graph.ts
graph.addConditionalEdges('cypherExecutor', (state) => {
  if (state.needsCorrection && (state.correctionAttempts ?? 0) < config.maxCorrectionAttempts)
    return 'correction'
  if (state.isMultiStep && (state.currentStep ?? 0) < (state.subQuestions?.length ?? 0))
    return 'generator'
  return 'response'
}, {
  correction: 'cypherCorrection',
  generator: 'cypherGenerator',
  response: 'analyticalResponse',
})
```

---

## Injeção de dependências via factory

Os nós não instanciam serviços diretamente. O `factory.ts` cria as instâncias e as injeta:

```typescript
// factory.ts
const neo4jService = new Neo4jService()
const openrouterService = new OpenRouterService()
const graph = createGraph(neo4jService, openrouterService)
```

```typescript
// graph.ts
export function createGraph(neo4jService: Neo4jService, llmService: OpenRouterService) {
  const nodes = {
    cypherGenerator: createCypherGeneratorNode(neo4jService, llmService),
    // ...
  }
}
```

Isso facilita testes — você pode passar mocks de `Neo4jService` sem tocar no banco real.

---

## Referências no projeto

| Conceito | Arquivo |
| --- | --- |
| Definição do state e montagem do grafo | [src/graph/graph.ts](../src/graph/graph.ts) |
| Factory de serviços | [src/graph/factory.ts](../src/graph/factory.ts) |
| Nós do grafo | [src/graph/nodes/](../src/graph/nodes/) |
| Limites (correção, sub-perguntas) | [src/config.ts](../src/config.ts) |
