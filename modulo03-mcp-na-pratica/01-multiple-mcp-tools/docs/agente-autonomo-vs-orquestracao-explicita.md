# Agente Autônomo vs. Orquestração Explícita

## O que muda em relação ao módulo 02

Em [`03-medical-appointment`](../../../modulo02-integracao-apis-llms/03-medical-appointment/docs/prompt-chaining.md) e [`06-rag-neo4j-students`](../../../modulo02-integracao-apis-llms/06-rag-neo4j-students/docs/langgraph-pipeline.md), cada etapa do processamento era **um nó separado do grafo**, escrito explicitamente no código: `identifyIntent → schedule → message`, ou `queryPlanner → cypherGenerator → cypherValidator → ...`. O código decidia a sequência; o LLM só preenchia o conteúdo de cada etapa.

Este projeto inverte parte dessa responsabilidade. O grafo tem só **dois nós**:

```typescript
// src/graph/graph.ts
return new StateGraph(GraphAnnotation)
  .addNode('intentParser', intentNode(openRouterService))
  .addNode('agent', agentNode(openRouterService))

  .addEdge(START, 'intentParser')
  .addConditionalEdges('intentParser', (state: GraphState) =>
    state.error ? END : 'agent'
  )
  .addEdge('agent', END)
  .compile();
```

Só existem duas paradas: extrair a intenção e depois **entregar tudo para um agente autônomo resolver sozinho**. Não existe nó `convertCSV`, nó `insertMongo`, nó `queryMongo`, nó `writeReport`. Essas cinco etapas inteiras acontecem *dentro* de uma única chamada a `agentNode`.

## Onde a decisão de "qual passo agora" migrou para

Ela saiu do grafo (código) e foi para o loop de tool calling do `createAgent`, guiado por um system prompt com passos numerados:

```typescript
// src/prompts/v1/agentNode.ts
export const getSystemPrompt = () => `
You are a data processing agent. You have access to these tools:
- csv_to_json: converts a CSV string to JSON
- filesystem tools (read_file, write_file, etc.): read and write files on disk
- MongoDB tools: insert documents, run queries on a MongoDB database

When given an intent, fileContent, and fileName, you MUST follow this exact sequence of steps.
Do NOT stop after the first tool call. Complete ALL steps before giving a final answer.

Step 0: Delete all user collections in MongoDB.
Step 1: If the fileContent is CSV (or fileName ends in .csv), call csv_to_json to convert it to a JSON array.
Step 2: If the intent mentions saving or exporting JSON to a path, use write_file to save the JSON to that path.
Step 3: Insert the JSON records as documents into MongoDB. Choose a collection name based on the fileName or intent context.
Step 4: Query MongoDB to answer the analytical question described in the intent.
Step 5: Use write_file to save the final report (your answer) as a .txt file inside the ./reports/ directory.
`.trim();
```

Repare que isso **não é um grafo determinístico**: é uma instrução em linguagem natural para um modelo que decide, a cada resposta, se chama mais uma tool ou se já pode parar. O `createAgent` roda esse loop de "pensar → decidir tool → executar → observar resultado → pensar de novo" internamente — o código não sabe, e não precisa saber, quantas idas e vindas o modelo vai fazer.

```
Prompt chaining (03-medical-appointment):     Agente autônomo (este projeto):

identifyIntent (LLM)                          intentParser (LLM)
     │                                              │
     ▼                                              ▼
schedule (código determinístico)              agent (LLM decide TUDO daqui pra frente)
     │                                              │
     ▼                                          ┌───┴────────────────────────┐
message (LLM)                                   │ loop interno do createAgent │
                                                 │ tool? → executa → observa   │
                                                 │ tool? → executa → observa   │
                                                 │ ... até decidir que terminou│
                                                 └─────────────────────────────┘
```

## Por que separar em dois modos dentro do mesmo serviço

O `OpenRouterService` tem um único método, `generateStructured`, que se comporta de duas formas dependendo se você passa um `schema`:

```typescript
// src/services/openRouterService.ts
async generateStructured<T>(
    systemPrompt: string,
    userPrompt: string,
    schema?: z.ZodSchema<T>,
): Promise<{ data?: T | string; }> {
    const agentConfig = schema
        ? { responseFormat: providerStrategy(schema), tools: [] }   // modo 1: extração estruturada, sem tools
        : { tools: await this.#getTools() };                        // modo 2: agente com ferramentas, sem schema

    const agent = createAgent({ ...agentConfig, model: this.llmClient });
    // ...
}
```

- **Com `schema`** (usado pelo `intentNode`): o agente não tem tools (`tools: []`), só extrai campos estruturados via `providerStrategy` — o mesmo mecanismo já documentado em [`03-medical-appointment/docs/structured-output.md`](../../../modulo02-integracao-apis-llms/03-medical-appointment/docs/structured-output.md). Não repetimos essa explicação aqui.
- **Sem `schema`** (usado pelo `agentNode`): o agente recebe as tools reais (MCP + `csv_to_json`) e nenhum `responseFormat` — ele é livre para decidir chamar quantas tools quiser, na ordem que quiser, antes de responder em texto livre.

O mesmo `createAgent` do LangChain serve para os dois casos — a diferença de comportamento vem inteiramente de quais argumentos você passa (`tools` vs. `responseFormat`), não de duas APIs diferentes.

## O que isso ganha e o que isso custa

**Ganha:**
- Menos código de orquestração — não é preciso escrever um nó por etapa nem um roteamento condicional para cada ramificação possível
- Flexibilidade: se o LLM perceber que o `fileContent` já é JSON, ele pula o `csv_to_json` sozinho (o próprio prompt já prevê isso: "If the fileContent is already JSON, skip Step 1")
- Um único ponto de observação (o `agentNode`) para um pipeline inteiro

**Custa:**
- Perda de determinismo: nada garante, em código, que o modelo vai seguir os 6 passos na ordem certa — a garantia é textual ("You MUST follow this exact sequence... Never stop early"), não estrutural
- Depuração mais difícil: se o passo 3 falhar, você não tem um nó `insertMongoNode` isolado para testar — precisa inspecionar o histórico de tool calls dentro de uma única execução do agente (veja [observabilidade-agent-loop.md](./observabilidade-agent-loop.md))
- Custo de tokens maior: cada tool call e seu resultado entram de volta no contexto da conversa, então um pipeline de 6 passos gera várias idas e vindas ao modelo dentro de uma única chamada a `agent.invoke`

Esse é exatamente o trade-off que o material do curso descreve como "dar autonomia ao modelo em vez de controlar rigidamente cada passo no código" — este projeto é a demonstração prática dele, em contraste direto com o pipeline explícito visto no módulo 02.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Grafo de 2 nós | [src/graph/graph.ts](../src/graph/graph.ts) | Só `intentParser` e `agent`; sem nós de execução |
| Prompt com passos numerados | [src/prompts/v1/agentNode.ts](../src/prompts/v1/agentNode.ts) | Step 0 a Step 5, condições de "skip" |
| Nó do agente | [src/graph/nodes/agentNode.ts](../src/graph/nodes/agentNode.ts) | Chama `generateStructured` sem schema |
| Dois modos no mesmo serviço | [src/services/openRouterService.ts](../src/services/openRouterService.ts) | `agentConfig` ternário: `responseFormat` vs `tools` |
| Extração de intenção (modo 1) | [src/graph/nodes/intentNode.ts](../src/graph/nodes/intentNode.ts) e [src/prompts/v1/identifyIntent.ts](../src/prompts/v1/identifyIntent.ts) | `IntentSchema` com `intent` livre (string) + `fileContent`/`fileName`/`fileType` |
| Evidência do pipeline rodando | [reports/](../reports/) | Arquivos `.txt` gerados pelo próprio agente via `write_file` no Step 5 |
