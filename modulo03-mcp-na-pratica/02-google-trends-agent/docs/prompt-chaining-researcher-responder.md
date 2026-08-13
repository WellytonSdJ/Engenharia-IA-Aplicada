# Prompt chaining: `researcher` → `responder`

## A cadeia

`src/graph/graph.ts` define o grafo inteiro:

```typescript
export function buildTrendsGraph(openRouterService: OpenRouterService) {
  return new StateGraph(GraphAnnotation)
    .addNode('researcher', createResearcherNode(openRouterService))
    .addNode('responder', createResponderNode(openRouterService))
    .addEdge(START, 'researcher')
    .addEdge('researcher', 'responder')
    .addEdge('responder', END)
    .compile();
}
```

Duas bordas fixas, sem `addConditionalEdges`, sem loop de qualidade (diferente, por exemplo, de um pipeline plan→draft→review com repetição até atingir uma nota mínima). Isso é **prompt chaining** no sentido mais direto: a saída de um nó vira a entrada do próximo, sempre na mesma ordem, sem o grafo decidir "se" ou "quantas vezes" cada nó roda.

## O estado

```typescript
// src/graph/state.ts
export const GraphAnnotation = z.object({
    messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta),
    trendsData: z.string().optional(),   // texto de saída do researcher, consumido pelo responder
    question: z.string().optional(),     // pergunta original do usuário, propagada para o responder
    keywords: z.array(z.string()).optional(),   // declarado no estado, mas nenhum nó escreve nesse campo hoje
});
```

`messages` usa `withLangGraph` + `MessagesZodMeta` para herdar o reducer padrão de mensagens do LangGraph (append, não substituição). Os outros três campos são atualizados por merge simples de objeto parcial — o comportamento padrão do `StateGraph` quando o campo não declara um reducer customizado.

## `researcher`: extrai contexto e busca dados

```typescript
// src/graph/nodes/researcherNode.ts
const userQuestion = state.messages.at(-1)!.content as string;

const result = await openRouterService.generateStructured(
    getKeywordsSystemPrompt(),
    userQuestion,
);

return {
    trendsData: JSON.stringify(result.data, null, 2),
    question: userQuestion,
};
```

Como visto em [`structured-output-zod.md`](./structured-output-zod.md), essa chamada roda em modo tools (nenhum schema é passado). O `result.data` é o texto final que o agente produziu depois de (possivelmente) chamar `google_trends` — não é o JSON bruto do SerpAPI. Esse texto é then re-serializado com `JSON.stringify` e guardado em `trendsData` como uma string (com aspas de string dentro, já que a entrada já era texto).

## `responder`: transforma dado em recomendação

```typescript
// src/graph/nodes/responderNode.ts
const userQuestion = state.question!;

const { data } = await openRouterService.generateStructured(
    getResponderSystemPrompt(),
    getResponderUserPrompt(userQuestion, state.trendsData ?? ''),
);

return { messages: [new AIMessage(data as string)] };
```

O `responder` recebe a pergunta original e o `trendsData` produzido pelo `researcher`, monta um único prompt de usuário concatenando os dois (`getResponderUserPrompt`), e pede ao LLM uma análise final em português. O resultado vira a última `AIMessage` do array `messages` — é essa mensagem que o `server.ts` devolve como resposta HTTP (`response.messages.at(-1)?.text`).

## Tratamento de erro

Os dois nós têm `try/catch` e, em caso de falha, retornam uma mensagem de erro amigável em vez de propagar a exceção — o grafo sempre chega ao `END` com alguma resposta, mesmo que degradada. Isso é diferente de interromper o grafo (como o `01-multiple-mcp-tools` faz quando `intentNode` falha): aqui não há verificação de estado de erro entre os nós, o `responder` roda de qualquer forma, mesmo que `trendsData` contenha a string de erro do `researcher`.
