# LangGraph — Introdução

> Este documento cobre os conceitos fundamentais do LangGraph conforme aplicados neste projeto. O projeto `04-song-highlights` usa os mesmos conceitos com checkpointer, store e sumarização — leia lá para o uso avançado.

---

## O que é

LangGraph é um framework para construir fluxos de trabalho como **grafos de estado**. Em vez de escrever código imperativo com if/else para controlar o fluxo, você define:

- **Estado** — um objeto compartilhado que todos os nós podem ler e modificar
- **Nós** — funções que recebem o estado e retornam uma versão atualizada
- **Edges** — as conexões entre nós (quem chama quem, e quando)

O framework então executa o grafo para você, garantindo que os nós rodem na ordem certa e que o estado seja atualizado corretamente a cada passo.

---

## O estado: definindo com Zod

Neste projeto, o estado é definido em `src/graph/graph.ts` usando Zod:

```typescript
// src/graph/graph.ts
import { z } from 'zod/v3'
import { MessagesZodMeta } from '@langchain/langgraph'
import { withLangGraph } from '@langchain/langgraph/zod'
import { BaseMessage } from 'langchain'

const GraphState = z.object({
    messages: withLangGraph(
        z.custom<BaseMessage[]>(),
        MessagesZodMeta
    ),
    output: z.string(),
    command: z.enum(['uppercase', 'lowercase', 'unknown'])
})

export type GraphState = z.infer<typeof GraphState>
```

Três campos:
- **`messages`** — o histórico de mensagens (HumanMessage, AIMessage). O wrapper `withLangGraph(..., MessagesZodMeta)` instrui o LangGraph a aplicar um reducer especial que acumula mensagens em vez de substituir. Ver [langchain-messages.md](./langchain-messages.md).
- **`output`** — a string que será transformada pelos nós de processamento (upper/lower)
- **`command`** — o enum que o nó de intent detecta e que a edge condicional usa para rotear

> **Por que Zod e não `Annotation.Root`?** O LangGraph oferece dois estilos de definição de estado. O `Annotation.Root` é mais antigo. A abordagem com Zod (`z.object + withLangGraph`) é mais moderna, integra validação em runtime automaticamente e é mais consistente com o resto do stack TypeScript. Os projetos seguintes usam a mesma abordagem.

---

## Os nós: funções puras

Cada nó é uma função que recebe `GraphState` e retorna uma versão atualizada. Neste projeto todos são síncronos e sem efeitos colaterais — eles apenas transformam strings:

```typescript
// src/graph/nodes/upperCaseNode.ts
import { type GraphState } from "../graph.ts";

export function upperCaseNode(state: GraphState): GraphState {
    const responseText = state.output.toUpperCase()
    return {
        ...state,
        output: responseText,
    }
}
```

Padrão obrigatório: sempre retorne `{ ...state, <campo modificado>: novoValor }`. Nunca mute o estado diretamente. O LangGraph trata o estado como imutável — cada nó produz um novo snapshot.

O nó `identifyIntent` é o único que preenche `command`, que é o campo usado pela edge condicional:

```typescript
// src/graph/nodes/identifyIntentNode.ts
export function identifyIntent(state: GraphState): GraphState {
    const input = state.messages.at(-1)?.text ?? ""
    const inputLower = input.toLowerCase()

    let command: GraphState['command'] = 'unknown'

    if (inputLower.includes('upper')) command = 'uppercase'
    else if (inputLower.includes('lower')) command = 'lowercase'

    return { ...state, command, output: input }
}
```

Observe: `state.messages.at(-1)?.text` lê a última mensagem do histórico. Isso é possível porque o estado guarda `BaseMessage[]`, não strings. Ver [langchain-messages.md](./langchain-messages.md).

---

## As edges: conectando os nós

Existem dois tipos de edge neste projeto:

**Edge fixa** (`addEdge`): sempre roteia do nó A para o nó B.

```typescript
.addEdge(START, "identifyIntent")   // sempre começa aqui
.addEdge("uppercase", "chatResponse")
.addEdge("lowercase", "chatResponse")
.addEdge("fallback", "chatResponse")
.addEdge("chatResponse", END)
```

**Edge condicional** (`addConditionalEdges`): lê o estado e decide para onde ir.

```typescript
.addConditionalEdges(
    "identifyIntent",                    // nó de origem
    (state: GraphState) => {             // função de roteamento
        switch(state.command) {
            case 'uppercase': return 'uppercase';
            case 'lowercase': return 'lowercase';
            default:          return 'fallback'
        }
    },
    {                                    // mapa de destinos possíveis
        'uppercase': 'uppercase',
        'lowercase': 'lowercase',
        'fallback':  'fallback',
    }
)
```

A função de roteamento recebe o estado completo e retorna uma string (o nome do próximo nó). O terceiro parâmetro é o mapa que declara quais destinos são possíveis — o LangGraph precisa disso para construir o grafo internamente.

---

## Montagem e compilação

Tudo é montado em `src/graph/graph.ts`:

```typescript
export function buildGraph() {
    const workflow = new StateGraph({ stateSchema: GraphState })
        .addNode("identifyIntent", identifyIntent)
        .addNode("chatResponse", chatResponseNode)
        .addNode('uppercase', upperCaseNode)
        .addNode('lowercase', lowerCaseNode)
        .addNode('fallback', fallbackNode)
        .addEdge(START, "identifyIntent")
        .addConditionalEdges("identifyIntent", ...)
        .addEdge("uppercase", "chatResponse")
        // ...
        .addEdge("chatResponse", END)

    return workflow.compile()
}
```

`workflow.compile()` valida o grafo (verifica se todos os nós referenciados existem, se não há caminhos sem saída, etc.) e retorna o objeto executável. É esse objeto que vai ser chamado com `.invoke()`.

---

## Como o grafo é executado

No `src/server.ts`:

```typescript
const graph = buildGraph()

// POST /chat
const response = await graph.invoke({
    messages: [new HumanMessage(question)]
})

return reply.send(response.output)
```

`.invoke()` recebe o estado inicial (só `messages` preenchido) e executa o grafo até o `END`. O retorno é o estado final — neste caso, `response.output` já tem o texto transformado.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Definição do estado | `src/graph/graph.ts` | `z.object`, `withLangGraph`, `MessagesZodMeta` |
| Montagem do grafo | `src/graph/graph.ts` | `StateGraph`, `.addNode`, `.addEdge`, `.addConditionalEdges`, `.compile()` |
| Nó de roteamento | `src/graph/nodes/identifyIntentNode.ts` | Como `command` é preenchido no estado |
| Nó de transformação | `src/graph/nodes/upperCaseNode.ts` | Padrão `{ ...state, campo: novoValor }` |
| Nó de fallback | `src/graph/nodes/fallbackNode.ts` | Como `AIMessage` é construído para retorno |
| Execução | `src/server.ts` | `.invoke({ messages: [...] })` e leitura de `response.output` |
| Factory LangGraph | `src/graph/factory.ts` | Exporta `graph()` para o servidor de dev do LangGraph |
