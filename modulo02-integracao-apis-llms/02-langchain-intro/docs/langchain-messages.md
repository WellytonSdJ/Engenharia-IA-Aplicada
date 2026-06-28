# LangChain Messages

## O que é

O LangChain define um sistema de tipos para representar mensagens em conversas com LLMs. Em vez de passar strings soltas, você usa objetos tipados que carregam o **papel** de quem enviou a mensagem.

Os três tipos principais:

| Tipo | Papel | Quando usar |
| --- | --- | --- |
| `HumanMessage` | Mensagem do usuário | Input de quem está usando o sistema |
| `AIMessage` | Mensagem do modelo | Output gerado pelo LLM (ou pelo sistema neste projeto) |
| `SystemMessage` | Instrução de sistema | Configuração do comportamento do LLM |
| `BaseMessage` | Tipo base | Usado como tipo genérico quando você aceita qualquer um |

Por que tipagem importa? Um LLM precisa saber *quem* disse o quê para manter coerência de contexto. Se você passar strings soltas, precisa construir o contexto manualmente. Com os tipos do LangChain, o contexto é implícito.

---

## Como está sendo usado neste projeto

Neste projeto não existe LLM de verdade — mas o sistema usa os tipos de mensagem do LangChain porque:
1. É o padrão que os projetos seguintes usam com modelos reais
2. O `MessagesZodMeta` permite que o LangGraph acumule mensagens automaticamente
3. O servidor recebe uma pergunta e precisa retornar uma resposta no mesmo formato

**Entrada (server.ts):**

```typescript
// src/server.ts
import { HumanMessage } from "langchain";

const response = await graph.invoke({
    messages: [new HumanMessage(question)]  // question vem do POST /chat
})
```

O estado inicial tem apenas `messages` preenchido com a pergunta do usuário como `HumanMessage`.

**Dentro dos nós:**

```typescript
// src/graph/nodes/identifyIntentNode.ts
const input = state.messages.at(-1)?.text ?? ""
```

`.at(-1)` pega a última mensagem (a mais recente). `.text` acessa o conteúdo textual. Isso funciona para `HumanMessage` e `AIMessage` pois ambos herdam de `BaseMessage`.

**Saída (chatResponseNode.ts):**

```typescript
// src/graph/nodes/chatResponseNode.ts
import { AIMessage } from "langchain";

const aiMessage = new AIMessage(responseText)

return {
    ...state,
    messages: [...state.messages, aiMessage]
}
```

O nó final empacota `state.output` como `AIMessage` e adiciona ao histórico. No `server.ts`, o código lê `response.output` diretamente (não `response.messages`), mas o padrão de acumular mensagens é o que os projetos seguintes vão usar para manter histórico de conversas.

---

## MessagesZodMeta e o reducer de mensagens

O campo `messages` no estado tem uma definição especial:

```typescript
// src/graph/graph.ts
import { MessagesZodMeta } from '@langchain/langgraph'
import { withLangGraph } from '@langchain/langgraph/zod'

messages: withLangGraph(
    z.custom<BaseMessage[]>(),
    MessagesZodMeta
)
```

`withLangGraph(..., MessagesZodMeta)` instrui o LangGraph a usar um **reducer** especial para o campo `messages`. Um reducer define como o valor de um campo é atualizado quando um nó retorna uma atualização.

Sem reducer: o valor antigo é substituído pelo novo.
Com `MessagesZodMeta`: as mensagens são **acumuladas** — o novo array é anexado ao existente.

Isso é importante porque sem ele, cada nó precisaria retornar o histórico completo de mensagens para não perder o contexto. Com o reducer, cada nó retorna apenas as mensagens novas e o LangGraph faz o merge automaticamente.

---

## Por que BaseMessage[] em vez de string[]

```
String[]                          BaseMessage[]
─────────────────────────────     ─────────────────────────────────────
messages: ["Olá", "Oi!"]         messages: [
                                     HumanMessage("Olá"),
                                     AIMessage("Oi!")
                                  ]
```

Com `string[]`:
- Você não sabe quem disse o quê sem convenções extras
- Para passar ao LLM, você precisa converter para o formato esperado pela API
- Não dá pra adicionar metadados (ID, timestamp, tool calls)

Com `BaseMessage[]`:
- O papel está embutido no tipo
- O LangChain converte para o formato correto de cada API automaticamente
- Suporta extensão: `ToolMessage`, `FunctionMessage`, etc.

Os projetos seguintes (especialmente `04-song-highlights`) usam `messages` para alimentar o modelo e construir o histórico de conversa multi-turn — e a consistência de tipos já estabelecida aqui torna isso simples.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| HumanMessage como input | `src/server.ts` | `new HumanMessage(question)` na chamada `.invoke()` |
| BaseMessage no estado | `src/graph/graph.ts` | `z.custom<BaseMessage[]>()` + `MessagesZodMeta` |
| Leitura da última mensagem | `src/graph/nodes/identifyIntentNode.ts` | `state.messages.at(-1)?.text` |
| AIMessage como output | `src/graph/nodes/chatResponseNode.ts` | `new AIMessage(responseText)` |
| AIMessage no fallback | `src/graph/nodes/fallbackNode.ts` | `.content.toString()` para acessar o texto |
