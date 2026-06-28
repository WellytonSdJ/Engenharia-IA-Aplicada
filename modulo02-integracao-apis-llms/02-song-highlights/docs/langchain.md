# LangChain

## O que é

LangChain é um framework para construir aplicações com LLMs. Ele não é um modelo — é uma camada de abstração que padroniza como você interage com diferentes providers (OpenAI, Anthropic, Mistral, etc.) e como você compõe operações em pipelines.

A ideia central é: em vez de montar manualmente requests HTTP para cada API de LLM, você trabalha com interfaces uniformes que funcionam do mesmo jeito independente do provider.

---

## Abstrações core usadas no projeto

### Messages

LangChain representa a conversa como uma lista de objetos de mensagem tipados:

```ts
import { SystemMessage, HumanMessage, AIMessage } from '@langchain/core/messages'

// System: instrução para o modelo (não é "fala" do usuário)
new SystemMessage("Você é um assistente musical")

// Human: input do usuário
new HumanMessage("Recomende algo de jazz")

// AI: resposta do modelo
new AIMessage("Experimente Kind of Blue do Miles Davis")
```

Essa tipagem importa porque:
- O LangGraph usa `HumanMessage.isInstance(msg)` para distinguir quem falou
- O reducer de mensagens do LangGraph trata `RemoveMessage` como instrução especial de deleção
- Cada tipo serializa diferente quando vai para a API do modelo

### ChatOpenAI

```ts
import { ChatOpenAI } from '@langchain/openai'

const llm = new ChatOpenAI({
  apiKey: '...',
  modelName: 'gpt-4o',
  temperature: 0.7,
  configuration: {
    baseURL: 'https://openrouter.ai/api/v1', // troca o endpoint sem mudar o código
  }
})
```

`ChatOpenAI` implementa a interface `BaseChatModel` do LangChain. Isso significa que qualquer código escrito contra essa interface funciona com qualquer modelo que implemente a mesma interface — você troca o provider mudando só a instância.

No projeto, o `baseURL` aponta para o OpenRouter em vez da OpenAI direta. O cliente não sabe a diferença.

### withStructuredOutput

```ts
const structuredLlm = llm.withStructuredOutput(MeuSchema)
const resultado = await structuredLlm.invoke(messages)
// resultado já é tipado e validado pelo schema Zod
```

Por baixo dos panos, `withStructuredOutput` usa uma das duas estratégias dependendo do modelo:

1. **Tool calling** — define o schema como uma "tool" e força o modelo a chamá-la. O JSON retornado é o argumento da tool.
2. **JSON mode** — instrui o modelo a responder em JSON e valida o output contra o schema.

O LangChain escolhe automaticamente baseado no que o modelo suporta. O resultado é sempre um objeto TypeScript validado — sem `JSON.parse` manual, sem `try/catch` por schema inválido.

No projeto, isso está encapsulado em `OpenRouterService.generateStructured`:

```ts
// services/openrouterService.ts
const structuredLlm = this.llmClient.withStructuredOutput(schema as z.ZodSchema)
const data = await structuredLlm.invoke(messages)
return { success: true, data: data as T }
```

---

## Por que usar JSON como prompt

No projeto, os prompts são serializados com `JSON.stringify`:

```ts
return JSON.stringify({
  role: 'Assistente musical',
  tarefas: [...],
  regras_de_extracao: {...},
})
```

Isso não é obrigatório, mas tem uma razão prática: modelos treinados com muitos dados de código e APIs respondem bem a inputs estruturados. Um prompt em JSON tende a gerar melhor seguimento de instruções em campos específicos do que texto corrido em markdown — especialmente quando você está pedindo uma resposta também estruturada (structured output).

---

## Integração com LangGraph

LangChain e LangGraph são pacotes separados, mas projetados para trabalhar juntos:

- LangChain fornece os tipos de mensagem (`BaseMessage`, `HumanMessage`, etc.)
- LangGraph usa esses tipos no state e no reducer
- `MessagesZodMeta` (LangGraph) ensina o Zod a usar o reducer de mensagens do LangGraph no campo `messages` do state

```ts
// graph.ts
messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta)
```

Sem `MessagesZodMeta`, o LangGraph substituiria o array inteiro a cada update. Com ele, o LangGraph faz append de novas mensagens e processa `RemoveMessage` como deleção.

---

## Referências no projeto

| Conceito | Arquivo |
|---|---|
| `ChatOpenAI` + `withStructuredOutput` | [src/services/openrouterService.ts](../src/services/openrouterService.ts) |
| `HumanMessage`, `AIMessage`, `RemoveMessage` | [src/graph/nodes/chatNode.ts](../src/graph/nodes/chatNode.ts), [src/graph/nodes/summarizationNode.ts](../src/graph/nodes/summarizationNode.ts) |
| `SystemMessage`, `HumanMessage` no invoke | [src/services/openrouterService.ts](../src/services/openrouterService.ts) |
| Schemas Zod para structured output | [src/prompts/v1/chatResponse.ts](../src/prompts/v1/chatResponse.ts), [src/prompts/v1/summarization.ts](../src/prompts/v1/summarization.ts) |
