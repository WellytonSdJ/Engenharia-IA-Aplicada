# Sumarização de Conversas

## O problema que resolve

LLMs têm uma janela de contexto finita. Numa conversa longa, você tem duas opções ruins:

1. Mandar o histórico completo a cada mensagem → custo alto, eventualmente estoura o limite
2. Descartar mensagens antigas → o modelo perde contexto e parece "amnésico"

Sumarização é o meio-termo: você comprime o histórico em um resumo estruturado e descarta as mensagens originais. O resumo vai para o contexto da próxima mensagem, mantendo o essencial sem o custo do histórico completo.

---

## Sumarização incremental

O padrão mais robusto não é sumarizar do zero a cada vez. É sumarização incremental: você passa o resumo anterior junto com a nova conversa, e o LLM faz o merge.

```text
[Resumo anterior] + [Novas mensagens] → [Resumo atualizado]
```

Vantagens:

- Informações de sessões antigas não se perdem
- O custo da sumarização é proporcional à conversa nova, não ao histórico total
- Se o usuário menciona algo que contraria o resumo anterior, o LLM pode corrigir

No projeto:

```ts
// summarizationNode.ts
const previousSummary = state.conversationSummary as ConversationSummary | undefined

const userPrompt = getSummarizationUserPrompt(
  conversationHistory,
  previousSummary,  // passado junto — pode ser undefined na primeira sumarização
)
```

```ts
// prompts/v1/summarization.ts
return JSON.stringify({
  conversa: conversationHistory.map(...).join('\n'),
  sumario_anterior: previousSummary || 'Nenhum',  // LLM recebe "Nenhum" se for a primeira vez
  instrucoes: [
    'Atualizar sumário com novas informações desta conversa',
    'Preservar info existente não discutida nas novas mensagens'
  ]
})
```

---

## Quando sumarizar

O trigger é por contagem de mensagens. O `chatNode` verifica após cada resposta:

```ts
// chatNode.ts
const totalMessages = state.messages.length
const needsSummarization = totalMessages >= config.maxMessagesToSummary
```

`maxMessagesToSummary` está definido em `config.ts` como `2` (valor de teste — em produção seria maior, ex: 10 ou 20).

Quando `needsSummarization` é `true`, a edge condition do `routeAfterChat` (ou `routeAfterSavePreferences`) direciona para o nó `summarize`.

---

## O que o summarizationNode faz

```text
1. Converte state.messages → array { role, content }
2. Busca resumo anterior do state (conversationSummary)
3. Chama LLM com histórico + resumo anterior → retorna ConversationSummary (Zod)
4. Persiste o novo resumo no SQLite (preferencesService.storeSummary)
5. Deleta todas as mensagens exceto as 2 últimas via RemoveMessage
6. Atualiza state: messages (removidos), conversationSummary (novo), needsSummarization = false
```

O passo 5 é o mais específico do LangGraph. `RemoveMessage` não deleta imediatamente — é uma instrução para o reducer de mensagens processar:

```ts
const deleteMessages = state.messages
  .slice(0, -2)  // pega tudo exceto as 2 últimas
  .map(m => new RemoveMessage({ id: m.id as string }))

return {
  messages: deleteMessages,  // o reducer vai processar cada RemoveMessage
  conversationSummary: result.data,
  needsSummarization: false,
}
```

---

## Schema do resumo

```ts
// prompts/v1/summarization.ts
export const SummarySchema = z.object({
  name: z.string().nullable(),
  age: z.number().nullable(),
  favoriteGenres: z.array(z.string()).nullable(),
  favoriteBands: z.array(z.string()).nullable(),
  keyPreferences: z.string(),           // resumo textual de 2-4 frases — único campo obrigatório
  importantContext: z.string().nullable(),
})
```

O resumo é estruturado (não texto livre) porque precisa ser mesclado com as preferências salvas no SQLite via `storeSummary`. Campos tipados facilitam o merge programático.

---

## Dois sistemas de persistência do resumo

O resumo é salvo em dois lugares com propósitos diferentes:

| | `state.conversationSummary` | SQLite via `preferencesService` |
| --- | --- | --- |
| Escopo | Thread atual (Postgres checkpointer) | Persistente entre sessões |
| Usado em | Próxima sumarização incremental | Carregado no system prompt no início da sessão |
| Quem lê | `summarizationNode` | `chatNode` e `index.ts` |

Na próxima sessão, `getBasicInfo` lê o SQLite e injeta o resumo no `userContext`, que vai para o system prompt:

```ts
// index.ts
const userContext = await preferencesService.getBasicInfo(actualUserId)

// chatNode.ts
const systemPrompt = getSystemPrompt(userContext)

// chatResponse.ts
preferencias_previamente_armazenadas: userContext || 'Nenhuma'
```

---

## Referências no projeto

| Conceito | Arquivo |
| --- | --- |
| Nó de sumarização | [src/graph/nodes/summarizationNode.ts](../src/graph/nodes/summarizationNode.ts) |
| Prompts e schema | [src/prompts/v1/summarization.ts](../src/prompts/v1/summarization.ts) |
| Trigger de sumarização | [src/graph/nodes/chatNode.ts](../src/graph/nodes/chatNode.ts) |
| Roteamento para summarize | [src/graph/nodes/edgeConditions.ts](../src/graph/nodes/edgeConditions.ts) |
| Persistência do resumo | [src/services/preferencesService.ts](../src/services/preferencesService.ts) |
