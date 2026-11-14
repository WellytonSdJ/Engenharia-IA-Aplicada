# Conversa persistente — o fio da meada entre requisições

> Este doc cobre a spec 008 ([`008-conversa-persistente`](../specs/008-conversa-persistente/spec.md)). Antes, leia [04-api-chat-http.md](./04-api-chat-http.md): ela estende o `POST /chat`. O próximo passo, a memória de **longo prazo**, está em [03 / 02-memoria-semantica.md](../../03-memoria-e-reflexao-em-agentes-autonomos/docs/02-memoria-semantica.md).
>
> **Em 30 segundos**
>
> - O modelo **não lembra de nada** entre chamadas. Quem lembra é o servidor, que guarda as mensagens e as **reinjeta** no prompt.
> - O `/chat` aceita um `conversationId` opcional. Sem ele, nasce uma conversa nova. Com ele, as **últimas 12 mensagens** entram no prompt.
> - O turno (pergunta + resposta) só é gravado **depois** do sucesso. Não existe resposta 200 com histórico pela metade.
> - Id bem formado mas inexistente → `422 UNKNOWN_CONVERSATION`.

---

## O que é: o modelo tem amnésia, o servidor tem caderno

Cada chamada a um LLM começa do zero. Se você pergunta "abra um incidente para o checkout" e depois "agora resolva **ele**", o segundo pedido não faz sentido sem o primeiro.

**Analogia:** um plantonista que esquece tudo a cada ligação. A solução é um **caderno de plantão**: antes de cada ligação, alguém lê para ele as últimas anotações. O plantonista é o modelo; o caderno é o `ConversationStore`.

```text
turno 1:  "abra incidente sev2 no checkout"      → conversa nova (id: conversation-abc)
turno 2:  "agora resolva ele" + id: conversation-abc
            → prompt enviado ao agente:
              Conversation so far:
              User: abra incidente sev2 no checkout
              Assistant: Incidente incident-123 aberto...

              User: agora resolva ele
```

> ✅ **Checkpoint**: "memória de conversa" é **texto reinjetado no prompt**, nada mágico. O modelo continua sem estado.

---

## Como está no projeto

### O fluxo do turno

```typescript
// src/services/chat.ts
export async function runChat(request: ChatRequest, deps: ChatDependencies): Promise<ChatResult> {
  const strategy = resolveStrategy(request.strategy, { reflect: request.reflect }, deps.registry);
  const conversation = request.conversationId
    ? { id: request.conversationId }
    : await deps.conversationStore.create();
  const history = await deps.conversationStore.lastMessages(conversation.id, MAX_HISTORY_MESSAGES);
  const input = composeInput(request.message, history);

  const result = await withTimeout(
    strategy.run(input, deps.model ? { model: deps.model } : undefined),
    deps.timeoutMs,
  );
  const parsed = ReasoningResultSchema.parse(result);

  await deps.conversationStore.append(conversation.id, [
    { role: "user", content: request.message },
    { role: "assistant", content: parsed.answer },
  ]);

  return ChatResultSchema.parse({
    ...parsed,
    conversationId: conversation.id,
    metrics: { ...parsed.metrics, historyMessages: history.length },
  });
}
```

```text
resolve estratégia → acha/cria conversa → últimas 12 msgs → monta prompt
   → roda agente (com timeout) → grava pergunta + resposta → 200 { ..., conversationId }
```

Repare na **ordem**: o `append` vem **depois** do `strategy.run`. Se o agente falhar ou estourar o tempo, nada é gravado.

### Montando o prompt com o histórico

```typescript
// src/services/chat.ts
function composeInput(message: string, history: Message[]): string {
  if (history.length === 0) {
    return message;
  }
  return `Conversation so far:\n${formatHistory(history)}\n\nUser: ${message}`;
}
```

Sem histórico, a mensagem vai pura. As estratégias nem sabem que existe conversa: elas recebem uma string, como sempre receberam.

### O store: mesmo estilo do `SqliteOpsStore`

```typescript
// src/models/conversation-store.ts
  async lastMessages(conversationId: string, limit: number): Promise<Message[]> {
    this.assertExists(conversationId);
    const statement: StatementSync = this.db.prepare(
      "SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?",
    );
    const rows = statement.all(conversationId, limit) as unknown as MessageRow[];
    return rows.reverse().map(toMessage);
  }
```

- Duas tabelas, `conversations` e `messages`, no mesmo arquivo `OPSPILOT_DB` (conexão própria). Os padrões são os de [02-persistencia-sqlite.md](./02-persistencia-sqlite.md): DDL idempotente, `CHECK (role IN ('user','assistant'))` e prepared statements.
- `seq` (e não `created_at`) define a ordem: as duas mensagens de um turno têm o mesmo timestamp.
- `ORDER BY seq DESC LIMIT ?` + `reverse()`: pega as **12 mais novas** e devolve em ordem cronológica.
- `assertExists` lança `UnknownConversationError`, que vira **422** na borda.

> ✅ **Checkpoint**: janela de 12 = só as mais recentes. Conversa longa perde o começo, e isso é proposital (o contexto do modelo tem limite).

---

## O contrato HTTP

```bash
curl -X POST http://localhost:3000/chat \
  -H "content-type: application/json" \
  -d '{"message": "abra um incidente sev2 para o checkout"}'
# → 200 { answer, trace, metrics: { ..., historyMessages: 0 }, conversationId: "conversation-..." }

curl -X POST http://localhost:3000/chat \
  -H "content-type: application/json" \
  -d '{"message": "agora resolva ele", "conversationId": "conversation-..."}'
# → 200 { ..., metrics: { ..., historyMessages: 2 } }
```

| Status | Quando (além dos de [04-api-chat-http.md](./04-api-chat-http.md)) |
| --- | --- |
| 422 | `UNKNOWN_CONVERSATION`: `conversationId` bem formado, mas inexistente |
| 400 | `conversationId` com formato inválido (ex.: string vazia) |

`metrics.historyMessages` diz quantas mensagens antigas entraram naquele turno. É ótimo para conferir se a conversa "pegou".

---

## Curto prazo × longo prazo

Esta é a **memória de curto prazo** do OpsPilot. No projeto 03 nasce a de **longo prazo**:

| | Conversa persistente (aqui) | Memória semântica ([03](../../03-memoria-e-reflexao-em-agentes-autonomos/docs/02-memoria-semantica.md)) |
| --- | --- | --- |
| Chave | `conversationId` | `userId` |
| O que guarda | mensagens brutas | fatos sobre a pessoa |
| Como busca | as 12 mais recentes, por ordem | as mais parecidas, por significado |

---

## ⚠️ Pegadinhas

- **Gravar antes de responder.** Se o `append` viesse antes do `run`, um timeout deixaria uma pergunta sem resposta no histórico, e o próximo turno ficaria confuso.
- **Ordenar por `created_at`.** Pergunta e resposta do mesmo turno têm o mesmo timestamp. Sem `seq`, a ordem vira sorteio.
- **Achar que 12 mensagens = 12 turnos.** São 6 turnos (cada turno grava 2 mensagens).
- **Inventar `conversationId`.** Um id inventado não cria conversa: ele dá 422. Para começar uma conversa, omita o campo.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Fluxo do turno | [`src/services/chat.ts`](../src/services/chat.ts) | `runChat`, `composeInput`, `MAX_HISTORY_MESSAGES = 12` |
| Store | [`src/models/conversation-store.ts`](../src/models/conversation-store.ts) | `create`, `append`, `lastMessages`, `seq`, `assertExists` |
| Borda HTTP | [`src/controllers/chat-controller.ts`](../src/controllers/chat-controller.ts) | `conversationId` no schema, 422 `UNKNOWN_CONVERSATION` |
| Domínio | [`src/models/domain.ts`](../src/models/domain.ts) | `ConversationSchema`, `MessageSchema` |
| Testes | [`src/models/conversation-store.test.ts`](../src/models/conversation-store.test.ts), [`src/services/chat.test.ts`](../src/services/chat.test.ts) | Janela de 12, falha não grava, 422 |

---

⬅️ Anterior: [05 · Servidor MCP do OpsPilot](./05-servidor-mcp.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [07 · Glossário](./07-glossario.md)**
