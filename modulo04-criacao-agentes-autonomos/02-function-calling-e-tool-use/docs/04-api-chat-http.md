# `POST /chat` — a API que é, ela mesma, um agente

> Este doc cobre a spec 004 ([`004-http-chat-endpoint`](../specs/004-http-chat-endpoint/spec.md)). Antes, leia [01 / 02-contrato-de-estrategia-e-trace.md](../../01-padroes-de-raciocinio-e-execucao/docs/02-contrato-de-estrategia-e-trace.md): `ReasoningStrategy`, `trace` e `metrics` são a resposta deste endpoint.
>
> **Em 30 segundos**
>
> - O OpsPilot sai da CLI e ganha uma porta HTTP: `POST /chat { message, strategy?, reflect? }`.
> - Atrás da rota não há um CRUD. Há **um agente inteiro** (ReAct ou Plan-and-Execute, com ou sem Reflection).
> - A borda faz três coisas: **valida** com Zod, **escolhe** a estratégia num registry e **protege** com timeout.
> - Os erros de domínio viram status HTTP honestos: 400, 422, 504 ou 500.

---

## O que é: um endpoint cujo "corpo" é um agente

Numa API comum, `POST /pedidos` grava uma linha e responde. Aqui, `POST /chat` **raciocina**: pode chamar 3 tools, replanejar e se autocriticar antes de responder. É como ligar para uma central e, em vez de cair numa gravação, falar com um plantonista que vai investigar antes de responder.

```text
cliente ──POST /chat──▶ Express ──▶ Zod (fronteira) ──▶ registry ──▶ estratégia.run()
                                         │ 400            │ 422           │ timeout → 504
                                         ▼                ▼               ▼
                                   corpo inválido   estratégia      { answer, trace, metrics }
                                                    desconhecida
```

O que muda para quem consome:

| Antes (01) | Agora (02) |
| --- | --- |
| `npm run arena -- "pergunta"` no terminal | qualquer cliente HTTP (curl, front, outro serviço) |
| Estratégia escolhida por flag da CLI | `strategy` no corpo, com default `react` |
| Sem limite de tempo | timeout de 60 s → `504` |

> ✅ **Checkpoint**: a rota é fina. Ela valida, escolhe, cronometra e traduz erro. Quem pensa é a estratégia.

---

## Como está no projeto

### 1. Validação na fronteira (Zod)

```typescript
// src/controllers/chat-controller.ts
export const ChatRequestSchema = z.object({
  message: z.string().trim().min(1),
  strategy: z.string().trim().min(1).optional(),
  reflect: z.boolean().default(false),
  conversationId: z.string().trim().min(1).optional(),
});
```

Nada passa para o agente sem esse filtro: `message` vazia ou `reflect: "sim"` param aqui, com as `issues` do Zod no corpo do `400`. (`conversationId` é da spec 008: veja [06-conversa-persistente.md](./06-conversa-persistente.md).)

### 2. Registry: trocar estratégia sem mudar a rota

```typescript
// src/agents/index.ts
export const strategyRegistry: StrategyRegistry = {
  react: reactStrategy,
  "plan-and-execute": planAndExecuteStrategy,
};

export const DEFAULT_STRATEGY = "react";

export function resolveStrategy(
  name: string | undefined,
  options: { reflect: boolean },
  registry: StrategyRegistry = strategyRegistry,
): ReasoningStrategy {
  const key = name ?? DEFAULT_STRATEGY;
  if (!Object.hasOwn(registry, key)) {
    throw new UnknownStrategyError(key, Object.keys(registry));
  }
  const base = registry[key];
  return options.reflect ? withReflection(base) : base;
}
```

Duas ideias aqui:
- **Registry** = um dicionário nome → estratégia. Nova estratégia? Uma linha no objeto, e a rota nem fica sabendo.
- `reflect: true` aplica o **decorator** `withReflection` por cima de qualquer estratégia ([01 / 06-reflection.md](../../01-padroes-de-raciocinio-e-execucao/docs/06-reflection.md)).

💡 `Object.hasOwn` e não `key in registry`: assim `strategy: "toString"` não "existe" por herança do protótipo.

### 3. Timeout com `Promise.race`

```typescript
// src/services/chat.ts
export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  let timer: NodeJS.Timeout;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(new ExecutionTimeoutError(ms)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => {
    clearTimeout(timer);
    promise.catch(() => {});
  });
}
```

Uma corrida entre o agente e um cronômetro: quem terminar primeiro vence. Dois detalhes:
- `clearTimeout` no `finally`: sem ele, o timer segura o processo vivo mesmo depois da resposta.
- `promise.catch(() => {})`: se o agente falhar *depois* de perder a corrida, o erro não vira `unhandledRejection`.

### 4. Erro de domínio → status HTTP

```typescript
// src/controllers/chat-controller.ts (resumo de toErrorResponse)
if (error instanceof z.ZodError)               → 400 INVALID_BODY (+ issues)
if (error instanceof UnknownStrategyError)     → 422 UNKNOWN_STRATEGY (+ available)
if (error instanceof ExecutionTimeoutError)    → 504 TIMEOUT
if (error instanceof UnknownConversationError) → 422 UNKNOWN_CONVERSATION
qualquer outro                                 → 500 INTERNAL (sem vazar stack)
```

| Status | Quando |
| --- | --- |
| 200 | sucesso: `{ answer, trace, metrics, conversationId }` |
| 400 | corpo inválido (JSON malformado, `message` vazia, tipos errados) |
| 422 | `strategy` fora do registry, ou `conversationId` inexistente |
| 504 | execução passou de 60 s |
| 500 | erro inesperado (logado no servidor, mensagem genérica para o cliente) |

> ✅ **Checkpoint**: **400** = "seu pedido está mal escrito"; **422** = "está bem escrito, mas pede algo que não existe"; **504** = "o agente demorou demais".

---

## Testando sem rede

O `createServer(deps)` recebe as dependências por injeção: registry, timeout, modelo e store de conversa. Os testes de [`src/http/server.test.ts`](../src/http/server.test.ts) montam o servidor com uma **estratégia fake** determinística e conferem status e corpo, sem chamar nenhum modelo.

```bash
npm run dev
curl -X POST http://localhost:3000/chat \
  -H "content-type: application/json" \
  -d '{"message": "quantos alertas criticos estao disparando?", "strategy": "react"}'
```

---

## ⚠️ Pegadinhas

- **Default no lugar errado.** O default `react` mora no `resolveStrategy`, não no schema. Assim a regra vale para qualquer chamador, não só para o HTTP.
- **`422` × `400`.** Estratégia desconhecida **não** é corpo inválido: o JSON está certo, só pede algo que não existe.
- **Timeout não cancela o agente.** O `Promise.race` responde `504`, mas a execução continua em segundo plano até terminar. Ele protege o **cliente**, não economiza tokens.
- **Vazar stack no 500.** O `chatErrorHandler` loga o erro com `console.error` e devolve só `"Internal server error."`.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Rota e error handler | [`src/http/server.ts`](../src/http/server.ts) | `createServer(deps)`, `express.json()` |
| Validação e status | [`src/controllers/chat-controller.ts`](../src/controllers/chat-controller.ts) | `ChatRequestSchema`, `toErrorResponse`, JSON malformado → 400 |
| Registry | [`src/agents/index.ts`](../src/agents/index.ts) | `strategyRegistry`, `resolveStrategy`, `withReflection` |
| Timeout | [`src/services/chat.ts`](../src/services/chat.ts) | `withTimeout`, `DEFAULT_CHAT_TIMEOUT_MS` |
| Erros de domínio | [`src/services/errors.ts`](../src/services/errors.ts) | `UnknownStrategyError`, `ExecutionTimeoutError` |
| Entry point | [`src/index.ts`](../src/index.ts) | `PORT` validado com `z.coerce.number()` |
| Contrato | [`specs/004-http-chat-endpoint/contracts/`](../specs/004-http-chat-endpoint/contracts/) | O contrato HTTP completo |

---

⬅️ Anterior: [03 · Tool externa resiliente](./03-tool-externa-resiliente.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [05 · Servidor MCP do OpsPilot](./05-servidor-mcp.md)**
