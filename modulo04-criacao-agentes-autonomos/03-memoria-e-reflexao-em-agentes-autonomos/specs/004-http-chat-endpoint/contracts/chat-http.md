# Contract: `POST /chat`

## Request

- **Method/Path**: `POST /chat`
- **Content-Type**: `application/json`

```json
{
  "message": "quais alertas estão disparando?",
  "strategy": "plan-and-execute",
  "reflect": true
}
```

| Campo | Obrigatório | Tipo | Observação |
| --- | --- | --- | --- |
| `message` | sim | string não vazia (após trim) | |
| `strategy` | não | string não vazia | padrão `react`; valores do registro: `react`, `plan-and-execute` |
| `reflect` | não | boolean | padrão `false`; `true` aplica `withReflection` à estratégia |

Campos extras são ignorados.

## Responses

### 200 OK

```json
{
  "answer": "Há 2 alertas firing: ...",
  "trace": [
    { "type": "thought", "content": "..." },
    { "type": "action", "content": "...", "tool": "list_alerts", "args": {} },
    { "type": "observation", "content": "..." },
    { "type": "critique", "content": "..." },
    { "type": "answer", "content": "..." }
  ],
  "metrics": { "llmCalls": 3, "latencyMs": 4210.5, "iterations": 2 }
}
```

O formato é o de `ReasoningResultSchema` (`src/agents/trace.ts`). Eventos `critique` só aparecem com `reflect: true`.

### 400 Bad Request: corpo inválido

```json
{
  "error": "INVALID_BODY",
  "message": "Request body is invalid.",
  "issues": [
    { "code": "too_small", "path": ["message"], "message": "String must contain at least 1 character(s)" }
  ]
}
```

- `issues` são as issues do Zod (`code`, `path`, `message` e campos específicos do tipo de issue).
- JSON malformado gera uma issue única `{ "code": "custom", "path": [], "message": "Malformed JSON body." }`.
- Nenhuma estratégia é executada.

### 422 Unprocessable Entity: estratégia desconhecida

```json
{
  "error": "UNKNOWN_STRATEGY",
  "message": "Unknown strategy \"tree-of-thought\".",
  "strategy": "tree-of-thought",
  "available": ["react", "plan-and-execute"]
}
```

### 504 Gateway Timeout

```json
{ "error": "TIMEOUT", "message": "Strategy execution exceeded 60000 ms." }
```

A execução interna pode continuar após o 504; o resultado é descartado.

### 500 Internal Server Error

```json
{ "error": "INTERNAL", "message": "Internal server error." }
```

A mensagem é fixa. O detalhe do erro só aparece no log do servidor.

## Precedência

`400` (parse/validação) → `422` (resolução) → `504`/`500`/`200` (execução).

## Contrato interno de montagem

```ts
createServer(deps?: Partial<{
  registry: StrategyRegistry;   // padrão: strategyRegistry de src/agents/index.ts
  timeoutMs: number;            // padrão: 60_000
  model: BaseChatModel;         // padrão: undefined
}>): express.Express

resolveStrategy(
  name: string | undefined,     // undefined → "react"
  options: { reflect: boolean },
  registry?: StrategyRegistry,
): ReasoningStrategy            // lança UnknownStrategyError
```
