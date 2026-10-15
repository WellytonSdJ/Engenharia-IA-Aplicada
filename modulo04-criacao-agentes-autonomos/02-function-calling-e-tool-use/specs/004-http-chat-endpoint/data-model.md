# Data Model: Endpoint HTTP de Chat

Não há persistência nova. As entidades abaixo são contratos em memória.

## ChatRequest (entrada HTTP, `ChatRequestSchema`)

| Campo | Tipo | Regra | Padrão |
| --- | --- | --- | --- |
| `message` | string | `trim()`, mínimo 1 caractere | obrigatório |
| `strategy` | string | `trim()`, mínimo 1 caractere quando presente | `"react"` (aplicado na resolução) |
| `reflect` | boolean | booleano estrito (`"true"` em string é inválido) | `false` |

- Campos desconhecidos são descartados.
- A validação falha → `ZodError` → 400.

## StrategyRegistry

- `Readonly<Record<string, ReasoningStrategy>>`.
- O registro de produção é `{ react: reactStrategy, "plan-and-execute": planAndExecuteStrategy }`.
- `resolveStrategy(name, { reflect }, registry)`:
  - nome ausente do registro (`Object.hasOwn` falso) → `UnknownStrategyError(name, Object.keys(registry))`;
  - `reflect: false` → a estratégia base;
  - `reflect: true` → `withReflection(base)` (`name` = `reflect:<base>`).

## ChatDependencies (injeção na montagem do servidor)

| Campo | Tipo | Padrão |
| --- | --- | --- |
| `registry` | `StrategyRegistry` | `strategyRegistry` |
| `timeoutMs` | number inteiro positivo | `60_000` |
| `model` | `BaseChatModel` opcional | `undefined` (as estratégias resolvem o modelo real) |

## ChatResponse (200)

É idêntico a `ReasoningResult` (`src/agents/trace.ts`): `{ answer: string, trace: TraceEvent[], metrics: { llmCalls, latencyMs, iterations? } }`, validado por `ReasoningResultSchema` antes do envio.

## Erros de domínio (novos, em `src/services/errors.ts`)

| Classe | `code` | Dados extras | Status na borda |
| --- | --- | --- | --- |
| `UnknownStrategyError` | `UNKNOWN_STRATEGY` | `strategy: string`, `available: string[]` | 422 |
| `ExecutionTimeoutError` | `TIMEOUT` | `timeoutMs: number` | 504 |

`DomainErrorCode` ganha `"UNKNOWN_STRATEGY" | "TIMEOUT"`.

## Ciclo de uma requisição

```text
recebida → corpo parseado ──falha──▶ 400
              │
              ▼
          validada (Zod) ──falha──▶ 400
              │
              ▼
        estratégia resolvida ──desconhecida──▶ 422
              │
              ▼
          executando ──> timeoutMs──▶ 504 (resultado tardio descartado)
              │        └─erro──────▶ 500
              ▼
          resultado validado ──▶ 200
```
