# Implementation Plan: Endpoint HTTP de Chat

**Branch**: `004-http-chat-endpoint` | **Date**: 2026-10-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-http-chat-endpoint/spec.md`

## Summary

Expor as estratégias de raciocínio por `POST /chat` em um app Express 5 montado em `src/http/server.ts`. O controller valida o corpo com Zod (`message`, `strategy?`, `reflect?`) e delega a um service de chat. O service resolve a estratégia pelo registro central em `src/agents/index.ts` (padrão `react`; `reflect: true` aplica `withReflection`) e executa a estratégia com um timeout de 60 s. Erros de domínio novos (`UnknownStrategyError`, `ExecutionTimeoutError`) são traduzidos para 422 e 504 somente na borda HTTP; corpo inválido vira 400 com as issues do Zod e falhas inesperadas viram 500 genérico. O servidor recebe registro, timeout e modelo por injeção, o que permite testes de integração por HTTP real em porta efêmera, com estratégia fake determinística e sem rede.

## Technical Context

**Language/Version**: TypeScript ESM, `strict: true`, Node.js 22 LTS

**Primary Dependencies**: Já existentes: Express 5, Zod, `@langchain/core`. Nenhuma dependência nova (sem `supertest`; os testes usam o `fetch` global do Node 22)

**Storage**: Sem persistência nova; as estratégias continuam resolvendo o `OpsStore` padrão

**Testing**: `node:test` via `tsx`; testes unitários do registro e do helper de timeout; testes de integração do endpoint com `app.listen(0)` + `fetch`; gates `npm run typecheck` e `npm test`

**Target Platform**: Servidor HTTP Node.js local (`npm run dev`)

**Project Type**: Aplicação TypeScript com agente, CLIs (arena/bench) e, agora, API HTTP

**Performance Goals**: Resposta em no máximo 60 s + margem de processamento; a validação e a resolução da estratégia não fazem chamadas ao modelo

**Constraints**: Não alterar estratégias, arena nem bench; não expor stack trace ou secrets; timeout configurável só por injeção (o cliente não controla); sem cancelamento cooperativo da execução

**Scale/Scope**: Uma rota, um registro, um service, um controller, dois erros de domínio e as suítes de teste correspondentes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. TypeScript ESM estrito**: **PASS**. Contratos tipados (`ChatRequest`, `ChatDependencies`, `StrategyRegistry`), sem `any`; os imports usam `.js`, como no resto do projeto.
- **II. Arquitetura em camadas**: **PASS**. O roteamento e o tratamento de erros HTTP ficam em `src/http/server.ts`; a validação e o mapeamento de status, no controller (`src/controllers/chat-controller.ts`); a resolução e a execução com timeout, no service (`src/services/chat.ts`); o registro de estratégias, em `src/agents/index.ts`. `withTimeout` é função pura, com o timer como único efeito.
- **III. Contratos e erros explícitos**: **PASS**. O corpo é validado por `ChatRequestSchema` e o resultado por `ReasoningResultSchema` antes da resposta. `UnknownStrategyError` e `ExecutionTimeoutError` estendem `OpsPilotError` e só viram status HTTP no controller.
- **IV. Testes e qualidade**: **PASS**. Cobertura de 200, `reflect`, 400, 422, 504, 500, do registro e do timeout; os gates oficiais são executados ao final.
- **V. Integração segura e simples**: **PASS**. Usa apenas Express e Zod já instalados; o `.env` continua carregado por `--env-file-if-exists` em `npm run dev`; sem `dotenv`; erros 500 não ecoam a mensagem original.

**Post-design re-check**: **PASS**. O design de Phase 1 não introduz violações; Complexity Tracking não se aplica.

## Project Structure

### Documentation (this feature)

```text
specs/004-http-chat-endpoint/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── chat-http.md
└── tasks.md             # gerado por /speckit-tasks
```

### Source Code (repository root)

```text
src/
├── index.ts                      # (vazio hoje) passa a iniciar o servidor HTTP via npm run dev
├── http/
│   ├── server.ts                 # NOVO: createServer(deps) + startServer(port)
│   └── server.test.ts            # NOVO: integração HTTP com estratégia fake
├── controllers/
│   └── chat-controller.ts        # NOVO: ChatRequestSchema, handler, mapeamento erro → status
├── services/
│   ├── chat.ts                   # NOVO: runChat(request, deps), withTimeout
│   ├── chat.test.ts              # NOVO: withTimeout e runChat sem HTTP
│   └── errors.ts                 # ALTERADO: códigos UNKNOWN_STRATEGY/TIMEOUT e classes
└── agents/
    ├── index.ts                  # NOVO: strategyRegistry + resolveStrategy
    └── index.test.ts             # NOVO: resolução base, refletida e desconhecida
```

**Structure Decision**: Projeto único existente. A camada HTTP entra em `src/http/`, conforme pedido. Controller e service seguem as pastas `src/controllers/` e `src/services/` que já existem. A arena (`src/arena.ts`) mantém seu registro com chaves `reflect:*` e não é migrada (spec, Assumptions).

## Decisões e rastreabilidade

| Decisão | Atende |
| --- | --- |
| `ChatRequestSchema = { message: string.trim().min(1), strategy?: string.trim().min(1), reflect?: boolean = false }`; campos extras ignorados (`z.object` padrão, não `strict`) | FR-001, FR-002, edge cases de espaços, `strategy` vazio e campos extras |
| A validação acontece antes de resolver a estratégia | Edge case "corpo inválido + estratégia desconhecida → 400", SC-002 |
| JSON malformado (`entity.parse.failed` do `express.json()`) → 400 com uma issue sintética `{ path: [], message }`; corpo ausente/não objeto falha no Zod → 400 | FR-002, edge case de corpo ausente |
| `resolveStrategy(name = "react", { reflect }, registry)` usa `Object.hasOwn` (nomes como `constructor` → 422, sem acessar o protótipo) e aplica `withReflection(base)` com as opções padrão quando `reflect` | FR-003 a FR-006, US4 |
| O registro de produção contém só `react` e `plan-and-execute`; `reflect:react` → 422 | FR-005, edge case de variante refletida |
| `withTimeout(promise, ms)` faz `Promise.race` com o timer, `clearTimeout` em `finally` e um `.catch` silencioso na promise original para evitar *unhandled rejection* tardia | FR-008, SC-004, edge case de resultado tardio |
| `ChatDependencies = { registry, timeoutMs, model? }`, com padrões `strategyRegistry` e `60_000`; `model` é repassado a `strategy.run` (necessário para o crítico fake no teste de `reflect`) | FR-009, FR-012 |
| O controller mapeia `UnknownStrategyError → 422`, `ExecutionTimeoutError → 504`, `ZodError → 400` e o resto → 500 `{ error: "INTERNAL", message: "Internal server error." }` | FR-006, FR-008, FR-010, FR-011 |
| O resultado passa por `ReasoningResultSchema.parse` antes do 200 | FR-007, US1 cenário 3 |
| `src/index.ts` chama `startServer(Number(process.env.PORT ?? 3000))` com o guard `pathToFileURL`, como em `arena.ts` | Assumption sobre `npm run dev` |

## Tratamento de erros

| Origem | Classe | Status | Corpo |
| --- | --- | --- | --- |
| JSON malformado | `SyntaxError` do body-parser (`type === "entity.parse.failed"`) | 400 | `{ error: "INVALID_BODY", message, issues: [{ path: [], message }] }` |
| Corpo fora do schema | `ZodError` | 400 | `{ error: "INVALID_BODY", message, issues }` |
| Estratégia inexistente | `UnknownStrategyError` (`UNKNOWN_STRATEGY`) | 422 | `{ error: "UNKNOWN_STRATEGY", message, strategy, available }` |
| Execução > `timeoutMs` | `ExecutionTimeoutError` (`TIMEOUT`) | 504 | `{ error: "TIMEOUT", message }` |
| Qualquer outro (incl. `IterationLimitError`, `AgentConfigurationError`, falha do modelo) | — | 500 | `{ error: "INTERNAL", message: "Internal server error." }` + `console.error` no servidor |

## Testes

- `src/agents/index.test.ts`: resolve `react` por padrão; resolve um nome base sem reflection (`name` igual ao base); com `reflect`, `name === "reflect:<base>"`; um nome desconhecido lança `UnknownStrategyError` com `available`; `constructor` e `reflect:react` são desconhecidos.
- `src/services/chat.test.ts`: `withTimeout` resolve dentro do prazo; rejeita com `ExecutionTimeoutError` fora do prazo; a rejeição tardia da promise original não gera *unhandled rejection*; `runChat` repassa `message` e `model` à estratégia.
- `src/http/server.test.ts` (integração, porta efêmera, sem rede): 200 com padrão `react` (registro fake com chave `react`); 200 com `strategy` nomeada; `reflect: true` com `ControlledCriticModel` injetado → trace com `critique`; `reflect` ausente → sem `critique`; 400 para JSON malformado, `message` ausente/vazio/só espaços, `strategy: ""` e tipo errado, sempre com `issues`, sem executar a estratégia; 422 com `available`; 400 vence 422; 504 com `timeoutMs: 50` e estratégia que nunca resolve; 500 sem vazar a mensagem do erro original.

## Riscos

- **Execução continua após o 504**: consome tokens e tempo do modelo em segundo plano. Fica aceito pela spec; documentado no contrato. Mitigação futura: `AbortSignal` em `ReasoningOptions`.
- **Testes de timeout instáveis**: usar um timeout curto (50 ms) e uma estratégia que nunca resolve evita depender do relógio para o caso feliz; `clearTimeout` impede que o processo de teste fique pendurado.
- **Duplicação do modelo fake de crítico**: `ControlledCriticModel` vive em `reflection.test.ts`. O teste HTTP precisa de um equivalente mínimo; aceitar uma cópia enxuta ou extraí-lo para um helper de teste compartilhado, decisão que fica para o tasks.
- **`src/index.ts` passa a abrir porta**: qualquer import desse arquivo em testes subiria o servidor; o guard `pathToFileURL` evita isso.
