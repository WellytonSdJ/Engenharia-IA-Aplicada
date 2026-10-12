# Tasks: Endpoint HTTP de Chat

**Input**: Design documents from `/specs/004-http-chat-endpoint/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/chat-http.md`, `quickstart.md`

**Tests**: Incluídos. A spec exige testes (FR-012) e a constituição exige testes para toda mudança de lógica (princípio IV). Dentro de cada fase, escreva os testes primeiro e confirme que falham.

**Organization**: Tarefas agrupadas por user story. A US4 (registro) é pré-requisito de todas as outras e por isso é entregue na Phase 2.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivos diferentes, sem dependência pendente)
- **[Story]**: user story da spec (US1–US4)

---

## Phase 1: Setup

**Purpose**: Reaproveitar a infraestrutura existente.

Nenhuma tarefa. Express 5, Zod, `@langchain/core`, `tsx` e `node:test` já estão instalados; nenhuma dependência nova entra (research R1).

---

## Phase 2: Foundational (bloqueia todas as stories; entrega a US4)

**Purpose**: Erros de domínio e registro central de estratégias.

**Independent Test (US4)**: `node --import tsx --test src/agents/index.test.ts` resolve nome base, variante refletida e nome desconhecido sem chamar modelo.

- [ ] T001 Adicionar `"UNKNOWN_STRATEGY" | "TIMEOUT"` a `DomainErrorCode` e criar, estendendo `OpsPilotError`, `UnknownStrategyError(strategy: string, available: string[])`, com mensagem `Unknown strategy "<strategy>".` e as propriedades `readonly strategy` e `readonly available`, e `ExecutionTimeoutError(timeoutMs: number)`, com mensagem `Strategy execution exceeded <timeoutMs> ms.` e a propriedade `readonly timeoutMs`, em `src/services/errors.ts`.
- [ ] T002 [P] Escrever `src/agents/index.test.ts`, que deve falhar antes de T003 existir. Usar um registro fake com estratégias `{ name, run }` determinísticas e verificar:
  - `resolveStrategy(undefined, { reflect: false }, registry)` retorna a estratégia `react`;
  - um nome base retorna a mesma instância;
  - `reflect: true` retorna estratégia com `name === "reflect:<base>"` (sem executá-la);
  - `"missing"`, `"constructor"` e `"reflect:react"` lançam `UnknownStrategyError` com `available` igual às chaves do registro;
  - `Object.keys(strategyRegistry)` é exatamente `["react", "plan-and-execute"]`.
- [ ] T003 Criar `src/agents/index.ts` com:
  - `export type StrategyRegistry = Readonly<Record<string, ReasoningStrategy>>`;
  - `export const strategyRegistry: StrategyRegistry = { react: reactStrategy, "plan-and-execute": planAndExecuteStrategy }`;
  - `export const DEFAULT_STRATEGY = "react"`;
  - `export function resolveStrategy(name: string | undefined, options: { reflect: boolean }, registry: StrategyRegistry = strategyRegistry): ReasoningStrategy`. A função usa `name ?? DEFAULT_STRATEGY` e `Object.hasOwn(registry, key)`; se o nome não existir, lança `UnknownStrategyError(key, Object.keys(registry))`; com `reflect`, retorna `withReflection(base)` com as opções padrão (research R3, R4).

  Rodar `src/agents/index.test.ts` e confirmar verde.

**Checkpoint**: Registro e erros prontos. A US4 está completa e testada.

---

## Phase 3: User Story 1 - Conversar com o agente por HTTP (Priority: P1) 🎯 MVP

**Goal**: `POST /chat` com `{ message, strategy? }` responde 200 com `{ answer, trace, metrics }`.

**Independent Test**: Subir `createServer({ registry: fake })` em porta efêmera e verificar o 200 do padrão `react` e o 200 de estratégia nomeada, sem rede.

### Tests for User Story 1

- [ ] T004 [US1] Criar `src/http/server.test.ts`.
  - **Helpers:**
    - `startTestServer(deps)` faz `createServer(deps).listen(0, "127.0.0.1")`, aguarda `listening`, devolve `{ url, close }` e o teste chama `close` em `after`/`finally`;
    - `postChat(url, body: unknown | string)` faz `fetch` com `content-type: application/json`, envia strings cruas sem `JSON.stringify` e devolve `{ status, body }`;
    - `fakeStrategy(name, calls)` registra `input` e `options` e retorna `{ answer: \`${name}: ${input}\`, trace: [{ type: "observation", content: "obs" }, { type: "answer", content: ... }], metrics: { llmCalls: 1, latencyMs: 1 } }`.
  - **Casos:**
    1. `{ message: "oi" }` → 200, `answer === "react: oi"`, `trace` e `metrics` iguais aos da fake, só a fake `react` chamada;
    2. `{ message: "oi", strategy: "plan-and-execute" }` → 200 e só a fake `plan-and-execute` chamada;
    3. o corpo do 200 passa em `ReasoningResultSchema.parse`.

### Implementation for User Story 1

- [ ] T005 [P] [US1] Criar `src/services/chat.ts` com:
  - a interface `ChatDependencies { registry: StrategyRegistry; timeoutMs: number; model?: BaseChatModel }`;
  - `DEFAULT_CHAT_TIMEOUT_MS = 60_000`;
  - `resolveChatDependencies(partial?: Partial<ChatDependencies>): ChatDependencies`, com padrões `strategyRegistry` e `60_000`; valida `timeoutMs` como inteiro positivo com Zod;
  - `async function runChat(request: { message: string; strategy?: string; reflect: boolean }, deps: ChatDependencies): Promise<ReasoningResult>`, que chama `resolveStrategy(request.strategy, { reflect: request.reflect }, deps.registry)`, depois `strategy.run(request.message, deps.model ? { model: deps.model } : undefined)`, e retorna `ReasoningResultSchema.parse(result)`. O timeout entra em T013.
- [ ] T006 [P] [US1] Criar `src/controllers/chat-controller.ts` com:
  - `export const ChatRequestSchema = z.object({ message: z.string().trim().min(1), strategy: z.string().trim().min(1).optional(), reflect: z.boolean().default(false) })` (não usar `.strict()`: campos extras são descartados);
  - `export function createChatHandler(deps: ChatDependencies): RequestHandler`, que faz `ChatRequestSchema.parse(req.body)`, chama `runChat` e responde `res.status(200).json(result)`. Erros são repassados com `next(error)`; o mapeamento de status entra em T014.
- [ ] T007 [US1] Criar `src/http/server.ts`:
  - `export function createServer(deps?: Partial<ChatDependencies>): Express` monta `express()`, aplica `express.json()` à rota, registra `POST /chat` com `createChatHandler(resolveChatDependencies(deps))` e, por enquanto, um middleware de erro final que responde 500 `{ error: "INTERNAL", message: "Internal server error." }` e faz `console.error(error)`;
  - `export function startServer(port: number, deps?: Partial<ChatDependencies>): Promise<http.Server>` resolve no evento `listening`.

  Depende de T005 e T006. Rodar T004 e confirmar verde.
- [ ] T008 [US1] Preencher `src/index.ts`, hoje vazio, para que ele chame `startServer(port)`, com `port` vindo de `z.coerce.number().int().positive().default(3000).parse(process.env.PORT)` e um log `opsPilot HTTP listening on http://localhost:<port>`. A chamada fica sob o guard `process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href`, com `.catch` que faz `console.error` e `process.exitCode = 1`, como em `src/arena.ts`. Adicionar o comentário no estilo do arena sobre o `.env` carregado por `npm run dev`.

**Checkpoint**: MVP. `npm run dev` sobe o servidor e o caminho 200 funciona.

---

## Phase 4: User Story 2 - Pedir reflection na requisição (Priority: P1)

**Goal**: `reflect: true` envolve a estratégia escolhida com `withReflection`.

**Independent Test**: Com `createServer({ registry: fake, model: criticoFake })`, verificar que `reflect: true` gera eventos `critique` e que `reflect` ausente ou `false` não gera.

### Tests for User Story 2

- [ ] T009 [P] [US2] Adicionar a `src/services/chat.test.ts`, um arquivo novo, testes de `runChat` com registro fake: o `message` chega intacto à estratégia; o `deps.model` é repassado em `options.model`; sem `model`, o `options` chega `undefined`.
- [ ] T010 [US2] Adicionar a `src/http/server.test.ts` um `ControlledCriticModel` mínimo. Seguir o padrão de `src/agents/reflection.test.ts`: `SimpleChatModel` cujo `withStructuredOutput` retorna `RunnableLambda` com saídas enfileiradas, e cujo `_call` lança erro. Casos:
  1. `{ message: "oi", reflect: true }` com crítico `[{ aproved: true, feedback: "ok" }]` → 200, o `trace` contém exatamente um evento `type: "critique"` e `metrics.llmCalls` é maior que o da fake sozinha;
  2. `{ message: "oi", reflect: false }` e `{ message: "oi" }` → 200, sem eventos `critique` e o crítico não é consumido.

### Implementation for User Story 2

- [ ] T011 [US2] Ajustar `src/services/chat.ts` e `src/controllers/chat-controller.ts` apenas se T009 ou T010 falharem. Pelo design, `reflect` e `model` já chegam pelo fluxo de T003, T005 e T006. Registrar no PR se nenhuma mudança foi necessária.

**Checkpoint**: `reflect` funciona de ponta a ponta com crítico determinístico.

---

## Phase 5: User Story 3 - Receber erros claros e distintos (Priority: P1)

**Goal**: 400 (com `issues`), 422, 504 e 500 conforme `contracts/chat-http.md`.

**Independent Test**: Corpos inválidos, estratégia desconhecida, estratégia que nunca resolve com `timeoutMs: 50` e estratégia que lança erro produzem cada status e corpo do contrato, sem rede.

### Tests for User Story 3

- [ ] T012 [P] [US3] Adicionar a `src/services/chat.test.ts` testes de `withTimeout`:
  - uma promise que resolve antes do prazo devolve o valor;
  - uma promise pendente com `ms = 20` rejeita com `ExecutionTimeoutError` e `timeoutMs === 20`;
  - uma promise que rejeita depois do timeout não gera *unhandled rejection*: registrar um listener `process.on("unhandledRejection")` durante o teste e afirmar zero chamadas após aguardar a rejeição tardia;
  - uma promise que rejeita antes do prazo propaga o erro original.
- [ ] T013 [P] [US3] Adicionar a `src/http/server.test.ts`:
  1. **400** com `error: "INVALID_BODY"` e `issues` não vazio, sem nenhuma chamada à fake, para:
     - o corpo cru `"{message"`, cuja issue tem `path` `[]` e `message` `"Malformed JSON body."`;
     - `{}`;
     - `{ message: "" }`;
     - `{ message: "   " }`;
     - `{ message: 1 }`;
     - `{ message: "oi", strategy: "" }`;
     - `{ message: "oi", reflect: "true" }`;
     - nos casos de campo, `issues[0].path[0]` é o campo esperado;
  2. **campos extras:** `{ message: "oi", extra: 1 }` → 200;
  3. **422:** `{ message: "oi", strategy: "missing" }` responde `{ error: "UNKNOWN_STRATEGY", strategy: "missing", available: [...chaves do registro fake] }`; `strategy: "reflect:react"` também dá 422;
  4. **precedência:** `{ message: "", strategy: "missing" }` → 400;
  5. **504:** `timeoutMs: 50` e uma fake que retorna `new Promise(() => {})` → `{ error: "TIMEOUT" }`, e a resposta chega em menos de 1 s;
  6. **500:** uma fake que lança `new Error("secret detail")` → 500 com `message === "Internal server error."`; o corpo serializado não contém `"secret detail"`.

### Implementation for User Story 3

- [ ] T014 [US3] Em `src/services/chat.ts`:
  - adicionar `export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T>`, que faz `Promise.race` com um timer que rejeita `new ExecutionTimeoutError(ms)`, chama `clearTimeout` em `finally` e anexa `promise.catch(() => {})` para descartar uma rejeição tardia (research R2);
  - envolver `strategy.run(...)` em `runChat` com `withTimeout(..., deps.timeoutMs)`.

  Rodar T012 e confirmar verde.
- [ ] T015 [US3] Em `src/controllers/chat-controller.ts`:
  - **`toErrorResponse(error: unknown): { status: number; body: Record<string, unknown> }`**, que mapeia:
    - `ZodError` → 400 `{ error: "INVALID_BODY", message: "Request body is invalid.", issues: error.issues }`;
    - `UnknownStrategyError` → 422 `{ error: "UNKNOWN_STRATEGY", message, strategy, available }`;
    - `ExecutionTimeoutError` → 504 `{ error: "TIMEOUT", message }`;
    - qualquer outro → 500 `{ error: "INTERNAL", message: "Internal server error." }`;
  - **`chatErrorHandler: ErrorRequestHandler`**:
    - trata o erro de parse do body (`(error as { type?: string }).type === "entity.parse.failed"`) como 400 `{ error: "INVALID_BODY", message: "Request body is invalid.", issues: [{ code: "custom", path: [], message: "Malformed JSON body." }] }`;
    - nos demais casos, usa `toErrorResponse` e só faz `console.error` quando o status for 500 (research R5, R6).
- [ ] T016 [US3] Em `src/http/server.ts`, trocar o middleware de erro provisório de T007 por `chatErrorHandler`. Rodar `src/http/server.test.ts` inteiro e confirmar verde.

**Checkpoint**: Todos os status do contrato cobertos e testados.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T017 [P] Atualizar o `README.md` do módulo (`../README.md`) com uma seção curta sobre `POST /chat`: como subir (`npm run dev`, `PORT`), um exemplo de requisição, a tabela de status e o link para `specs/004-http-chat-endpoint/contracts/chat-http.md`.
- [ ] T018 Rodar `npm run typecheck` e corrigir os erros.
- [ ] T019 Rodar `npm test`. Toda a suíte deve passar, incluindo arena, bench e reflection, sem acesso à rede.
- [ ] T020 Validar manualmente os cenários de [quickstart.md](./quickstart.md), seção 2. Requer `.env` com credenciais; se não houver, registrar como não executado.
- [ ] T021 Revisar `spec.md` e `plan.md` caso a implementação tenha divergido; marcar as tarefas concluídas neste arquivo.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1**: vazia.
- **Phase 2**: T001 → T003; T002 pode ser escrito em paralelo com T001. Bloqueia todas as stories.
- **Phase 3 (US1)**: depende da Phase 2. T004, T005 e T006 em paralelo → T007 → T008.
- **Phase 4 (US2)**: depende de T005, T006 e T007. T009 e T010 → T011.
- **Phase 5 (US3)**: depende de T005, T006 e T007. T012 e T013 em paralelo → T014 → T015 → T016.
- **Phase 6**: depende das stories desejadas.

### User Story Dependencies

- **US4 (P2)**: entregue na Phase 2, sem dependências.
- **US1 (P1)**: depende só da Phase 2.
- **US2 (P1)** e **US3 (P1)**: dependem da US1 (servidor, service e controller existentes) e são independentes entre si.

### Parallel Opportunities

- T001 ∥ T002.
- T004 ∥ T005 ∥ T006 (arquivos diferentes).
- Depois da US1, a US2 e a US3 podem andar em paralelo, mas ambas editam `src/http/server.test.ts` (T010 e T013) e `src/services/chat.test.ts` (T009 e T012); faça essas tarefas em sequência ou combine as mudanças.
- T017 ∥ T018.

## Parallel Example: User Story 1

```text
Task: "T004 [US1] Testes 200 em src/http/server.test.ts"
Task: "T005 [US1] runChat e ChatDependencies em src/services/chat.ts"
Task: "T006 [US1] ChatRequestSchema e handler em src/controllers/chat-controller.ts"
```

## Implementation Strategy

### MVP First

1. Phase 2 (erros e registro).
2. Phase 3 (US1) → `npm test` → `npm run dev` + um `curl` com `{ "message": "..." }`.
3. Parar e validar.

### Incremental Delivery

1. MVP (US1 + US4).
2. US3 (erros), para fechar o contrato antes de expor o endpoint.
3. US2 (reflection).
4. Polish: README, gates e quickstart.

## Notes

- Nenhuma tarefa altera `src/arena.ts`, `src/bench.ts` ou as estratégias (spec, Escopo).
- Os testes nunca chamam `createModel()`: todo registro de teste é fake, e o teste de `reflect` injeta `model`.
- Commit sugerido por fase, seguindo a convenção `feature:`/`docs:` do histórico.
