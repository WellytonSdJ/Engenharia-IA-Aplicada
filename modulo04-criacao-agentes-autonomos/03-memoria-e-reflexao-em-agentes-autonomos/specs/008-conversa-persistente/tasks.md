---

description: "Task list for feature implementation"
---

# Tasks: Conversa Persistente

**Input**: Design documents from `/specs/008-conversa-persistente/`

**Prerequisites**: plan.md, spec.md, e a spec `004-http-chat-endpoint` já implementada (`src/services/chat.ts`, `src/controllers/chat-controller.ts`, `src/http/server.ts` precisam existir)

**Tests**: A constituição do projeto ("IV. Testes e Qualidade Não Negociáveis") exige teste para toda mudança de lógica — as tarefas de teste abaixo não são opcionais.

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e validação incremental.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivos diferentes, sem dependência de tarefa incompleta)
- **[Story]**: US1 (continuar a mesma conversa), US2 (histórico no prompt), US3 (validar sem rede)
- Caminhos de arquivo são absolutos em relação à raiz do módulo `02-function-calling-e-tool-use/`

## Phase 0: Pre-check

- [X] T000 Confirmar que `src/services/chat.ts`, `src/controllers/chat-controller.ts` e `src/http/server.ts` já existem e que `npm test` passa antes de iniciar esta feature (pré-requisito da spec `004-http-chat-endpoint`). Se não existirem, parar aqui e implementar a spec 004 primeiro.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Criar o domínio e os erros compartilhados por toda a feature.

- [X] T001 Em `src/models/domain.ts` (ou em `src/models/conversation-store.ts`, conforme research.md decidir), adicionar `ConversationSchema = z.object({ id: z.string().min(1), createdAt: z.string() })` e `MessageSchema = z.object({ id: z.string().min(1), conversationId: z.string().min(1), role: z.enum(["user", "assistant"]), content: z.string(), createdAt: z.string() })`, com `export type Conversation`/`export type Message` via `z.infer`.
- [X] T002 [P] Em `src/services/errors.ts`, adicionar `UnknownConversationError` como subclasse de `OpsPilotError`, seguindo o mesmo padrão de `UnknownStrategyError` (da spec 004).

**Checkpoint**: tipos e erro de domínio prontos — o store pode ser implementado.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implementar o `ConversationStore` completo — bloqueante para todas as user stories.

**🚨 CRITICAL**: Nenhuma user story pode ser validada antes desta fase estar completa.

- [X] T003 Criar `src/models/conversation-store.ts` com a classe `ConversationStore`: construtor recebe `path?: string`, resolvendo para a mesma convenção de `OPSPILOT_DB` usada por `SqliteOpsStore` (ou caminho próprio, conforme research.md); usa `node:sqlite` (`DatabaseSync`); DDL idempotente (`CREATE TABLE IF NOT EXISTS`) para `conversations` e `messages`, com `FOREIGN KEY` e `CHECK (role IN ('user','assistant'))`.
- [X] T004 Em `src/models/conversation-store.ts`, implementar `async create(): Promise<Conversation>`: gera `id = `conversation-${crypto.randomUUID()}``, insere com `created_at = now`, retorna `ConversationSchema.parse(...)`.
- [X] T005 Em `src/models/conversation-store.ts`, implementar `async append(conversationId: string, messages: { role: "user" | "assistant"; content: string }[]): Promise<void>`: valida que a conversa existe (senão `UnknownConversationError`); insere cada mensagem com prepared statement, preservando a ordem recebida.
- [X] T006 Em `src/models/conversation-store.ts`, implementar `async lastMessages(conversationId: string, limit: number): Promise<Message[]>`: valida que a conversa existe (senão `UnknownConversationError`); consulta as `limit` mensagens mais recentes por `created_at`/ordem de inserção e devolve-as já ordenadas da mais antiga para a mais recente.
- [X] T007 [P] Criar `src/models/conversation-store.test.ts` (`:memory:`) cobrindo: `create` gera ids distintos em chamadas sucessivas; `append` grava na ordem correta e falha com `UnknownConversationError` para um id inexistente; `lastMessages` com menos de 12 mensagens devolve todas na ordem certa; com mais de 12, devolve exatamente as 12 mais recentes; conversa vazia devolve lista vazia; `lastMessages`/`append` para id inexistente lançam `UnknownConversationError`.
- [X] T008 Rodar `npm run typecheck` e `node --import tsx --test "src/models/conversation-store.test.ts"` e confirmar que passam antes de prosseguir para as user stories.

**Checkpoint**: `ConversationStore` completo e testado isoladamente — o service de chat pode agora consumi-lo.

---

## Phase 3: User Story 1 - Continuar a mesma conversa entre requisições (Priority: P1) 🎯 MVP

**Goal**: `POST /chat` aceita `conversationId` opcional, cria ou reutiliza a conversa e devolve o identificador na resposta.

**Independent Test**: Com estratégia fake e `ConversationStore` em memória, enviar duas mensagens reutilizando o `conversationId` da primeira resposta.

- [X] T009 [US1] Em `src/controllers/chat-controller.ts`, estender `ChatRequestSchema` (da spec 004) com `conversationId: z.string().trim().min(1).optional()`.
- [X] T010 [US1] Em `src/services/chat.ts`, estender `ChatDependencies` (da spec 004) com `conversationStore: ConversationStore`; em `runChat`, antes de executar a estratégia: se `conversationId` ausente, chamar `conversationStore.create()` e usar o id gerado; se presente, validar a existência (propagando `UnknownConversationError` se não existir).
- [X] T011 [US1] Em `src/services/chat.ts`, após a execução bem-sucedida da estratégia, chamar `conversationStore.append(conversationId, [{ role: "user", content: message }, { role: "assistant", content: answer }])`.
- [X] T012 [US1] Em `src/controllers/chat-controller.ts`, incluir `conversationId` na resposta 200, ao lado de `answer`, `trace` e `metrics`.
- [X] T013 [US1] Em `src/controllers/chat-controller.ts`, mapear `UnknownConversationError` para o mesmo padrão de erro de cliente já usado por `UnknownStrategyError` (status e corpo consistentes com a spec 004), sem executar a estratégia quando o id for inválido.
- [X] T014 [US1] [P] Em `src/http/server.test.ts`, adicionar teste de integração: `POST /chat` sem `conversationId` cria uma conversa e devolve o id; uma segunda chamada reutilizando esse id é aceita e devolve o mesmo id; uma chamada com `conversationId` inexistente recebe erro de cliente sem executar a estratégia fake (assertar que a estratégia fake não foi invocada, se o harness permitir essa contagem).

**Checkpoint**: criação e continuação de conversa funcionam de ponta a ponta — US1 coberta.

---

## Phase 4: User Story 2 - Agente recebe as últimas mensagens no prompt (Priority: P1)

**Goal**: Antes de cada execução, até 12 mensagens recentes da conversa entram no prompt, e `metrics.historyMessages` reporta o total injetado.

**Independent Test**: Popular uma conversa com N mensagens (N ≤ 12 e N > 12) e executar um turno via composição com estratégia fake.

- [X] T015 [US2] Em `src/services/chat.ts`, antes de chamar `strategy.run`, buscar `const history = await conversationStore.lastMessages(conversationId, 12)` e repassar como contexto anterior ao input da estratégia (mapeando `role: "user"` para `HumanMessage` e `role: "assistant"` para `AIMessage`, concatenadas antes da mensagem atual).
- [X] T016 [US2] Em `src/services/chat.ts`, incluir `historyMessages: history.length` no objeto `metrics` devolvido por `runChat`.
- [X] T017 [US2] [P] Em `src/services/chat.test.ts`, adicionar testes: conversa com 5 mensagens prévias → as 5 entram no input e `historyMessages === 5`; conversa com 13 mensagens prévias → apenas as 12 mais recentes entram e `historyMessages === 12`; conversa recém-criada (sem histórico) → `historyMessages === 0` e a mensagem atual ainda é processada.

**Checkpoint**: histórico é injetado corretamente e a métrica reflete o que foi usado — US2 coberta.

---

## Phase 5: User Story 3 - Validar store e chat sem rede (Priority: P2)

**Goal**: Garantir que toda a suíte desta feature (store + endpoint) roda sem rede e sem LLM.

**Independent Test**: Rodar os testes do store com `:memory:` e os testes de `/chat` com registro fake e `ConversationStore` em memória.

- [X] T018 [US3] Confirmar que `src/http/server.test.ts` usa `ConversationStore(":memory:")` nos testes desta feature (não o arquivo de produção).
- [X] T019 [US3] Em `src/services/chat.test.ts`, adicionar teste cobrindo a política de falha no meio do turno: estratégia fake que lança erro → `conversationStore.append` nunca é chamado para aquele turno (a mensagem do usuário não é persistida sozinha).
- [X] T020 [US3] Rodar `npm run typecheck` e `npm test` no módulo inteiro e confirmar que tudo passa, sem acesso à rede.

**Checkpoint**: suíte completa, determinística, sem rede — US3 coberta.

---

## Phase 6: Polish

- [X] T021 [P] Atualizar `specs/008-conversa-persistente/spec.md`, alterando **Status** de "Draft" para "Implemented".
- [X] T022 [P] Atualizar a documentação do projeto (`docs/README.md` e/ou `docs/00-START-HERE.md` de `02-function-calling-e-tool-use`) mencionando a conversa persistente e o campo `conversationId` em `/chat`.
