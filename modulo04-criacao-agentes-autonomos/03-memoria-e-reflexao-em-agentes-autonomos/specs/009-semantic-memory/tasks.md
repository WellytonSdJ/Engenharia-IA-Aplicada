---

description: "Task list for feature implementation"
---

# Tasks: Memória Semântica

**Input**: Design documents from `/specs/009-semantic-memory/`

**Prerequisites**: plan.md, spec.md e a spec `008-conversa-persistente` implementada (`src/services/chat.ts` com histórico)

**Tests**: obrigatórios pela constituição (IV).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo
- **[Story]**: US1 (guardar sem duplicar), US2 (recall no prompt), US3 (esquecer e validar sem rede)

## Phase 0: Pre-check

- [X] T000 Confirmar `npm test` verde no projeto copiado do 02 (99 testes) antes de começar.

## Phase 1: Setup

- [X] T001 Emendar a constituição para v1.2.0: `@huggingface/transformers` como dependência de produção e `npm run test:semantic` como script oficial.
- [X] T002 `npm install @huggingface/transformers@^4.2.0`; adicionar `"test:semantic": "node --import tsx --test \"src/**/*.semantic-test.ts\""` ao `package.json`; ignorar `.cache/` e `models/` no `.gitignore`.
- [X] T003 [P] Em `src/services/errors.ts`: código `EMBEDDING`, `InvalidMemoryInputError extends ValidationError`, `EmbeddingError`.
- [X] T004 [P] Em `src/models/domain.ts`: `RecalledMemorySchema` e `RememberResultSchema`.

## Phase 2: Foundational

- [X] T005 Criar `src/services/embeddings.ts`: `EMBEDDING_DIM = 384`, `MODEL_ID`, interface `Embedder`, `getDefaultEmbedder()` lazy singleton com import dinâmico, pooling `mean`, `normalize: true`, `EmbeddingError` em falha.
- [X] T006 [P] Criar `src/services/fake-embedder.ts`: `set`, `setAxis` e fallback determinístico por hash.
- [X] T007 Criar `src/models/memory-store.ts`: `MemoryStore`, `SqliteMemoryStore` (DDL idempotente + índice por `user_id`), `dot`, `embeddingToBuffer`/`bufferToEmbedding`, constantes `DEDUP_THRESHOLD`, `RECALL_MIN_SCORE`, `RECALL_TOP_K`.

**Checkpoint**: store funcional e isolado da borda HTTP.

## Phase 3: User Story 1 - Guardar fatos sem duplicar (P1) 🎯 MVP

- [X] T008 [US1] `remember` com dedup `> 0.92` por usuário.
- [X] T009 [US1] Criar `src/controllers/memory-controller.ts` (`RememberRequestSchema`, `createRememberHandler`: 201/200) e registrar `POST /memories` em `src/http/server.ts`.
- [X] T010 [US1] [P] `src/models/memory-store.test.ts`: insere, dedup devolve id existente, dedup é por usuário, isolamento.
- [X] T011 [US1] [P] `src/http/server.test.ts`: `/memories` 201 → 200 no dedup; 400 para corpos inválidos.

## Phase 4: User Story 2 - Recall no prompt do `/chat` (P1)

- [X] T012 [US2] `recall` (pontua → filtra ≥ 0.3 → ordena → top-3), com atalho para usuário sem memórias.
- [X] T013 [US2] `src/services/chat.ts`: `memoryStore` em `ChatDependencies`, `userId` em `ChatRequest`, `formatMemoriesForPrompt`, `composeInput` com memórias antes do histórico, `metrics.recalledMemories`.
- [X] T014 [US2] `src/controllers/chat-controller.ts`: `userId` obrigatório; `InvalidMemoryInputError` → 400, `EmbeddingError` → 503.
- [X] T015 [US2] `src/agents/system-prompt.ts` (`OPSPILOT_SYSTEM_PROMPT`, `MEMORY_GUIDANCE`) aplicado ao ReAct e ao planner do Plan-and-Execute.
- [X] T016 [US2] [P] Testes: top-3 ordenado, piso 0.3, atalho sem embedder (`memory-store.test.ts`); bloco injetado e `recalledMemories` (`chat.test.ts`, `server.test.ts`); 503 quando o modelo falha.
- [X] T017 [US2] `src/models/memory-store.semantic-test.ts`: dimensão/norma do vetor real, recall sem palavras em comum, dedup com paráfrase real.

## Phase 5: User Story 3 - Esquecer e validar sem rede (P2)

- [X] T018 [US3] `forget` com `AND user_id = ?`.
- [X] T019 [US3] [P] Testes de `forget` (dono certo, outro usuário, id repetido) e de entrada vazia.
- [X] T020 [US3] Atualizar os testes existentes (`chat.test.ts`, `server.test.ts`) com `userId` e `SqliteMemoryStore(":memory:", new FakeEmbedder())`, para nenhum teste baixar o modelo.
- [X] T021 Rodar `npm run typecheck`, `npm test` e `npm run test:semantic`.

## Phase 6: Polish

- [X] T022 [P] Status da spec → Implemented.
- [X] T023 [P] Documentar em `docs/` do projeto 03.
