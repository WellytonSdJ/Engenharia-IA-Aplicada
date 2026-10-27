---

description: "Task list for feature implementation"
---

# Tasks: Refletor de Aprendizado

**Input**: Design documents from `/specs/010-learning-reflector/`

**Prerequisites**: plan.md, spec.md e a spec `009-semantic-memory` implementada (`MemoryStore`, `userId` no `/chat`)

**Tests**: obrigatórios pela constituição (IV).

## Format: `[ID] [P?] [Story] Description`

- **[Story]**: US1 (aprender após o turno), US2 (só fatos duráveis), US3 (esquecer via tool)

## Phase 1: Foundational

- [X] T001 Criar `src/services/learning-reflector.ts`: `LEARNING_REFLECTOR_PROMPT`, `LearningReflectionSchema`, `LearningReflectorFn`, `createLLMLearningReflector(modelFactory)` com fallback "sem aprendizado", `scheduleLearning({ reflector, memoryStore, userId, userMessage })` que nunca lança.

## Phase 2: User Story 1 - Aprender após o turno (P1) 🎯 MVP

- [X] T002 [US1] `src/services/chat.ts`: `learningReflector?` em `ChatDependencies`; depois do `append`, `void scheduleLearning(...)`.
- [X] T003 [US1] `src/index.ts`: `startServer(port, { learningReflector: createLLMLearningReflector(createModel) })`.
- [X] T004 [US1] [P] `chat.test.ts`: `runChat` resolve com o `remember` ainda travado; refletor não roda quando a estratégia falha.
- [X] T005 [US1] [P] `learning-reflector.test.ts`: fato positivo grava uma vez (com trim); fato aprendido volta no recall do `SqliteMemoryStore`.

## Phase 3: User Story 2 - Só fatos duráveis (P1)

- [X] T006 [US2] [P] `learning-reflector.test.ts`: pontual, segredo e `hasLearning: true` com fato vazio não gravam; refletor/`remember` com erro não lançam; prompt cita DURÁVEIS, PONTUAIS e SEGREDOS.
- [X] T007 [US2] [P] `learning-reflector.test.ts`: `createLLMLearningReflector` envia prompt + mensagem e cai em "sem aprendizado" quando o modelo falha.

## Phase 4: User Story 3 - Esquecer via tool (P2)

- [X] T008 [US3] `src/agents/strategy.ts`: `ReasoningOptions.memory?: { store, userId }`; `src/services/chat.ts` passa `memory` em toda execução.
- [X] T009 [US3] `src/services/tools.ts`: `ForgetPreferenceInputSchema` (só `query`), `createForgetPreferenceToolDefinition(store, userId)` (recall top-1 + forget) e `createForgetPreferenceTool`.
- [X] T010 [US3] `src/agents/react.ts`: adiciona a tool quando `options.memory` existe.
- [X] T011 [US3] [P] `tools.test.ts`: apaga só a memória mais parecida do usuário certo; sem match não altera nada; schema sem `userId`.
- [X] T012 [US3] [P] `react.test.ts`: tool presente só com `memory`.

## Phase 5: Polish

- [X] T013 Rodar `npm run typecheck` e `npm test`.
- [X] T014 [P] Status da spec → Implemented; documentar em `docs/` do projeto 03.
