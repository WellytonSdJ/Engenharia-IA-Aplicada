---

description: "Task list for feature implementation"
---

# Tasks: Persistência Real de Operações

**Input**: Design documents from `/specs/005-persistencia-sqlite/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/sqlite-ops-store.md, contracts/tools.md, quickstart.md

**Tests**: A constitution do projeto ("IV. Testes e Qualidade Não Negociáveis") exige teste para toda mudança de lógica — as tarefas de teste abaixo não são opcionais.

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e validação incremental.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivos diferentes, sem dependência de tarefa incompleta)
- **[Story]**: US1 (persistência sobrevive a reinício), US2 (consultar runbook), US3 (testes/bench sem infra externa), US4 (seed idempotente)
- Caminhos de arquivo são absolutos em relação à raiz do módulo `02-function-calling-e-tool-use/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Preparar domínio, erros e configuração compartilhados por todas as user stories antes de qualquer lógica de store.

- [ ] T001 Em `src/models/domain.ts`, adicionar `RunbookSchema = z.object({ id: z.string().min(1), serviceId: z.string().min(1), title: z.string().min(1), content: z.string().min(1) })` e `export type Runbook = z.infer<typeof RunbookSchema>`, seguindo o data-model.md (relação `Service 1 --- 0..1 Runbook`, `serviceId` único).
- [ ] T002 Em `src/models/domain.ts`, estender `IncidentSchema` com `summary: z.string().nullable()` (campo novo, opcional/anulável per data-model.md) e propagar o campo para `IncidentWithServiceSchema` (via `.extend`, já herda automaticamente — apenas confirmar que o teste de contrato cobre o campo).
- [ ] T003 Remover `mysql`, `mysql2`, `sequelize` de `dependencies` e `@types/sequelize` de `devDependencies` em `package.json` (plan.md: nenhuma dependência de banco tradicional permanece); rodar `npm install` para atualizar `package-lock.json`.
- [ ] T004 Adicionar `data/` ao `.gitignore` do módulo (constitution emendada: "Diretórios de dados usados pelo arquivo SQLite devem permanecer fora do controle de versão").
- [ ] T005 [P] Remover `src/models/sequelize-store.ts`, `src/models/sequelize-store.test.ts`, `src/models/json-store.ts`, `src/models/json-store.test.ts` e `src/data/ops-store.json` (plan.md: stores tradicionais fora do projeto).

**Checkpoint**: domínio com `Runbook`/`summary`, dependências tradicionais removidas, `data/` ignorado — pronto para a Fase 2.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implementar o `SqliteOpsStore` com as 4 tabelas e o seed idempotente — bloqueante para todas as user stories, já que todas dependem do store real.

**🚨 CRITICAL**: Nenhuma user story pode ser validada antes desta fase estar completa.

- [ ] T006 Criar `src/models/sqlite-store.ts` com a classe `SqliteOpsStore` implementando `OpsStore` (interface já em `src/models/in-memory-store.ts`) mais o método novo `findRunbook(service: string): Promise<Runbook | null>`. Construtor recebe `path?: string`; resolve para `process.env.OPSPILOT_DB ?? "./data/opspilot.db"` quando omitido; usa `node:sqlite` (`DatabaseSync`). Para `path !== ":memory:"`, garantir que o diretório do arquivo existe (`node:fs` `mkdirSync(dirname(path), { recursive: true })`) antes de abrir o banco.
- [ ] T007 No construtor de `SqliteOpsStore` (`src/models/sqlite-store.ts`), executar DDL idempotente (`CREATE TABLE IF NOT EXISTS`) para as 4 tabelas conforme `data-model.md`: `services(id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, team TEXT NOT NULL, description TEXT NOT NULL)`; `alerts(id TEXT PRIMARY KEY, service_id TEXT NOT NULL REFERENCES services(id), title TEXT NOT NULL, severity TEXT NOT NULL CHECK (severity IN ('low','medium','high','critical')), status TEXT NOT NULL CHECK (status IN ('firing','resolved')), created_at TEXT NOT NULL, resolved_at TEXT)`; `incidents(id TEXT PRIMARY KEY, title TEXT NOT NULL, service_id TEXT NOT NULL REFERENCES services(id), severity TEXT NOT NULL CHECK (severity IN ('low','medium','high','critical')), status TEXT NOT NULL CHECK (status IN ('open','resolved')), created_at TEXT NOT NULL, resolved_at TEXT, summary TEXT)`; `runbooks(id TEXT PRIMARY KEY, service_id TEXT NOT NULL UNIQUE REFERENCES services(id), title TEXT NOT NULL, content TEXT NOT NULL)`.
- [ ] T008 Em `src/models/sqlite-store.ts`, implementar `listAlerts(status: AlertFilter = "firing")` com prepared statement (`db.prepare(sql).all(...)`), `JOIN` com `services`, validado por `AlertFilterSchema` na entrada e `AlertWithServiceSchema` (array) na saída; `service_id` órfão lança `ResourceNotFoundError` (contracts/sqlite-ops-store.md).
- [ ] T009 Em `src/models/sqlite-store.ts`, implementar `listIncidents(status: IncidentFilter = "open")` análogo a T008, incluindo `summary` (pode ser `null`) no retorno validado por `IncidentWithServiceSchema`.
- [ ] T010 Em `src/models/sqlite-store.ts`, implementar `createIncident({ title, service, severity })`: valida `title` não vazio e `severity` com `SeveritySchema` (`ValidationError` se inválido, sem escrita); busca serviço por nome via prepared statement; serviço não encontrado lança `ResourceNotFoundError` listando os nomes conhecidos (`Known services: ...`, mesma mensagem de `in-memory-store.ts`); insere com `id = `incident-${crypto.randomUUID()}``, `status = 'open'`, `resolved_at = NULL`, `summary = NULL`; retorna `IncidentSchema.parse(...)`.
- [ ] T011 Em `src/models/sqlite-store.ts`, implementar `resolveIncident(id)` dentro de uma única transação SQLite (`db.exec("BEGIN")`/`COMMIT`/`ROLLBACK` ou API de transação do driver): busca por `id` (não encontrado → `ResourceNotFoundError`), verifica `status === 'open'` (senão `InvalidTransitionError`, sem escrita), `UPDATE` para `resolved` com `resolved_at = now`. A transação garante que duas chamadas concorrentes sobre o mesmo id não resolvam duas vezes de forma inconsistente (FR-008).
- [ ] T012 Em `src/models/sqlite-store.ts`, implementar `findRunbook(service: string): Promise<Runbook | null>`: busca serviço por nome (não encontrado → `ResourceNotFoundError` com a mesma mensagem de serviços conhecidos); serviço encontrado sem runbook → retorna `null`; serviço com runbook → `RunbookSchema.parse(...)`.
- [ ] T013 Em `src/models/sqlite-store.ts`, implementar `reset()`: dentro de uma transação, apaga `incidents`, `alerts`, `runbooks`, `services` (ordem que respeita FKs) e recarrega o seed do cenário "Mercadinho": 6 serviços (`api`, `checkout`, `payments`, `identity`, `notifications`, `catalog` — mesmos ids/times/descrições de `DEFAULT_SERVICES` em `src/models/in-memory-store.ts`), 6 alertas (mesmos de `DEFAULT_ALERTS`: 3 `firing` — api/high, checkout/medium, payments/critical — e 3 `resolved` — api/medium, identity/high, notifications/low), e 3 runbooks (checkout, payments, identity) com título e conteúdo placeholder coerente com cada serviço. `notifications`, `api` e `catalog` ficam sem runbook.
- [ ] T014 Em `src/models/sqlite-store.ts`, implementar `readSummary()` com `COUNT(*)` por tabela/filtro (`services`, `alerts` total, `firing`, `resolved`, `incidents` total), validado por `StoreSummarySchema`.
- [ ] T015 [P] Criar `src/models/sqlite-store.test.ts` (`:memory:`) cobrindo: seed idempotente (chamar `reset()` duas vezes seguidas produz a mesma contagem via `readSummary()` — FR-010); abrir incidente válido e para serviço inexistente (mensagem lista serviços conhecidos); listar incidentes com os três filtros (`open`/`resolved`/`all`); resolver incidente aberto, resolver incidente já resolvido (`InvalidTransitionError`) e resolver id inexistente (`ResourceNotFoundError`); `findRunbook` para serviço com runbook, serviço sem runbook (retorna `null`) e serviço inexistente (`ResourceNotFoundError`); violação de `CHECK` ao tentar inserir severidade ou status fora do domínio diretamente via SQL cru no teste (confirma que o `CHECK` da tabela rejeita, não apenas a validação Zod da camada de cima — FR-007).
- [ ] T016 Rodar `npm run typecheck` e `node --import tsx --test "src/models/sqlite-store.test.ts"` e confirmar que passam antes de prosseguir para as user stories.

**Checkpoint**: `SqliteOpsStore` completo e testado isoladamente — as user stories podem agora consumi-lo.

---

## Phase 3: User Story 1 - Operações sobrevivem a reinícios (Priority: P1) 🎯 MVP

**Goal**: A composição de produção passa a usar `SqliteOpsStore`, então um incidente aberto antes de um reinício do processo continua visível depois.

**Independent Test**: Seguir `quickstart.md` ("Persistência entre execuções"): abrir um incidente em um processo `node`, listar incidentes em um segundo processo `node` novo e confirmar que o incidente aparece.

### Implementation for User Story 1

- [ ] T017 [US1] Em `src/services/tools.ts`, trocar a importação `JsonOpsStore` por `SqliteOpsStore` (`src/models/sqlite-store.js`) e `export const defaultOpsStore = new SqliteOpsStore();` (usa o default `OPSPILOT_DB`/`./data/opspilot.db` do construtor).
- [ ] T018 [US1] Em `src/services/seed.ts`, trocar `JsonOpsStore` por `SqliteOpsStore` (mesma instância/caminho usado por `defaultOpsStore`) e atualizar a mensagem de log para incluir a contagem de runbooks (ex.: `Seed ready: 6 services, 6 alerts (3 firing, 3 resolved), 3 runbooks.`), conforme `quickstart.md`.
- [ ] T019 [US1] Atualizar `src/services/tools.test.ts`: trocar a store usada nos dois testes existentes de `InMemoryOpsStore` para `new SqliteOpsStore(":memory:")` seguido de `await store.reset()` antes de cada teste, mantendo as mesmas asserções (tools validam entrada e retornam JSON conforme contrato; erros de domínio não deixam mutação parcial).
- [ ] T020 [US1] Seguir `quickstart.md` ("Persistência entre execuções"): rodar `npm run seed`, depois dois processos `node` separados (abrir incidente em um, listar em outro) e confirmar visualmente que o incidente sobrevive ao novo processo.

**Checkpoint**: incidentes abertos via tools sobrevivem a reinício do processo — MVP entregue.

---

## Phase 4: User Story 2 - Consultar runbook de um serviço (Priority: P1)

**Goal**: Nova tool `consultar_runbook(service)` expõe `findRunbook` ao agente, diferenciando serviço sem runbook de serviço inexistente.

**Independent Test**: Seguir `quickstart.md` ("Consultar runbook"): chamar a tool para `payments` (runbook encontrado), `notifications` (sem runbook, sem erro) e um serviço inexistente (erro com serviços conhecidos).

### Implementation for User Story 2

- [ ] T021 [US2] Em `src/services/tools.ts`, adicionar `ConsultarRunbookInputSchema = z.object({ service: z.string().trim().min(1).describe("Nome exato do serviço para buscar o runbook (ex.: 'checkout', 'payments', 'identity'); use o nome como aparece na lista de serviços conhecida.") })` e `ConsultarRunbookOutputSchema = z.union([RunbookSchema.pick({ title: true, content: true }).extend({ service: z.string() }), z.object({ service: z.string(), found: z.literal(false) })])`, conforme contracts/tools.md.
- [ ] T022 [US2] Em `src/services/tools.ts`, implementar a tool `consultar_runbook` dentro de `createOpsTools(store)`: chama `store.findRunbook(service)`; se `null`, retorna `{ service, found: false }` (sem lançar erro — "sem runbook" é uma observação de domínio, não uma falha); se encontrado, retorna `{ service, title, content }`; serviço inexistente propaga o `ResourceNotFoundError` da store através de `executeTool`/`ToolBoundaryError`, igual às demais tools. `description`: "Use quando o usuário pedir os passos de investigação/mitigação recomendados para um serviço (ex.: 'qual o runbook de payments?'); não serve para listar alertas ou incidentes.".
- [ ] T023 [US2] Em `src/services/tools.ts`, incluir `consultarRunbook` no array retornado por `createOpsTools` e no tipo `OpsTools`.
- [ ] T024 [P] [US2] Em `src/services/tools.test.ts`, adicionar teste cobrindo: `consultar_runbook` para `payments` retorna `found` implícito (título/conteúdo presentes); para `notifications` retorna `{ service: "notifications", found: false }` sem lançar; para serviço inexistente rejeita com `ToolBoundaryError`.
- [ ] T025 [US2] Seguir `quickstart.md` ("Consultar runbook") manualmente para confirmar os três casos (encontrado, sem runbook, serviço inexistente) end-to-end via `defaultOpsStore`.

**Checkpoint**: `consultar_runbook` funcional e testada — ambas as user stories P1 completas.

---

## Phase 5: User Story 3 - Testes e bench/arena sem infraestrutura externa (Priority: P2)

**Goal**: Confirmar que toda a suíde de testes e os cenários de bench/arena continuam rodando sem qualquer banco de dados externo, usando armazenamento efêmero isolado por execução.

**Independent Test**: Rodar `npm test` e `npm run bench` em sequência e confirmar que nenhum dos dois toca o arquivo real `data/opspilot.db` nem deixa resíduo entre execuções.

### Implementation for User Story 3

- [ ] T026 [US3] Revisar `src/controllers/bench-controller.ts` e `src/controllers/arena-controller.ts` (ou onde a store é instanciada por execução) para confirmar que continuam usando `InMemoryOpsStore` por célula/execução (plan.md: nenhuma mudança de contrato aqui) — ajustar apenas se algum import quebrou por conta da remoção de `JsonOpsStore`/`SequelizeOpsStore` na Fase 1.
- [ ] T027 [P] [US3] Rodar a suíte completa (`npm test`) em uma máquina/terminal sem qualquer processo de banco de dados ativo e confirmar 100% de sucesso (SC-002).
- [ ] T028 [US3] Rodar `npm run bench` e `npm run bench -- --scenario C2 --no-replanner` (quickstart.md) e confirmar que a tabela de resultados e os acertos permanecem iguais aos do comportamento anterior à migração (nenhuma regressão de comportamento observável nas estratégias).

**Checkpoint**: suíte de testes e avaliação continuam independentes de infraestrutura externa.

---

## Phase 6: User Story 4 - Carga inicial previsível e repetível (Priority: P2)

**Goal**: Confirmar formalmente a idempotência do seed em uso real (não só no teste unitário do store da Fase 2).

**Independent Test**: Rodar `npm run seed` duas vezes seguidas sobre o mesmo `OPSPILOT_DB` e comparar a saída/contagem.

### Implementation for User Story 4

- [ ] T029 [US4] Seguir `quickstart.md` ("Seed"): rodar `npm run seed` duas vezes seguidas e confirmar que a mensagem de saída (6 serviços, 6 alertas — 3 firing/3 resolved —, 3 runbooks) é idêntica nas duas execuções, sem duplicação (SC-004). Caso a implementação da Fase 2 (T013) já cubra isso via teste automatizado, esta tarefa é a confirmação manual/documental de que o comportamento também se sustenta via `npm run seed` real (não só `:memory:`).

**Checkpoint**: todas as quatro user stories validadas.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Consistência das descrições de tool, limpeza final e validação de documentação.

- [ ] T030 [P] Em `src/services/tools.ts`, revisar e reescrever as descrições e os schemas Zod de `list_alerts`, `open_incident`, `list_incidents` e `resolve_incident` (as 4 tools já existentes) seguindo as 6 regras de `research.md`: `description` diz quando usar; vocabulário ambíguo mapeado (já parcialmente feito em `open_incident`, estender se faltar); `.describe()` em todo campo de `ListAlertsInputSchema`, `OpenIncidentInputSchema`, `ResolveIncidentInputSchema`, `ListIncidentsInputSchema` (hoje nenhum campo tem `.describe()`); enums explicitados por extenso na descrição; nenhuma promessa de efeito colateral não garantido; erros de domínio previsíveis antecipados quando ajudam o modelo a se recuperar — usar o texto exato proposto em `contracts/tools.md`.
- [ ] T031 Atualizar `docs/tools-e-store-deterministico.md` e `docs/roteiro-de-implementacao.md` (se referenciarem `JsonOpsStore`/`SequelizeOpsStore`/MySQL) para refletir `SqliteOpsStore` como persistência real, linkando para `specs/005-persistencia-sqlite/`.
- [ ] T032 Rodar a validação completa de `quickstart.md` (typecheck, testes, seed duas vezes, persistência entre processos, consulta de runbook, bench/arena) de ponta a ponta.
- [ ] T033 Rodar `npm run typecheck` e `npm test` uma última vez sobre o estado final do branch e confirmar que ambos passam (constitution: gate não negociável antes de considerar a implementação concluída).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: sem dependências — pode começar imediatamente.
- **Foundational (Phase 2)**: depende da Fase 1 (precisa de `RunbookSchema`/`summary` em `domain.ts`) — bloqueia todas as user stories.
- **User Stories (Phase 3-6)**: todas dependem da Fase 2 completa.
  - US1 (Fase 3) e US2 (Fase 4) são P1 e independentes entre si (arquivos distintos dentro de `tools.ts`, mas US2 soma sobre o mesmo arquivo que US1 já editou em T017 — executar Fase 3 antes da Fase 4 evita conflito de edição concorrente no mesmo arquivo).
  - US3 (Fase 5) e US4 (Fase 6) são P2, validações que dependem do store (Fase 2) e, para US3, de a composição já estar trocada (Fase 3) para ter sentido prático, mas tecnicamente validam comportamento que já existia (bench/arena com `InMemoryOpsStore`) — podem rodar a qualquer momento após a Fase 2.
- **Polish (Phase 7)**: depende de todas as user stories desejadas estarem completas.

### Parallel Opportunities

- T005 (remoção de stores antigos) pode rodar em paralelo com T001/T002 (ambos em arquivos diferentes), mas depois de T003/T004 por ordem lógica de limpeza — na prática, T001-T005 tocam arquivos distintos e podem ser despachados juntos.
- T015 (teste do `SqliteOpsStore`) só pode começar depois de T006-T014 (mesma classe).
- T024 [P] (teste de `consultar_runbook`) pode rodar em paralelo com T025 (validação manual), já que T025 não edita arquivo.
- T027 [P] (rodar `npm test` sem infra externa) é independente de T028 (bench).

---

## Implementation Strategy

### MVP First (User Story 1)

1. Completar Fase 1: Setup.
2. Completar Fase 2: Foundational (`SqliteOpsStore` completo e testado) — bloqueante.
3. Completar Fase 3: User Story 1 (composição real usa SQLite).
4. **Parar e validar**: incidente sobrevive a reinício (quickstart.md).

### Incremental Delivery

1. Setup + Foundational → fundação pronta.
2. US1 → testar independentemente → persistência real funcionando (MVP).
3. US2 → testar independentemente → `consultar_runbook` funcionando.
4. US3 e US4 → confirmações de que nada quebrou fora do escopo direto da mudança.
5. Polish → descrições de tool revisadas, docs atualizadas, gates finais verdes.
