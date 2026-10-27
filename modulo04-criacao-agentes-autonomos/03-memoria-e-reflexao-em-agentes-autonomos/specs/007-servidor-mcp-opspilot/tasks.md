---

description: "Task list for feature implementation"
---

# Tasks: Servidor MCP do OpsPilot

**Input**: Design documents from `/specs/007-servidor-mcp-opspilot/`

**Prerequisites**: plan.md, spec.md

**Tests**: A constituição do projeto ("IV. Testes e Qualidade Não Negociáveis") exige teste para toda mudança de lógica — as tarefas de teste abaixo não são opcionais.

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e validação incremental.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivos diferentes, sem dependência de tarefa incompleta)
- **[Story]**: US1 (descoberta de tools), US2 (mesmas operações via MCP), US3 (stdio seguro), US4 (subir com um comando)
- Caminhos de arquivo são absolutos em relação à raiz do módulo `02-function-calling-e-tool-use/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Adicionar a dependência do SDK MCP e preparar a extração de definições de tool compartilhadas.

- [X] T001 Adicionar `@modelcontextprotocol/sdk` ao `package.json` (dependencies) e rodar `npm install`.
- [X] T002 Em `src/services/tools.ts`, extrair `createOpsToolDefinitions(store: OpsStore, options: { fetchFn?: typeof fetch } = {})`, retornando um array de objetos `{ name, description, inputSchema, run }` — um por tool já existente (`list_alerts`, `open_incident`, `resolve_incident`, `list_incidents`, `consultar_runbook`, `check_provider_status`), sem alterar a lógica interna de cada `run`.
- [X] T003 Em `src/services/tools.ts`, reescrever `createOpsTools(store, options)` para consumir `createOpsToolDefinitions(store, options)` e mapear cada definição para `tool(definition.run, { name: definition.name, description: definition.description, schema: definition.inputSchema })`, preservando o array e a ordem já retornados hoje.
- [X] T004 Rodar `npm run typecheck` e `node --import tsx --test "src/services/tools.test.ts"` e confirmar que a suíte existente continua passando sem alteração de comportamento após a extração.

**Checkpoint**: `createOpsToolDefinitions` disponível como fonte única de schemas/handlers — o servidor MCP pode agora consumi-la.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Criar o servidor MCP mínimo com as três tools operacionais — bloqueante para todas as user stories.

**🚨 CRITICAL**: Nenhuma user story pode ser validada antes desta fase estar completa.

- [X] T005 Criar `src/mcp/server.ts` com `export function createMcpServer(store: OpsStore)`: instancia `McpServer` (ou classe equivalente do SDK) com `{ name: "opspilot", version: "1.0.0" }`; obtém `createOpsToolDefinitions(store)` e filtra apenas `list_alerts`, `open_incident`, `resolve_incident`; para cada uma, registra a tool no servidor com `inputSchema` e um handler que chama `run(...)` e devolve o resultado no formato de conteúdo de texto esperado pelo SDK.
- [X] T006 Em `src/mcp/server.ts`, garantir que nenhuma linha usa `console.log` (ou qualquer escrita em stdout fora do transporte do SDK) — qualquer diagnóstico usa `console.error`.
- [X] T007 Em `src/mcp/server.ts`, criar `export async function startStdioServer()`: instancia o `SqliteOpsStore` de produção, chama `createMcpServer`, conecta via `StdioServerTransport` do SDK; captura falha de bootstrap (ex.: erro ao abrir o store) e registra em `console.error` antes de finalizar o processo.
- [X] T008 Em `src/mcp/server.ts`, adicionar o guard de entrypoint (`import.meta.url === pathToFileURL(process.argv[1]).href` ou padrão equivalente já usado em `src/arena.ts`/`src/bench.ts`) chamando `startStdioServer()` apenas quando o arquivo é executado diretamente, não quando importado em teste.
- [X] T009 [P] Criar `src/mcp/server.test.ts`: instanciar `createMcpServer` com um `SqliteOpsStore(":memory:")` seedado (`reset()`); listar as tools registradas e assertar exatamente `["list_alerts", "open_incident", "resolve_incident"]`; assertar o nome do servidor igual a `opspilot`.
- [X] T010 Rodar `npm run typecheck` e `node --import tsx --test "src/mcp/server.test.ts"` e confirmar que passam antes de prosseguir para as user stories.

**Checkpoint**: servidor MCP mínimo funcionando e testado — as user stories podem agora ser validadas sobre ele.

---

## Phase 3: User Story 1 - Cliente MCP descobre as ferramentas operacionais (Priority: P1) 🎯 MVP

**Goal**: Confirmar, com teste explícito, que a listagem de tools e a identidade do servidor atendem ao contrato da US1.

**Independent Test**: Instanciar o servidor em teste, listar as tools e verificar nome do servidor e catálogo exato.

- [X] T011 [US1] Confirmar (ou completar, se faltar) em `src/mcp/server.test.ts` a asserção de que a listagem não contém tools fora do catálogo desta feature (`list_incidents`, `consultar_runbook`, `check_provider_status` NÃO aparecem).

**Checkpoint**: US1 coberta por teste explícito.

---

## Phase 4: User Story 2 - Mesmas operações do plantão via MCP (Priority: P1)

**Goal**: Garantir que invocar as tools via MCP produz o mesmo efeito observável que as tools do agente sobre o mesmo store.

**Independent Test**: Com store seedado, invocar as três operações via handlers MCP e comparar com as tools do agente sobre o mesmo store.

- [X] T012 [US2] Em `src/mcp/server.test.ts`, adicionar teste que invoca o handler de `list_alerts` registrado no servidor e compara o resultado com `createOpsTools(store)[0].invoke(...)` sobre o mesmo store seedado (mesmo conteúdo).
- [X] T013 [US2] Em `src/mcp/server.test.ts`, adicionar teste que invoca o handler de `open_incident` com título/serviço/severidade válidos e confirma, consultando o store diretamente (`store.listIncidents("all")`), que o incidente foi criado com o status esperado.
- [X] T014 [US2] Em `src/mcp/server.test.ts`, adicionar teste que invoca o handler de `resolve_incident` para um incidente aberto conhecido e confirma a transição para resolvido no store; e outro teste que invoca com um ID inexistente, confirmando que o handler devolve texto de erro legível sem lançar exceção e sem derrubar o processo de teste.

**Checkpoint**: US2 coberta — paridade comportamental entre MCP e as tools do agente comprovada por teste.

---

## Phase 5: User Story 3 - Stdio seguro para diagnóstico (Priority: P2)

**Goal**: Garantir que nenhuma escrita acidental em stdout comprometa o canal do protocolo.

**Independent Test**: Revisar o código do servidor por escrita em stdout fora do transporte.

- [X] T015 [P] [US3] Adicionar um teste simples (ou verificação em `server.test.ts`) que lê o conteúdo-fonte de `src/mcp/server.ts` e falha se encontrar a substring `console.log`.

**Checkpoint**: US3 coberta — regra crítica de stdio protegida por teste automatizado.

---

## Phase 6: User Story 4 - Subir o servidor com um comando (Priority: P3)

**Goal**: Permitir subir o servidor localmente com um único comando npm.

**Independent Test**: Inspecionar `package.json` e confirmar o script e o entrypoint.

- [X] T016 [US4] Adicionar o script `"mcp": "node --env-file-if-exists=.env --import tsx src/mcp/server.ts"` ao `package.json`.
- [X] T017 [US4] Rodar `npm run mcp` manualmente (com um `.env` válido ou sem variáveis, usando o default de `OPSPILOT_DB`) e confirmar que o processo inicia sem exceção e aguarda conexão stdio (interromper manualmente após confirmar).

**Checkpoint**: US4 coberta — servidor sobe com um comando documentado.

---

## Phase 7: Polish

- [X] T018 [P] Rodar `npm run typecheck` e `npm test` no módulo inteiro e confirmar que tudo passa, incluindo a suíte nova de `src/mcp/`.
- [X] T019 [P] Atualizar `specs/007-servidor-mcp-opspilot/spec.md`, alterando **Status** de "Draft" para "Implemented".
- [X] T020 [P] Atualizar a documentação do projeto (`docs/README.md` e/ou `docs/00-START-HERE.md` de `02-function-calling-e-tool-use`) mencionando o servidor MCP e o comando `npm run mcp`.
