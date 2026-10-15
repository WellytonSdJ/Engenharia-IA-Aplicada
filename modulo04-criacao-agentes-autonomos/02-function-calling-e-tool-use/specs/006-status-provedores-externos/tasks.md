---

description: "Task list for feature implementation"
---

# Tasks: Status de Provedores Externos

**Input**: Design documents from `/specs/006-status-provedores-externos/`

**Prerequisites**: plan.md, spec.md

**Tests**: A constituição do projeto ("IV. Testes e Qualidade Não Negociáveis") exige teste para toda mudança de lógica — as tarefas de teste abaixo não são opcionais.

**Organization**: Tarefas agrupadas por user story (spec.md) para permitir implementação e validação incremental.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: pode rodar em paralelo (arquivos diferentes, sem dependência de tarefa incompleta)
- **[Story]**: US1 (distinguir falha interna de externa), US2 (falha externa como observação), US3 (testes sem rede)
- Caminhos de arquivo são absolutos em relação à raiz do módulo `02-function-calling-e-tool-use/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Preparar os schemas e o mapa de URLs compartilhados por toda a feature antes de qualquer lógica de rede.

- [X] T001 Em `src/services/tools.ts`, adicionar `CheckProviderStatusInputSchema = z.object({ provider: z.enum(["github", "cloudflare"]).default("github").describe("Provedor externo a consultar: github ou cloudflare. Default: github.") })`.
- [X] T002 Em `src/services/tools.ts`, adicionar `ProviderStatusResponseSchema = z.object({ status: z.object({ indicator: z.string(), description: z.string() }) }).passthrough()` (aceita campos extras no JSON, valida só os obrigatórios).
- [X] T003 [P] Em `src/services/tools.ts`, declarar o mapa `const PROVIDER_STATUS_URLS: Record<"github" | "cloudflare", string> = { github: "https://www.githubstatus.com/api/v2/status.json", cloudflare: "https://www.cloudflarestatus.com/api/v2/status.json" }`.

**Checkpoint**: schemas e mapa de URLs prontos — a lógica de rede pode ser implementada.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implementar a função de consulta com timeout e retentativa — bloqueante para todas as user stories, já que a tool depende diretamente dela.

**🚨 CRITICAL**: Nenhuma user story pode ser validada antes desta fase estar completa.

- [X] T004 Em `src/services/tools.ts`, implementar `async function fetchProviderStatus(provider: "github" | "cloudflare", fetchFn: typeof fetch): Promise<string>` que: monta a URL via `PROVIDER_STATUS_URLS[provider]`; faz até duas tentativas; cada tentativa chama `fetchFn(url, { signal: AbortSignal.timeout(5000) })`; se a resposta não for `ok`, trata `status >= 500` como elegível a retentativa e `status` 4xx como falha final imediata (sem segunda tentativa); erro de rede/abort (exceção lançada pelo `fetchFn`) é elegível a retentativa; valida o corpo com `ProviderStatusResponseSchema.safeParse`, payload inválido é falha final (sem retentativa adicional além da já feita por outro motivo); retorna, em sucesso, a string `` `${indicator} — ${description}` ``.
- [X] T005 Em `fetchProviderStatus` (`src/services/tools.ts`), garantir que qualquer falha final (segunda tentativa esgotada, 4xx, payload inválido) resulte em uma string de erro legível como valor de retorno — a função nunca lança exceção para quem a chama (contrato interno que a tool depende para cumprir FR-010).
- [X] T006 [P] Criar a suíte de testes de `fetchProviderStatus`/`check_provider_status` em `src/services/tools.test.ts`, cobrindo: sucesso com payload válido do GitHub (retorno compacto); `provider: "cloudflare"` consulta a URL correta; `provider` omitido usa `github`; timeout (fetch fake que sempre aborta) → erro legível após uma retentativa, sem exceção; 500 na primeira tentativa seguido de 200 válido na segunda → retorna o status da segunda tentativa; 404 → erro legível imediato, sem segunda chamada (assertar número de chamadas ao fetch fake); payload com `status` ausente ou com tipos errados → erro legível, sem lançar exceção.
- [X] T007 Rodar `npm run typecheck` e `node --import tsx --test "src/services/tools.test.ts"` e confirmar que passam antes de prosseguir para as user stories.

**Checkpoint**: `fetchProviderStatus` completo e testado isoladamente — a tool pode agora ser registrada no catálogo.

---

## Phase 3: User Story 1 - Distinguir falha interna de falha do provedor (Priority: P1) 🎯 MVP

**Goal**: Registrar `check_provider_status` no catálogo de tools, retornando o status compacto do provedor.

**Independent Test**: Invocar a tool com fetch fake retornando payload válido; verificar retorno compacto, sem rede real.

- [X] T008 [US1] Em `src/services/tools.ts`, alterar a assinatura de `createOpsTools` para `createOpsTools(store: OpsStore, options: { fetchFn?: typeof fetch } = {})`, com `const fetchFn = options.fetchFn ?? fetch`.
- [X] T009 [US1] Em `createOpsTools` (`src/services/tools.ts`), criar a tool `checkProviderStatus` com `tool(async ({ provider }) => fetchProviderStatus(provider, fetchFn), { name: "check_provider_status", description: "Use quando houver suspeita de problema externo, dúvida se a causa é nossa ou do provedor (GitHub ou Cloudflare), ou dependência externa possivelmente fora do ar. Não altera nenhum dado. Consulta a página pública de status do provedor e retorna indicador + descrição oficiais em uma linha. Em caso de falha na consulta (timeout, indisponibilidade ou resposta inesperada), retorna uma mensagem de erro legível em vez de travar.", schema: CheckProviderStatusInputSchema })`.
- [X] T010 [US1] Em `createOpsTools` (`src/services/tools.ts`), incluir `checkProviderStatus` no array retornado, ao lado das demais tools (`listAlerts`, `openIncident`, `resolveIncident`, `listIncidents`, `consultarRunbook`).
- [X] T011 [US1] Rodar `node --import tsx --test "src/services/tools.test.ts"` e confirmar que a tool registrada no catálogo se comporta como a função testada na Fase 2 (teste de integração simples: instanciar `createOpsTools` com fetch fake e invocar a tool pelo nome).

**Checkpoint**: a tool está disponível no catálogo e responde corretamente a consultas de status, cobrindo US1.

---

## Phase 4: User Story 2 - Falha externa não quebra o raciocínio (Priority: P1)

**Goal**: Garantir, com testes focados, que toda falha (timeout, 5xx, 4xx, payload inválido) chega ao agente como observação em texto, nunca como exceção que interrompe a execução.

**Independent Test**: Simular timeout, 5xx e payload inválido com fetch fake; verificar que a tool sempre retorna texto e nenhuma exceção escapa dela.

- [X] T012 [US2] Confirmar (ou adicionar, se faltar) em `src/services/tools.test.ts` um teste que invoca a tool `check_provider_status` (via `createOpsTools`, não só a função interna) nos três cenários de falha — timeout, 5xx persistente e payload inválido — e assertar que o `await tool.invoke(...)` resolve normalmente (sem lançar), com o texto de erro como resultado.
- [X] T013 [US2] Revisar a descrição da tool (`src/services/tools.ts`, T009) para confirmar que ela já comunica ao modelo que o resultado pode ser uma mensagem de indisponibilidade, não apenas um status saudável — ajustar o texto se necessário.

**Checkpoint**: falhas externas comprovadamente não propagam exceção para fora da tool — US2 coberta.

---

## Phase 5: User Story 3 - Testar sem rede (Priority: P2)

**Goal**: Confirmar que toda a suíte desta feature roda sem acesso à rede real.

**Independent Test**: Rodar `npm test` numa máquina sem acesso à internet (ou com a rede desligada) e confirmar que a suíte de `check_provider_status` passa.

- [X] T014 [US3] Revisar `src/services/tools.test.ts` e confirmar que nenhum teste desta feature usa o `fetch` global real — todos passam `fetchFn` fake explicitamente para `createOpsTools`.
- [X] T015 [US3] Rodar `npm run typecheck` e `npm test` no módulo e confirmar que tudo passa.

**Checkpoint**: suíte completa, determinística, sem rede — US3 coberta.

---

## Phase 6: Polish

- [X] T016 [P] Atualizar `specs/006-status-provedores-externos/spec.md`, alterando **Status** de "Draft" para "Implemented".
- [X] T017 [P] Atualizar a documentação do projeto (`docs/README.md` e/ou `docs/00-START-HERE.md` de `02-function-calling-e-tool-use`) mencionando a tool `check_provider_status` como capacidade disponível.
