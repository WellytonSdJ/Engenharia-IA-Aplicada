# Implementation Plan: Persistência Real de Operações

**Branch**: `005-persistencia-sqlite` | **Date**: 2026-10-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-persistencia-sqlite/spec.md`

## Summary

Substituir toda a persistência "tradicional" (MySQL + Sequelize) e o store em arquivo JSON por um único `SqliteOpsStore`, baseado no módulo nativo `node:sqlite` (`DatabaseSync`), que implementa a interface `OpsStore` já existente. O store passa a cobrir quatro tabelas — `services`, `alerts`, `incidents`, `runbooks` — com DDL idempotente, `CHECK` nos campos de domínio fechado e prepared statements em toda query. `Runbook` é um conceito de domínio novo. O catálogo de tools ganha `consultar_runbook(service)` e todas as descrições existentes são revisadas (quando usar, `.describe()` por campo, enums explícitos). A composição de produção passa a injetar o `SqliteOpsStore`; `InMemoryOpsStore` continua servindo testes e bench/arena, que já criam uma instância nova por execução.

## Technical Context

**Language/Version**: Node.js 22 LTS (ambiente de execução local roda Node 24, compatível) / TypeScript 5.8, ESM, `strict: true`.

**Primary Dependencies**: `@langchain/core` (tool factory já existente), `zod` (schemas de domínio e de entrada/saída das tools), módulo nativo `node:sqlite` (`DatabaseSync`) — sem dependência de pacote externo de banco de dados.

**Storage**: SQLite via `node:sqlite`. Arquivo configurado por `OPSPILOT_DB` (default `./data/opspilot.db`); `:memory:` em testes e em bench/arena.

**Testing**: `node:test` executado via `tsx` (`npm test`), cobrindo o novo `SqliteOpsStore` e as tools existentes contra ele.

**Target Platform**: Processo Node.js único (CLI `arena`/`bench`/`seed` e, quando existir, a API Express de `src/index.ts`).

**Project Type**: Single project (backend-only), estrutura já em `src/{models,services,controllers,agents}`.

**Performance Goals**: N/A — carga de uso é de um único operador de plantão por vez; sem meta de throughput específica.

**Constraints**: Sem infraestrutura externa de banco de dados; inicialização de schema deve ser automática (sem passo manual de setup); seed deve ser idempotente.

**Scale/Scope**: 4 tabelas, ~6 serviços, ~6 alertas, poucos incidentes por sessão, 3 runbooks fixos no seed.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio/Regra | Como o plano atende |
| --- | --- |
| I. TypeScript ESM Estrito | `SqliteOpsStore` em TS ESM, `strict: true`, sem `any`; usa os tipos de `domain.ts`. |
| II. Arquitetura em Camadas | `SqliteOpsStore` vive em `src/models/` (Model); `src/services/tools.ts` e `seed.ts` continuam a única borda de regra de negócio sobre o store; nenhum Controller acessa SQL diretamente. |
| III. Contratos e Erros Explícitos | Toda leitura/escrita do store continua validada por Zod (schemas de `domain.ts`, agora incluindo `Runbook`); erros de domínio (`ResourceNotFoundError`, `InvalidTransitionError`, `ValidationError`) continuam sendo o que a store lança, traduzidos na borda das tools. |
| IV. Testes Não Negociáveis | Novo arquivo de teste para `SqliteOpsStore` (`:memory:`); `tools.test.ts` passa a usar o store SQLite `:memory:` além do `InMemoryOpsStore`; `npm run typecheck` e `npm test` verdes ao final. |
| V. Integração Segura e Simples | Não introduz nenhuma biblioteca nova — `node:sqlite` é nativo do runtime; remove duas dependências externas (Sequelize/MySQL) ao invés de adicionar; não toca em `.env`/secrets. |
| Additional Constraints (persistência) | Atende literalmente a emenda já feita na constitution (`node:sqlite`, `OPSPILOT_DB`, `:memory:`, prepared statements, `data/` fora do versionamento). |

Nenhuma violação identificada. Nenhuma entrada em Complexity Tracking é necessária.

## Project Structure

### Documentation (this feature)

```text
specs/005-persistencia-sqlite/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   ├── sqlite-ops-store.md
│   └── tools.md
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── models/
│   ├── domain.ts              # + RunbookSchema/Runbook; IncidentSchema + summary opcional
│   ├── in-memory-store.ts     # sem mudança de contrato; usado em testes e bench/arena
│   ├── in-memory-store.test.ts
│   ├── sqlite-store.ts        # NOVO: SqliteOpsStore (implementa OpsStore)
│   ├── sqlite-store.test.ts   # NOVO
│   ├── sequelize-store.ts     # REMOVIDO
│   ├── sequelize-store.test.ts # REMOVIDO
│   ├── json-store.ts          # REMOVIDO
│   └── json-store.test.ts     # REMOVIDO
├── services/
│   ├── tools.ts                # + consultar_runbook; descrições revisadas; defaultOpsStore = SqliteOpsStore
│   ├── tools.test.ts           # passa a cobrir o store SQLite ':memory:'
│   ├── seed.ts                 # passa a semear o SqliteOpsStore (serviços, alertas, runbooks)
│   └── errors.ts               # sem mudança de contrato
├── controllers/                # sem mudança (já constroem o próprio OpsStore por execução)
└── data/
    └── ops-store.json           # REMOVIDO (artefato do JsonOpsStore)

data/                            # NOVO diretório em runtime, fora do versionamento (.gitignore)
└── opspilot.db                  # criado em runtime pelo SqliteOpsStore, nunca commitado

package.json                     # remove mysql, mysql2, sequelize, @types/sequelize
.gitignore                       # + data/
```

**Structure Decision**: Mantém a arquitetura em camadas já adotada (Model/Service/Controller). O novo store entra em `src/models/` ao lado de `in-memory-store.ts`, como mais uma implementação de `OpsStore`; nenhuma nova camada é criada. `src/data/ops-store.json` é removido porque era o arquivo de dados do `JsonOpsStore`, que deixa de existir.

## Complexity Tracking

*Sem violações a justificar.*
