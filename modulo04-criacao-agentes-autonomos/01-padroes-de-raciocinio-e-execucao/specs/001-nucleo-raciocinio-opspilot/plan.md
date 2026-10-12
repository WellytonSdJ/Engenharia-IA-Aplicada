# Implementation Plan: Núcleo de raciocínio do OpsPilot

**Branch**: `001-nucleo-raciocinio-opspilot` | **Date**: 2026-09-28 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-nucleo-raciocinio-opspilot/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command; its definition describes the execution workflow.

## Summary

Implementar um núcleo de raciocínio comparável para o OpsPilot. O núcleo terá um contrato comum para estratégias, eventos de trace e métricas; uma fábrica única de `ChatOpenAI` configurada para OpenRouter; ferramentas operacionais sobre um arquivo JSON persistente; estratégias ReAct e Plan-and-Execute; uma arena CLI; seed reproduzível; e testes determinísticos sem rede.

O store JSON será injetado nas ferramentas como fonte de dados do runtime, permitindo que o modelo leia alertas/incidentes atuais e persista novos incidentes. Testes usarão caminhos temporários ou stores em memória isolados. Plan-and-Execute terá estado `input`, `plan`, `done` e `answer`; o replanner decidirá continuar, ajustar ou finalizar. O limite de oito passos será aplicado ao trabalho acumulado, incluindo ajustes.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: TypeScript 5.8 em Node.js 22 LTS, ESM com `module`/`moduleResolution` `NodeNext` e `strict: true`.

**Primary Dependencies**: `@langchain/core` 0.3.x, `@langchain/langgraph` 0.2.x, `@langchain/openai` 0.3.x, Zod 3.x, Express 5.x e APIs de filesystem nativas do Node.js.

**Storage**: arquivo JSON em `src/data/ops-store.json` como fonte única operacional no runtime; adapters em memória e arquivos temporários para testes isolados.

**Testing**: `node:test` executado por `tsx` com `npm test`; typecheck por `npm run typecheck`. Testes do store e do formatador de trace não fazem rede.

**Target Platform**: Node.js 22 em ambiente local/servidor, com CLI e API Express existentes como pontos de integração.

**Project Type**: Serviço Node.js com biblioteca interna de agentes e entradas CLI para arena, seed e benchmark.

**Performance Goals**: Execuções locais sem chamadas redundantes além das necessárias para o grafo; cada resultado deve informar `llmCalls` e `latencyMs`. A arena deve manter resultados separados e reproduzíveis no modo in-memory.

**Constraints**: Temperatura zero; leitura exclusiva de `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` pela fábrica; nenhum segredo em logs ou versionamento; máximo de 8 passos acumulados no planner mesmo após replans; limite de iterações configurável; testes determinísticos sem rede; gravação atômica do arquivo JSON.

**Scale/Scope**: 5 serviços e 6 alertas no seed; 3 alertas `firing` e 3 `resolved`; quatro ferramentas operacionais; duas estratégias; uma arena mínima; testes focados no store e no trace.

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

- **I. TypeScript ESM Estrito**: PASS. O plano preserva TypeScript ESM, Node.js 22, `strict: true` e os scripts existentes.
- **II. Arquitetura em Camadas**: PASS. Models/stores, services/estratégias e controllers/CLI serão separados; efeitos de LLM e filesystem ficarão nas bordas.
- **III. Contratos e Erros Explícitos**: PASS. Zod validará ferramentas, trace, métricas, planner e flags da arena; erros de domínio serão traduzidos nas bordas.
- **IV. Testes e Qualidade Não Negociáveis**: PASS. Testes serão adicionados junto à lógica e o plano exige `npm run typecheck` e `npm test`.
- **V. Integração Segura e Simples**: PASS. A fábrica centraliza OpenRouter, o `.env` não será lido diretamente pelo agente e a persistência usa APIs nativas sem dependências novas.
- **Gates de workflow**: PASS. A especificação foi criada e revisada antes deste plano; o plano deverá ser revisado antes de `/speckit.tasks`.

## Project Structure

### Documentation (this feature)

```text
specs/001-nucleo-raciocinio-opspilot/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/
├── agents/
│   ├── model.ts
│   ├── react.ts
│   ├── plan-and-execute.ts
│   ├── strategy.ts
│   └── trace.ts
├── controllers/
│   └── arena-controller.ts
├── models/
│   ├── domain.ts
│   ├── in-memory-store.ts
│   └── json-store.ts
├── services/
│   ├── tools.ts
│   ├── seed.ts
│   └── errors.ts
├── arena.ts
├── index.ts
├── bench.ts
└── data/
    └── ops-store.json

tests/
└── (existing test convention: `src/**/*.test.ts`)
```

**Structure Decision**: Single TypeScript project. O código novo ficará em `src/` organizado por responsabilidade: contratos e estratégias em `agents/`, domínio e persistência em `models/`, ferramentas/seed/erros em `services/`, e a arena como entrypoint CLI. Os testes permanecerão próximos aos módulos em arquivos `src/**/*.test.ts`, conforme o script existente.

## Phase 0: Research Decisions

- Confirmar a API instalada do `createReactAgent` para conectar `ChatOpenAI` e tools e obter mensagens/valores do grafo para o trace.
- Usar `StateGraph` com estado tipado `input`, `plan`, `done`, `answer`; cada execução acrescenta `[step, result]` ao canal `done`, e o replanner devolve decisão `continue`, `adjust` ou `finish`. O limite é calculado como `done.length + plan.length <= 8`, também após ajustes.
- Usar `ChatOpenAI` com `configuration.baseURL` apontando para `https://openrouter.ai/api/v1`, `apiKey` de `OPENROUTER_API_KEY`, `model` de `OPENROUTER_MODEL` e `temperature: 0`.
- Manter `JsonOpsStore` como adapter padrão injetado nas tools; carregar/validar o documento JSON por operação e gravar alterações via arquivo temporário e rename atômico. Testes usam caminhos temporários e não editam `src/data/ops-store.json`.

## Phase 1: Design Outputs

- [research.md](./research.md): decisões e alternativas sobre LangGraph, OpenRouter, store e Sequelize.
- [data-model.md](./data-model.md): entidades, schemas Zod, estados e transições.
- [contracts/strategy.md](./contracts/strategy.md): contrato comum das estratégias, trace e métricas.
- [contracts/tools.md](./contracts/tools.md): contratos das ferramentas e do store.
- [contracts/arena-cli.md](./contracts/arena-cli.md): flags e formato de saída da arena.
- [quickstart.md](./quickstart.md): comandos de validação após a implementação.

## Constitution Check - Post-Design

- **Status**: PASS.
- Os artefatos de design preservam TypeScript ESM estrito, MVC/serviços separados, contratos Zod, testes sem rede, fábrica única do modelo e proteção de secrets. O caminho de dados ativo usa apenas JSON, sem conexão MySQL.
- Não há violações ou decisões pendentes que impeçam a decomposição em tarefas.

## Complexity Tracking

Não há violações da constituição que exijam justificativa. A separação entre store in-memory e adapter Sequelize é uma fronteira explícita de teste/persistência, não um projeto adicional.
