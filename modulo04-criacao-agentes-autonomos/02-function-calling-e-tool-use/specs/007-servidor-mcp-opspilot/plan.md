# Implementation Plan: Servidor MCP do OpsPilot

**Branch**: `007-servidor-mcp-opspilot` | **Date**: 2026-10-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/007-servidor-mcp-opspilot/spec.md`

## Summary

Expor `list_alerts`, `open_incident` e `resolve_incident` por um servidor MCP próprio (`src/mcp/server.ts`), usando o SDK oficial `@modelcontextprotocol/sdk` com transporte stdio. O servidor reutiliza a mesma instância de `SqliteOpsStore` e os mesmos schemas Zod já definidos em `src/services/tools.ts` — para isso, exporta-se de `tools.ts` uma função `createOpsToolDefinitions(store)` que devolve, para cada tool, `{ name, description, inputSchema, handler }` em um formato agnóstico de framework, consumido tanto pela fábrica `createOpsTools` (LangChain `tool()`) quanto pelo novo servidor MCP (`server.registerTool(...)`). Nenhuma duplicação de schema. O código do servidor nunca escreve em stdout fora do transporte — toda mensagem de diagnóstico vai para stderr via `console.error`.

## Technical Context

**Language/Version**: TypeScript ESM, `strict: true`, Node.js 22 LTS

**Primary Dependencies**: Nova: `@modelcontextprotocol/sdk` (servidor MCP, transporte stdio). Já existentes: `zod` (schemas compartilhados), `SqliteOpsStore`.

**Storage**: Reutiliza o `SqliteOpsStore` já implementado (spec 005); mesma variável `OPSPILOT_DB` / default `./data/opspilot.db`. Testes usam `:memory:`.

**Testing**: `node:test` via `tsx`; teste do servidor instanciando-o em processo (sem subir stdio real) para validar a listagem de tools e invocar os três handlers diretamente contra um store `:memory:` seedado.

**Target Platform**: Processo Node.js separado, iniciado via `npm run mcp` (stdio local).

**Project Type**: Novo entrypoint dentro do projeto único existente — pasta nova `src/mcp/`.

**Performance Goals**: N/A — uso interativo local via stdio, sem meta de throughput.

**Constraints**: Zero `console.log` (ou qualquer escrita em stdout fora do transporte) em todo o código de `src/mcp/`; nenhum schema duplicado — os três handlers MCP chamam exatamente a mesma lógica usada pelas tools do agente.

**Scale/Scope**: Um novo módulo (`src/mcp/server.ts`), uma pequena refatoração de extração em `src/services/tools.ts` (sem mudar comportamento das tools existentes), um script npm novo e a suíte de testes do servidor.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio/Regra | Como o plano atende |
| --- | --- |
| I. TypeScript ESM Estrito | Servidor em TS ESM, `strict: true`, sem `any`; tipos dos handlers derivados dos schemas Zod existentes via `z.infer`. |
| II. Arquitetura em Camadas | `src/mcp/server.ts` é um entrypoint fino, análogo a `src/arena.ts`/`src/bench.ts`: monta o store, pede as definições de tool a `src/services/tools.ts` (Service) e as registra no SDK MCP — nenhuma regra de negócio nova nessa camada. |
| III. Contratos e Erros Explícitos | Os handlers MCP devolvem a mesma string de observação que as tools do agente devolvem hoje em caso de erro de domínio (nenhuma tradução nova de erro é necessária, pois `executeTool` já converte erros de domínio em texto antes de chegar ao handler). |
| IV. Testes Não Negociáveis | Teste dedicado valida: nome do servidor (`opspilot`), catálogo exato de tools (as três, nem mais nem menos) e que invocar cada handler contra um store seedado produz o mesmo resultado que a tool equivalente do agente. |

**Post-design re-check**: **PASS**. A extração de `createOpsToolDefinitions` não altera o comportamento observável de `createOpsTools` (verificado pela suíte existente de `tools.test.ts` continuando verde).

## Project Structure

### Documentation (this feature)

```text
specs/007-servidor-mcp-opspilot/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── mcp-server.md
└── tasks.md             # gerado por /speckit.tasks
```

### Source Code (repository root do projeto 02)

```text
src/
├── services/
│   ├── tools.ts           # ALTERADO: extrai createOpsToolDefinitions(store), consumido por createOpsTools
│   └── tools.test.ts       # inalterado em comportamento; eventual ajuste de import
└── mcp/
    ├── server.ts           # NOVO: createMcpServer(store), startStdioServer()
    └── server.test.ts      # NOVO: nome do servidor, catálogo de tools, invocação dos três handlers
```

**Structure Decision**: Pasta nova `src/mcp/` para o entrypoint do servidor, paralela a `src/http/` (quando existir, spec 004) e a `src/arena.ts`/`src/bench.ts`. A lógica de negócio não migra para lá — apenas o código específico do protocolo MCP (registro de tools, transporte stdio).

## Decisões e rastreabilidade

| Decisão | Atende |
| --- | --- |
| `createOpsToolDefinitions(store, options?)` em `src/services/tools.ts` retorna um array de `{ name, description, inputSchema, run }`, onde `run` é a mesma função assíncrona hoje passada a `tool(...)`; `createOpsTools` passa a mapear esse array para `tool()` do LangChain, sem duplicar lógica | FR-004, FR-005 (fonte única) |
| O servidor MCP usa só as três primeiras definições do catálogo (`list_alerts`, `open_incident`, `resolve_incident`) — filtra por nome explicitamente, não por posição no array, para não depender da ordem | FR-003, FR-010 |
| `createMcpServer(store)` em `src/mcp/server.ts`: instancia `McpServer` do SDK com `{ name: "opspilot", version: "1.0.0" }`; para cada definição filtrada, chama `server.registerTool(name, { description, inputSchema }, handler)`, onde o handler chama `run(...)` e envolve o resultado no formato de conteúdo esperado pelo SDK (`{ content: [{ type: "text", text: resultado }] }`) | FR-001, FR-002, FR-003 |
| `startStdioServer()`: cria o `SqliteOpsStore` de produção, chama `createMcpServer`, conecta via `StdioServerTransport` do SDK; todo log de bootstrap/erro usa `console.error` | FR-006, FR-007 |
| Script npm `"mcp": "node --env-file-if-exists=.env --import tsx src/mcp/server.ts"`, com guard `pathToFileURL` (mesmo padrão de `arena.ts`/`bench.ts`) para não iniciar o servidor quando o módulo for importado em teste | FR-007 |
| Dependência `@modelcontextprotocol/sdk` adicionada ao `package.json` | FR-009 |
| Erros de domínio dentro de um handler MCP (ex.: serviço inexistente em `open_incident`) continuam virando texto de observação (mesma função `executeTool` já usada), nunca uma exceção MCP não tratada | Edge case "entrada falha na validação", "resolve_incident com ID inexistente" |

## Testes

- `src/mcp/server.test.ts`: instanciar `createMcpServer` com um `SqliteOpsStore(":memory:")` seedado; listar as tools registradas (via API de teste do SDK ou inspecionando o registro interno) e assertar exatamente `["list_alerts", "open_incident", "resolve_incident"]`; assertar `server.name === "opspilot"` (ou equivalente exposto pelo SDK); invocar o handler de `list_alerts` e comparar com o resultado de `createOpsTools(store)[0].invoke(...)` sobre o mesmo store; invocar `open_incident` com dados válidos e confirmar o incidente criado no store; invocar `resolve_incident` com um ID inexistente e confirmar que o handler devolve texto de erro, sem lançar.
- `src/services/tools.test.ts`: a suíte existente continua verde após a extração de `createOpsToolDefinitions` (nenhuma mudança de comportamento esperada nas tools do agente).

## Riscos

- **Divergência de versão do SDK MCP**: pinar a versão exata no `package.json`, como as demais dependências do projeto.
- **Acoplamento de teste ao formato interno do SDK para listar tools**: se a API pública do SDK não expuser um "list" direto, o teste usa o próprio array de definições filtradas (testável sem subir o protocolo) mais um teste de integração mínimo via `StdioServerTransport` em processo, se viável; decisão final registrada em `research.md`.
- **Alguém reintroduzir `console.log` no futuro**: mitigado por um teste estático simples (grep na suíte, ou lint) que falha se houver `console.log` em `src/mcp/`.
