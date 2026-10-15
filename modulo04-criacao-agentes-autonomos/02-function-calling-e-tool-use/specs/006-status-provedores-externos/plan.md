# Implementation Plan: Status de Provedores Externos

**Branch**: `006-status-provedores-externos` | **Date**: 2026-10-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/006-status-provedores-externos/spec.md`

## Summary

Adicionar a tool `check_provider_status` ao catálogo existente (`src/services/tools.ts`), consultando a API pública `statuspage.io` de GitHub ou Cloudflare (sem chave). A tool recebe `provider` (enum `github` | `cloudflare`, default `github`), faz a requisição HTTP com timeout de 5 s (`AbortSignal.timeout`), tenta novamente uma vez em caso de falha de rede ou 5xx, valida a resposta com Zod e devolve uma linha compacta (`indicator — description`). Qualquer falha final (timeout esgotado, 4xx, payload inválido) é capturada dentro da tool e devolvida como string de erro legível — nunca lança exceção para fora. A função de fetch é injetável via parâmetro de fábrica, seguindo o mesmo padrão de injeção de dependência já usado para o `OpsStore` nas demais tools.

## Technical Context

**Language/Version**: TypeScript ESM, `strict: true`, Node.js 22 LTS

**Primary Dependencies**: Já existentes: `@langchain/core` (`tool()`), `zod`. Nenhuma dependência nova — `fetch` e `AbortSignal.timeout` são globais do Node 22.

**Storage**: N/A — a tool não lê nem escreve no `OpsStore`; é uma consulta HTTP de leitura a uma API pública de terceiros.

**Testing**: `node:test` via `tsx`; testes unitários da tool com fetch fake injetado, cobrindo sucesso, timeout, 5xx com retentativa, 4xx sem retentativa e payload inválido — sem acesso à rede real.

**Target Platform**: Mesmo processo Node.js das demais tools (CLI arena/bench, endpoint `/chat`, servidor MCP quando existir).

**Project Type**: Extensão do catálogo de tools existente — nenhuma camada nova.

**Performance Goals**: Resposta em até ~10 s no pior caso (timeout de 5 s × até 2 tentativas); não é um caminho crítico de latência do sistema.

**Constraints**: Sem autenticação; apenas GitHub e Cloudflare nesta feature; retentativa apenas para falha de rede/timeout/5xx, nunca para 4xx; a tool nunca lança exceção para fora de si mesma (erro é observação, como as demais tools do catálogo).

**Scale/Scope**: Uma tool nova, um schema de entrada, um schema de validação da resposta externa, a função de fetch injetável e a suíte de testes correspondente.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio/Regra | Como o plano atende |
| --- | --- |
| I. TypeScript ESM Estrito | Tool em TS ESM, `strict: true`, sem `any`; schemas Zod tipam entrada e a resposta externa. |
| II. Arquitetura em Camadas | A tool vive em `src/services/tools.ts`, junto das demais (Service); não introduz nova camada; a fábrica recebe a função de fetch por injeção, como o store é injetado nas outras tools. |
| III. Contratos e Erros Explícitos | Entrada validada por `.describe()` + enum Zod; resposta externa validada por `ProviderStatusResponseSchema`; qualquer falha é convertida em string de observação dentro da própria tool — não propaga `OpsPilotError` nem exceção nativa para fora. |
| IV. Testes Não Negociáveis | Suíte cobrindo sucesso, timeout, retentativa (5xx → sucesso), 4xx sem retentativa e payload inválido, toda com fetch fake; `npm run typecheck` e `npm test` verdes ao final. |

**Post-design re-check**: **PASS**. Nenhuma violação introduzida pelo design de Phase 1.

## Project Structure

### Documentation (this feature)

```text
specs/006-status-provedores-externos/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── check-provider-status.md
└── tasks.md             # gerado por /speckit.tasks
```

### Source Code (repository root do projeto 02)

```text
src/services/
├── tools.ts              # ALTERADO: + checkProviderStatus, schemas, createOpsTools recebe fetchFn opcional
└── tools.test.ts         # ALTERADO: + suíte de check_provider_status com fetch fake
```

**Structure Decision**: Nenhuma pasta nova. A tool entra no mesmo módulo `src/services/tools.ts` das demais, por ser outra capacidade do mesmo catálogo consumido pelas estratégias de raciocínio (ReAct/Plan-and-Execute) e, futuramente, pelo servidor MCP.

## Decisões e rastreabilidade

| Decisão | Atende |
| --- | --- |
| `CheckProviderStatusInputSchema = z.object({ provider: z.enum(["github", "cloudflare"]).default("github").describe(...) })` | FR-001, FR-002, FR-003 |
| Mapa fixo `PROVIDER_STATUS_URLS = { github: "https://www.githubstatus.com/api/v2/status.json", cloudflare: "https://www.cloudflarestatus.com/api/v2/status.json" }` | FR-004, FR-005 |
| `ProviderStatusResponseSchema = z.object({ status: z.object({ indicator: z.string(), description: z.string() }) }).passthrough()` — aceita campos extras sem falhar | FR-008, edge case de JSON com campos extras |
| `createOpsTools(store, { fetchFn = fetch }: { fetchFn?: typeof fetch } = {})` — a tool usa `fetchFn` em vez do `fetch` global diretamente | FR-011 |
| `fetchProviderStatus(provider, fetchFn)`: cada tentativa usa `fetchFn(url, { signal: AbortSignal.timeout(5000) })`; captura erro de rede/abort e status HTTP; 5xx ou erro de rede → uma retentativa; 4xx → falha final imediata, sem retentativa | FR-006, FR-007, edge case de 4xx |
| Resultado da tool: sucesso → `` `${indicator} — ${description}` ``; qualquer falha final → string de erro legível (ex.: `"Não foi possível consultar o status de ${provider}: ${motivo}"`), nunca `throw` | FR-009, FR-010 |
| A tool inteira roda dentro de um `try/catch` que nunca relança — qualquer exceção inesperada também se torna string de erro | FR-010 |
| `check_provider_status` entra no retorno de `createOpsTools`, ao lado das demais tools já existentes | FR-001 |

## Testes

- `src/services/tools.test.ts` (nova suíte `check_provider_status`): fetch fake retornando payload válido do GitHub → linha compacta com indicador/descrição; fetch fake para Cloudflare com `provider: "cloudflare"` → consulta a URL correta; `provider` omitido → usa `github`; fetch fake que sempre expira (`AbortSignal` disparado) → erro legível após uma retentativa, sem exceção; fetch fake que devolve 500 na primeira chamada e 200 válido na segunda → retorna o status da segunda tentativa; fetch fake que devolve 404 → erro legível imediato, sem segunda chamada (assertar contagem de chamadas); fetch fake com corpo que não casa com o schema (`status` ausente ou tipos errados) → erro legível, sem lançar.

## Riscos

- **Retentativa dobra a latência no caso de falha persistente**: aceito pela spec (no máximo uma retentativa, timeout de 5 s por tentativa); pior caso é ~10 s, ainda dentro de um turno de conversa razoável.
- **Dependência de uma API pública de terceiros em produção**: mitigado nos testes com fetch fake (zero rede real na suíte); em produção, a falha já é tratada como observação, não quebra o agente.
- **Enum fixo de provedores**: adicionar um novo provedor no futuro exige editar o mapa de URLs e o enum; aceito como escopo v1, conforme a spec.
