# Implementation Plan: Conversa Persistente

**Branch**: `008-conversa-persistente` | **Date**: 2026-10-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/008-conversa-persistente/spec.md`

## Summary

Estender `POST /chat` (feature `004-http-chat-endpoint`) com um `ConversationStore` baseado em SQLite nativo (`node:sqlite`, `DatabaseSync`), no mesmo estilo do `SqliteOpsStore` — tabelas `conversations` e `messages`, DDL idempotente, prepared statements. O corpo da requisição ganha `conversationId` opcional; a resposta devolve `conversationId` (criado ou reutilizado) e `metrics.historyMessages`. Antes de executar a estratégia, o service de chat carrega até 12 mensagens recentes via `lastMessages` e as injeta no histórico passado ao agente (como mensagens anteriores na conversa). Depois de uma execução bem-sucedida, o turno completo (mensagem do usuário + resposta do agente) é persistido via `append`, dentro de uma única operação que evita uma resposta 200 parcial caso a estratégia falhe no meio do caminho.

## Technical Context

**Language/Version**: TypeScript ESM, `strict: true`, Node.js 22 LTS

**Primary Dependencies**: Já existentes: `zod`, `@langchain/core` (mensagens `HumanMessage`/`AIMessage`), `node:sqlite` nativo. Nenhuma dependência nova.

**Storage**: Novo `ConversationStore` via `node:sqlite`. Pode compartilhar o mesmo arquivo de banco do `SqliteOpsStore` (mesma variável `OPSPILOT_DB`) ou abrir uma conexão própria sobre o mesmo arquivo — decisão registrada em `research.md`; testes usam `:memory:`.

**Testing**: `node:test` via `tsx`; suíte `:memory:` do `ConversationStore` (`create`, `append`, `lastMessages`, limite de 12); testes de integração de `/chat` com estratégia fake e `ConversationStore` em memória, sem rede.

**Target Platform**: Mesmo servidor Express do endpoint `/chat` (spec 004).

**Project Type**: Extensão do endpoint HTTP existente — uma camada de persistência nova e ajustes no service/controller de chat já implementados pela spec 004.

**Performance Goals**: `lastMessages` limitado a 12 registros por consulta; sem impacto perceptível de latência adicional ao turno de chat.

**Constraints**: Limite de histórico fixo em 12 (não configurável nesta feature); CLI/arena/bench não recebem `conversationId`; política de falha no meio do turno deve ser atômica o suficiente para nunca devolver 200 com persistência parcial enganosa.

**Scale/Scope**: Um novo store (duas tabelas), extensão do schema de entrada/saída de `/chat`, ajuste no service de chat para injetar histórico e persistir o turno, e as suítes de teste correspondentes.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio/Regra | Como o plano atende |
| --- | --- |
| I. TypeScript ESM Estrito | `ConversationStore` em TS ESM, `strict: true`, sem `any`; schemas Zod para `Conversation`/`Message` e para a extensão de `ChatRequestSchema`/`ChatResponseSchema`. |
| II. Arquitetura em Camadas | `ConversationStore` vive em `src/models/` (Model), ao lado de `SqliteOpsStore`; o service de chat (`src/services/chat.ts`, já criado pela spec 004) ganha a orquestração de histórico; o controller HTTP só repassa `conversationId` no corpo/resposta, sem lógica de persistência própria. |
| III. Contratos e Erros Explícitos | `conversationId` inválido ou inexistente é um erro de domínio novo (`UnknownConversationError` ou equivalente), traduzido para status HTTP somente na borda, como os demais erros da spec 004. |
| IV. Testes Não Negociáveis | Suíte `:memory:` do store (create/append/lastMessages, limite 12) e suíte de integração de `/chat` cobrindo criação, continuação e a métrica `historyMessages`; `npm run typecheck` e `npm test` verdes ao final. |

**Post-design re-check**: **PASS**. Nenhuma violação introduzida; a extensão reaproveita os padrões já estabelecidos pelas specs 004 e 005.

## Project Structure

### Documentation (this feature)

```text
specs/008-conversa-persistente/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── conversation-store.md
└── tasks.md             # gerado por /speckit.tasks
```

### Source Code (repository root do projeto 02)

```text
src/
├── models/
│   ├── conversation-store.ts       # NOVO: ConversationStore via node:sqlite
│   └── conversation-store.test.ts  # NOVO: :memory:, create/append/lastMessages, limite 12
├── services/
│   ├── chat.ts                     # ALTERADO (da spec 004): injeta histórico via lastMessages, persiste turno via append
│   ├── chat.test.ts                # ALTERADO: cobre historyMessages e a política de falha no meio do turno
│   └── errors.ts                   # ALTERADO: + UnknownConversationError
├── controllers/
│   └── chat-controller.ts          # ALTERADO (da spec 004): ChatRequestSchema ganha conversationId?; resposta inclui conversationId
└── http/
    └── server.test.ts              # ALTERADO: + cenários de conversationId (criação, continuação, id inválido/inexistente)
```

**Structure Decision**: Nenhuma pasta nova além do arquivo do store. A feature estende arquivos já criados pela spec 004 (`chat.ts`, `chat-controller.ts`, `server.test.ts`), seguindo exatamente a mesma divisão de camadas.

## Decisões e rastreabilidade

| Decisão | Atende |
| --- | --- |
| `ConversationStore` com tabelas `conversations(id TEXT PRIMARY KEY, created_at TEXT NOT NULL)` e `messages(id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id), role TEXT NOT NULL CHECK (role IN ('user','assistant')), content TEXT NOT NULL, created_at TEXT NOT NULL)`, DDL idempotente no construtor | FR-001, FR-002 |
| `create(): Promise<Conversation>`, `append(conversationId, messages): Promise<void>`, `lastMessages(conversationId, limit): Promise<Message[]>` (ordenadas da mais antiga para a mais recente, já truncadas ao limite) | FR-001 |
| Produção: `ConversationStore` abre o mesmo arquivo de `OPSPILOT_DB` (conexão própria sobre o mesmo caminho); testes usam `:memory:` isolado por execução | FR-003 |
| `ChatRequestSchema` (spec 004) ganha `conversationId: z.string().trim().min(1).optional()` | FR-004 |
| `ChatResponseSchema` (spec 004) ganha `conversationId: z.string()`; `metrics` ganha `historyMessages: z.number().int().nonnegative()` | FR-005, FR-009 |
| Fluxo em `runChat` (spec 004, `src/services/chat.ts`): sem `conversationId` → `store.create()`; com `conversationId` → `store.lastMessages(id, 1)` (ou consulta de existência dedicada) para validar que a conversa existe antes de prosseguir; inexistente → lança `UnknownConversationError`, tratado como erro de cliente na borda HTTP (sem executar a estratégia) | FR-006, FR-007 |
| Histórico: `store.lastMessages(conversationId, 12)` roda antes de chamar `strategy.run`; as mensagens voltam como `HumanMessage`/`AIMessage` conforme o papel e entram no `input` da estratégia como contexto anterior; `historyMessages = history.length` | FR-008, FR-009 |
| Persistência do turno: só depois de `strategy.run` resolver com sucesso, `store.append(conversationId, [{ role: "user", content: message }, { role: "assistant", content: answer }])`; se a estratégia falhar, nada é anexado (turno atômico do ponto de vista observável) — decisão que resolve o edge case de política de falha no meio do turno | FR-010, edge case de falha no meio do turno |
| Erro de domínio novo `UnknownConversationError` em `src/services/errors.ts`, mapeado no controller para um status de erro de cliente (alinhado ao padrão já usado pela spec 004 para `UnknownStrategyError`) | FR-007 |

## Testes

- `src/models/conversation-store.test.ts` (`:memory:`): `create` gera um identificador novo por chamada; `append` grava mensagens na ordem; `lastMessages` com uma conversa de menos de 12 mensagens devolve todas; com mais de 12, devolve exatamente as 12 mais recentes, na ordem correta; conversa sem nenhuma mensagem devolve lista vazia.
- `src/services/chat.test.ts`: `runChat` sem `conversationId` cria uma conversa nova e inclui o id na resposta; com `conversationId` válido, reutiliza a conversa e injeta o histórico correto no input da estratégia; com `conversationId` inexistente, lança `UnknownConversationError` sem invocar a estratégia; após uma execução cujo histórico tem 13 mensagens anteriores, `historyMessages` é `12`; após uma execução sem histórico prévio, `historyMessages` é `0`; se a estratégia fake lançar, nenhuma mensagem é anexada à conversa.
- `src/http/server.test.ts` (integração, porta efêmera): duas chamadas `POST /chat` reutilizando o `conversationId` da primeira resposta mostram continuidade; `conversationId` com string vazia → 400 com issues; `conversationId` bem formado mas inexistente → erro de cliente sem executar a estratégia.

## Riscos

- **Co-localização do arquivo SQLite da conversa com o de operações**: abrir duas conexões (`SqliteOpsStore` e `ConversationStore`) sobre o mesmo arquivo pode exigir atenção a modo de acesso (`DatabaseSync` permite múltiplas instâncias no mesmo processo); validar em `research.md` antes da implementação; alternativa simples é um arquivo separado (`OPSPILOT_CONVERSATIONS_DB`) se a co-localização causar atrito.
- **Crescimento ilimitado da tabela de mensagens**: fora de escopo nesta feature (sem política de retenção); aceito pela spec como decisão futura.
- **Depende da spec 004 estar implementada**: esta feature estende arquivos (`chat.ts`, `chat-controller.ts`) que só existem depois do `/speckit.implement` da spec 004; não pode ser implementada isoladamente antes disso.
