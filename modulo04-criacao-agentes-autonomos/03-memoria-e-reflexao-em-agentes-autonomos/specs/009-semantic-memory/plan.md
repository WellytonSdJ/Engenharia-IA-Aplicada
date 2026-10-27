# Implementation Plan: Memória Semântica

**Branch**: `009-semantic-memory` | **Date**: 2026-10-26 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/009-semantic-memory/spec.md`

## Summary

Adicionar um `SqliteMemoryStore` (tabela `memories` no mesmo `OPSPILOT_DB`) com `remember`/`recall`/`forget` por `userId`, alimentado por embeddings locais `all-MiniLM-L6-v2` (`@huggingface/transformers`, lazy singleton). A busca é força bruta em TypeScript: produto escalar contra cada fato do usuário, filtro ≥ 0,3, top-3. O `/chat` passa a exigir `userId`, faz recall da mensagem e injeta um bloco `Relevant memories:` no input da estratégia, antes do histórico da spec 008. Uma rota nova `POST /memories` grava fatos manualmente.

## Technical Context

**Language/Version**: TypeScript ESM, `strict: true`, Node.js 22 LTS

**Primary Dependencies**: existentes (`zod`, `express`, `node:sqlite`) + **nova** `@huggingface/transformers@^4.2.0` (pipeline `feature-extraction`)

**Storage**: tabela `memories` no arquivo `OPSPILOT_DB`; embedding 384-d em BLOB (`Float32Array` ↔ `Buffer`); `:memory:` nos testes

**Testing**: `node:test` via `tsx`. `npm test` usa `FakeEmbedder` (sem rede). `npm run test:semantic` roda `*.semantic-test.ts` com o modelo real.

**Performance Goals**: recall sobre dezenas de fatos por usuário < 1 s; cold start do modelo só na primeira chamada que precisa de embedding

**Constraints**: limiares fixos 0,92 / 0,3, top-3; prepared statements; estratégias não mudam de assinatura (recebem a string já enriquecida)

## Constitution Check

| Princípio/Regra | Como o plano atende |
| --- | --- |
| I. TypeScript ESM Estrito | Contratos explícitos: `MemoryStore`, `Embedder`, schemas Zod `RecalledMemorySchema`/`RememberResultSchema`. |
| II. Arquitetura em Camadas | `SqliteMemoryStore` em `src/models/`; `embeddings.ts`/`fake-embedder.ts` em `src/services/`; `memory-controller.ts` em `src/controllers/`; orquestração em `src/services/chat.ts`. |
| III. Contratos e Erros Explícitos | `userId`/`fact` validados com Zod na borda; `InvalidMemoryInputError` (400) e `EmbeddingError` (503) traduzidos só no controller. |
| IV. Testes Não Negociáveis | Store, chat, HTTP e teste semântico real. |
| V. Integração Segura e Simples | Dependência nova exige **emenda da constituição** (v1.2.0): embeddings locais, sem API key nova, sem serviço externo de vetores. |

**Post-design re-check**: **PASS** com a emenda v1.2.0.

## Project Structure

```text
src/
├── models/
│   ├── domain.ts                    # ALTERADO: RecalledMemorySchema, RememberResultSchema
│   ├── memory-store.ts              # NOVO: MemoryStore + SqliteMemoryStore, dot, BLOB <-> Float32Array
│   ├── memory-store.test.ts         # NOVO: FakeEmbedder, :memory:
│   └── memory-store.semantic-test.ts# NOVO: modelo real (npm run test:semantic)
├── services/
│   ├── embeddings.ts                # NOVO: Embedder + lazy singleton all-MiniLM-L6-v2
│   ├── fake-embedder.ts             # NOVO: double determinístico
│   ├── errors.ts                    # ALTERADO: InvalidMemoryInputError, EmbeddingError
│   └── chat.ts                      # ALTERADO: userId, recall, formatMemoriesForPrompt, recalledMemories
├── controllers/
│   ├── chat-controller.ts           # ALTERADO: userId obrigatório; 400/503 para erros de memória
│   └── memory-controller.ts         # NOVO: POST /memories
├── agents/
│   ├── system-prompt.ts             # NOVO: OPSPILOT_SYSTEM_PROMPT + MEMORY_GUIDANCE
│   ├── react.ts                     # ALTERADO: usa OPSPILOT_SYSTEM_PROMPT
│   └── plan-and-execute.ts          # ALTERADO: planner recebe MEMORY_GUIDANCE
└── http/server.ts                   # ALTERADO: rota /memories
```

**Structure Decision**: o projeto de referência põe tudo em `src/memory/`. Aqui os arquivos mantêm **os mesmos nomes** (`memory-store.ts`, `embeddings.ts`, `fake-embedder.ts`), mas cada um vai para a sua camada MVC, como manda a constituição.

## Decisões e rastreabilidade

| Decisão | Atende |
| --- | --- |
| `remember`: embed → compara com todas as memórias do usuário → se algum `dot > 0.92`, devolve `{ id: existente, stored: false }`; senão insere `memory-<uuid>` | FR-002 |
| `recall`: pontua tudo → filtra `≥ 0.3` → ordena desc → `slice(0, 3)` (filtrar antes de cortar) | FR-003 |
| `recall` de usuário sem linhas devolve `[]` sem chamar o embedder | edge case cold start |
| `forget`: `DELETE ... WHERE id = ? AND user_id = ?`; devolve `changes === 1` | FR-004 |
| `created_at` como TEXT ISO, igual aos outros stores do projeto (a referência usa INTEGER) | FR-005 |
| Import dinâmico de `@huggingface/transformers` dentro do loader; singleton guarda a *promise*; em falha, zera a promise para tentar de novo | FR-006, edge case |
| Ordem no prompt: `Relevant memories:` → `Conversation so far:` → `User: <mensagem>`; sem nenhum dos dois, a mensagem segue intacta | FR-007 |
| `EmbeddingError` → **503** `EMBEDDING_UNAVAILABLE` (a referência usa 500): é indisponibilidade de dependência | edge case |
| `POST /memories` → 201 novo / 200 dedup, corpo `{ id, stored, userId, fact }` | FR-008 |
| `OPSPILOT_SYSTEM_PROMPT` ensina o ReAct a ler o bloco de memórias; o planner do Plan-and-Execute recebe só `MEMORY_GUIDANCE` (ele não usa `createReactAgent`) | FR-007 |

## Riscos

- **Download na primeira execução** (~23 MB + runtime ONNX): mitigado pelo lazy loading e pelo atalho "usuário sem memórias não gera embedding".
- **Modelo em inglês**: o MiniLM foi treinado sobretudo em inglês; frases em português funcionam, mas com scores menores. Os limiares são os da referência; calibrar fica para depois.
- **Força bruta**: O(n) por recall. Ok para dezenas/centenas de fatos por usuário; um índice ANN só se o volume crescer muito.
