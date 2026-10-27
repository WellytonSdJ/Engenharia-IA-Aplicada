# Feature Specification: Memória Semântica

**Feature Branch**: `009-semantic-memory`

**Created**: 2026-10-26

**Status**: Implemented

**Input**: User description: "Memória semântica: MemoryStore por userId — remember (dedup > 0.92), recall top-3 por produto escalar (mínimo 0.3), forget; tabela memories com embedding all-MiniLM-L6-v2 local em BLOB; /chat ganha userId e injeta o recall no prompt; POST /memories para gravar fatos; teste: recall acha fato sem palavra em comum. @huggingface/transformers com pooling mean + normalize: true, lazy singleton."

> Equivale à spec `008-semantic-memory` do projeto de referência do curso. A numeração aqui é deslocada em +1 porque o projeto 01 ganhou a spec extra `003-fix-replanner-output`.

## Escopo

Dar ao OpsPilot uma **memória de longo prazo por usuário**, que sobrevive entre conversas. A conversa persistente (spec `008-conversa-persistente`) lembra o *fio de uma conversa*; esta feature lembra *fatos sobre a pessoa* (preferências, contexto operacional), e os recupera **por significado**, não por palavra igual.

Fora de escopo: autenticação real (o cliente declara o `userId`), `GET /memories` e `DELETE /memories`, exposição de memória na CLI/arena/bench/MCP, aprendizado automático de fatos (fica para a spec `010-learning-reflector`), vector database externo.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Guardar fatos por usuário sem duplicar (Priority: P1)

Como plantonista, registro fatos relevantes (via `POST /memories`). O sistema guarda o fato sob o meu `userId` e não cria cópia de um fato quase idêntico a um já guardado.

**Why this priority**: Sem `remember` confiável e isolado por usuário, o recall não tem o que buscar; o dedup evita inflar a memória com reformulações.

**Independent Test**: `SqliteMemoryStore` em `:memory:` com `FakeEmbedder`; dois `remember` quase equivalentes resultam em uma entrada só; fatos de outro `userId` não interferem.

**Acceptance Scenarios**:

1. **Given** um `userId` sem memórias, **When** `remember(userId, fato)` roda, **Then** o fato fica persistido e disponível para recall.
2. **Given** um fato já guardado, **When** se tenta guardar outro com similaridade > 0,92, **Then** nada novo é inserido e o id do existente é devolvido (`stored: false`).
3. **Given** dois usuários, **When** cada um grava fatos, **Then** o recall de um nunca devolve fatos do outro.

---

### User Story 2 - Recall semântico top-3 no prompt do `/chat` (Priority: P1)

Antes de executar a estratégia, o `/chat` faz `recall(userId, mensagem)` e injeta no prompt até 3 fatos com similaridade ≥ 0,3, num bloco `Relevant memories:`.

**Why this priority**: É o valor da feature: o agente usa o que sabe do plantonista sem ele repetir.

**Independent Test**: Fato memorizado + consulta **sem palavras em comum** → o fato volta no top-3 (modelo real, `npm run test:semantic`); com estratégia fake, o input recebido contém o bloco de memórias.

**Acceptance Scenarios**:

1. **Given** memórias e uma consulta relacionada, **When** `recall` roda, **Then** volta no máximo 3 fatos, ordenados por score decrescente, todos com score ≥ 0,3.
2. **Given** um fato e uma pergunta sem sobreposição de palavras, **When** o recall roda com o modelo real, **Then** o fato é encontrado.
3. **Given** `POST /chat` com `userId` e memórias relevantes, **When** o turno roda, **Then** o prompt inclui as memórias antes do histórico e `metrics.recalledMemories` informa quantas entraram.
4. **Given** usuário sem memórias relevantes, **When** `/chat` roda, **Then** o turno segue normal, sem bloco de memória, com `recalledMemories: 0`.

---

### User Story 3 - Esquecer e validar sem rede (Priority: P2)

`forget(userId, id)` remove uma memória do próprio usuário. A suíte padrão (`npm test`) roda sem rede, com `FakeEmbedder`.

**Acceptance Scenarios**:

1. **Given** um fato memorizado, **When** `forget` roda com o id e o dono certos, **Then** o fato some dos recalls.
2. **Given** um id inexistente ou de outro usuário, **When** `forget` roda, **Then** devolve `false` e não apaga nada.

### Edge Cases

- `userId` ausente ou vazio em `/chat` ou `/memories`: 400 (Zod).
- Fato/consulta vazios chegando direto ao store: `InvalidMemoryInputError`.
- Mais de 3 memórias acima do piso: só as 3 de maior score.
- Limiares: dedup estrito (`> 0,92`), recall inclusivo (`≥ 0,3`).
- Falha ao carregar o modelo (cold start sem rede): `EmbeddingError` → 503 `EMBEDDING_UNAVAILABLE`, sem corromper o store; a próxima chamada tenta carregar de novo.
- Usuário sem nenhuma memória: o recall devolve `[]` **sem** gerar embedding (não carrega o modelo à toa).
- Memória semântica e histórico (spec 008) coexistem no prompt, com escopos independentes (`userId` × `conversationId`).

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: `MemoryStore` com `remember`, `recall` e `forget`, sempre escopados por `userId`.
- **FR-002**: `remember` gera o embedding, aplica dedup (> 0,92 contra as memórias do mesmo usuário) e persiste.
- **FR-003**: `recall` devolve até 3 fatos com score ≥ 0,3, ordenados por produto escalar decrescente.
- **FR-004**: `forget` remove só a memória do próprio usuário; devolve `false` se não removeu nada.
- **FR-005**: Tabela `memories (id, user_id, fact, embedding BLOB, created_at)` no padrão SQLite do projeto (`node:sqlite`, DDL idempotente, prepared statements), no mesmo arquivo `OPSPILOT_DB`.
- **FR-006**: Embeddings locais com `Xenova/all-MiniLM-L6-v2` via `@huggingface/transformers`, pooling `mean`, `normalize: true`, carregamento lazy singleton.
- **FR-007**: `POST /chat` exige `userId` (Zod) e injeta o recall no prompt antes da estratégia; resposta ganha `metrics.recalledMemories`.
- **FR-008**: `POST /memories` com `{ userId, fact }` grava um fato: 201 se novo, 200 se deduplicado.
- **FR-009**: Teste automatizado provando que o recall acha um fato **sem palavras em comum** com a consulta (modelo real).
- **FR-010**: `npm test` continua sem rede (usa `FakeEmbedder`); `npm run typecheck` verde.

### Key Entities

- **Memory**: `id`, `userId`, `fact`, `embedding` (384 floats), `createdAt`.
- **RecalledMemory**: `id`, `fact`, `score` (só existe no recall).
- **RememberResult**: `id`, `stored`.
- **Embedder**: texto → vetor normalizado.

## Success Criteria _(mandatory)_

- **SC-001**: Em 100% dos testes, `recall` devolve no máximo 3 fatos e nenhum abaixo de 0,3.
- **SC-002**: O dedup impede a segunda inserção quando a similaridade passa de 0,92.
- **SC-003**: O teste "fato sem palavra em comum" passa em `npm run test:semantic`.
- **SC-004**: `/chat` com memórias relevantes entrega os fatos no input da estratégia (asserção com estratégia fake).
- **SC-005**: `npm test` e `npm run typecheck` verdes, sem rede.

## Assumptions

- Similaridade = produto escalar de vetores L2-normalizados (= cosseno).
- `userId` obrigatório no `/chat`; quem chama declara a identidade (sem SSO).
- A constituição é emendada (v1.2.0) para aceitar `@huggingface/transformers` como dependência de produção e o script `test:semantic`.
- O teste com modelo real fica fora do `npm test` (`*.semantic-test.ts`), porque a primeira execução baixa ~23 MB.
