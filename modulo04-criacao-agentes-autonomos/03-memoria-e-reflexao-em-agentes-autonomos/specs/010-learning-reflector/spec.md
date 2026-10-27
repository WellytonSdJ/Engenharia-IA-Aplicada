# Feature Specification: Refletor de Aprendizado

**Feature Branch**: `010-learning-reflector`

**Created**: 2026-10-26

**Status**: Implemented

**Input**: User description: "Refletor de aprendizado: após cada resposta, um withStructuredOutput({ hasLearning, fact }) lê a última mensagem do usuário e destila fatos duráveis (nunca pedido pontual, nunca segredo) -> memoryStore.remember assíncrono; tool forget_preference."

> Equivale à spec `009-learning-reflector` do projeto de referência do curso (numeração deslocada em +1).

## Escopo

Fechar o ciclo da memória semântica (spec `009-semantic-memory`): em vez de depender só de `POST /memories`, o OpsPilot **aprende sozinho** depois de cada turno bem-sucedido do `/chat`. Um refletor LLM com saída estruturada lê a mensagem do usuário e decide se há um fato **durável** para lembrar. E o plantonista ganha a tool `forget_preference` para pedir, conversando, que o agente esqueça algo.

É o padrão Reflection (spec `002-camada-reflection`) reaproveitado com outro objetivo: o `withReflection` critica a **resposta** antes de entregar; o refletor de aprendizado olha a **mensagem do usuário** depois de entregar.

Fora de escopo: evento `learning` no trace (o aprendizado acontece depois da resposta, então não cabe no trace dela), refletor na CLI/arena/bench, `forget_preference` no servidor MCP e no Plan-and-Execute, DLP de verdade para segredos.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Aprender preferências duráveis após o turno (Priority: P1)

Depois que o `/chat` responde, o refletor analisa a última mensagem do usuário. Se `hasLearning` for verdadeiro, o sistema chama `memoryStore.remember(userId, fact)` **sem atrasar** a resposta HTTP.

**Independent Test**: Refletor fake + `MemoryStore` espião; o `remember` é chamado uma vez, e o `runChat` retorna antes de o `remember` terminar.

**Acceptance Scenarios**:

1. **Given** um turno bem-sucedido com mensagem "sempre priorize o checkout", **When** o refletor devolve `{ hasLearning: true, fact }`, **Then** `remember(userId, fact)` é chamado e o fato fica disponível para recall.
2. **Given** o mesmo turno, **When** a resposta 200 é montada, **Then** ela não espera o `remember`.
3. **Given** `hasLearning: false`, **When** o turno termina, **Then** `remember` não é chamado.

---

### User Story 2 - Só fatos duráveis: nunca pedido pontual nem segredo (Priority: P1)

O prompt do refletor proíbe aprender pedidos pontuais ("liste alertas", "abra um incidente") e segredos (senhas, tokens, API keys).

**Acceptance Scenarios**:

1. **Given** um pedido pontual, **When** o refletor avalia, **Then** `hasLearning` é falso e nada é gravado.
2. **Given** uma mensagem com segredo, **When** o refletor avalia, **Then** `hasLearning` é falso e nada é gravado.
3. **Given** uma preferência durável, **When** o refletor avalia, **Then** `fact` é um enunciado estável, não a frase bruta.

---

### User Story 3 - Esquecer preferência via tool (Priority: P2)

O ReAct ganha a tool `forget_preference(query)`: faz recall da descrição, pega o melhor resultado e o apaga, sempre do usuário do turno.

**Acceptance Scenarios**:

1. **Given** um fato memorizado, **When** o agente chama `forget_preference` com uma descrição parecida, **Then** o fato some dos recalls daquele usuário.
2. **Given** nenhuma memória parecida, **When** a tool roda, **Then** responde que não encontrou e não altera nada.
3. **Given** o schema da tool, **When** o modelo o lê, **Then** não existe campo `userId`: o usuário é o do turno, fechado no código.

### Edge Cases

- Refletor falha (timeout, JSON inválido, sem chave): vira "sem aprendizado"; o `/chat` já respondeu e não muda.
- `remember` assíncrono falha: registrado no stderr, sem afetar turnos futuros.
- `hasLearning: true` com `fact` vazio: tratado como sem aprendizado.
- Estratégia falha: o refletor nem roda (só turnos bem-sucedidos aprendem).
- `reflect: true` (spec 002) continua independente: critique antes, aprendizado depois.
- Turnos simultâneos do mesmo usuário: os `remember` podem terminar fora de ordem; o dedup da spec 009 evita duplicatas.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: Após cada `/chat` bem-sucedido, se houver refletor configurado, ele lê a mensagem do usuário daquele turno.
- **FR-002**: Saída estruturada `{ hasLearning: boolean, fact: string }` via `withStructuredOutput`.
- **FR-003**: `hasLearning === true` e `fact` não vazio → `memoryStore.remember(userId, fact)` **fire-and-forget**.
- **FR-004**: Caso contrário, `remember` não é chamado.
- **FR-005**: O prompt do refletor proíbe pedidos pontuais e segredos.
- **FR-006**: Falhas do refletor ou do `remember` nunca alteram status nem corpo da resposta.
- **FR-007**: Tool `forget_preference` no catálogo do ReAct quando o turno tem usuário (`ReasoningOptions.memory`).
- **FR-008**: `forget_preference` respeita o isolamento por `userId` e segue as regras de descrição de tools do projeto.
- **FR-009**: Testes com refletor fake: aprende, não aprende pontual/segredo, falhas engolidas, `/chat` não espera o `remember`, tool remove o fato certo.
- **FR-010**: Reutiliza o `MemoryStore` da spec 009 (nenhum store novo).

### Key Entities

- **LearningReflection**: `hasLearning`, `fact`.
- **LearningReflectorFn**: `(userMessage) => Promise<LearningReflection>`, injetável.
- **forget_preference**: entrada `{ query }`; efeito em `MemoryStore`.

## Success Criteria _(mandatory)_

- **SC-001**: Preferência durável + refletor positivo → `remember` exatamente uma vez com o fato esperado.
- **SC-002**: Pedido pontual e segredo → `remember` nunca chamado.
- **SC-003**: `runChat` resolve com o `remember` ainda pendente (teste com promise travada).
- **SC-004**: Após `forget_preference`, o recall do mesmo usuário não devolve o fato; o de outro usuário continua intacto.
- **SC-005**: `npm test` e `npm run typecheck` verdes.

## Assumptions

- O refletor usa a mesma fábrica de modelo (`createModel`, OpenRouter); nos testes é fake.
- O refletor só é ligado em `src/index.ts` (servidor real). Testes de `/chat` não o recebem.
- `forget_preference` resolve o alvo por texto (recall top-1 acima de 0,3 + forget por id).
- O `userId` chega à tool por closure (as tools do ReAct são criadas a cada execução). O projeto de referência usa `AsyncLocalStorage` porque lá as tools são criadas uma vez só, na inicialização.
