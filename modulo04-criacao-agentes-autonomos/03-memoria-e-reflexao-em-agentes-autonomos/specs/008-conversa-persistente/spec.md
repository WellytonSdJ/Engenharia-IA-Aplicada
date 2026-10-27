# Feature Specification: Conversa Persistente

**Feature Branch**: `008-conversa-persistente`

**Created**: 2026-10-14

**Status**: Implemented

**Input**: User description: "Conversa persistente: ConversationStore (create/append/lastMessages) + tabela messages no mesmo estilo do SqliteOpsStore; POST /chat ganha conversationId opcional, devolvido na resposta; até 12 últimas mensagens injetadas no prompt via composição; métrica historyMessages; testes em :memory: + estratégia fake."

## Escopo

Estender o endpoint `POST /chat` (feature `004-http-chat-endpoint`) com continuidade de conversa: o cliente pode enviar um `conversationId` opcional; quando ausente, uma nova conversa é criada; quando presente, a conversa existente é reutilizada. As mensagens de cada turno (usuário e agente) são persistidas e, antes de cada execução, as até 12 mensagens mais recentes da conversa são injetadas no prompt. A métrica `historyMessages` reporta quantas mensagens de histórico entraram naquela execução.

Fora de escopo: autenticação, streaming de resposta, limite de histórico configurável pelo cliente, exposição de `conversationId` na CLI/arena/bench, e qualquer mudança nas estratégias de raciocínio além de receberem histórico no prompt.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Continuar a mesma conversa entre requisições (Priority: P1)

Como cliente do endpoint `/chat`, troco várias mensagens sobre o mesmo incidente. Em cada requisição posso omitir ou reenviar o identificador da conversa; o sistema cria uma conversa nova quando necessário, persiste as mensagens e devolve o `conversationId` na resposta para a próxima rodada.

**Why this priority**: Sem continuidade, cada chamada a `/chat` é isolada; o cliente precisa repetir contexto manualmente e o agente não acompanha o fio da conversa.

**Independent Test**: Com uma estratégia fake e um store de conversa em memória, enviar duas mensagens reutilizando o `conversationId` da primeira resposta e verificar que a segunda execução vê o histórico e a resposta inclui o mesmo identificador.

**Acceptance Scenarios**:

1. **Given** o servidor de chat ativo sem `conversationId` no corpo, **When** o cliente envia `POST /chat` com `{ message }`, **Then** o sistema cria uma conversa, persiste a mensagem do usuário e a resposta do agente, e responde 200 com `conversationId` além de `answer`, `trace` e `metrics`.
2. **Given** um `conversationId` já existente, **When** o cliente envia outra mensagem com esse identificador, **Then** o sistema reutiliza a conversa, anexa o novo turno e devolve o mesmo `conversationId` na resposta.
3. **Given** o fluxo com estratégia fake, **When** a suíte de integração roda sem acesso à rede, **Then** a criação e a continuação da conversa passam com asserções corretas.

---

### User Story 2 - Agente recebe as últimas mensagens no prompt (Priority: P1)

Como sistema, antes de executar a estratégia eu carrego as últimas mensagens da conversa (até 12) e as incluo no prompt enviado ao agente; a métrica `historyMessages` reporta quantas mensagens de histórico foram efetivamente injetadas.

**Why this priority**: Persistir sem injetar no prompt não muda o comportamento do agente; o valor está em raciocinar com o fio recente da conversa.

**Independent Test**: Popular uma conversa com N mensagens (N ≤ 12 e N > 12), executar um turno via composição com estratégia fake e verificar o conteúdo do prompt e o valor de `historyMessages`.

**Acceptance Scenarios**:

1. **Given** uma conversa com menos de 12 mensagens persistidas, **When** um novo turno é executado, **Then** todas as mensagens existentes entram no prompt e `metrics.historyMessages` é igual a esse total.
2. **Given** uma conversa com mais de 12 mensagens, **When** um novo turno é executado, **Then** apenas as 12 mais recentes entram no prompt e `metrics.historyMessages` é `12`.
3. **Given** uma conversa recém-criada, sem histórico prévio, **When** a primeira mensagem é processada, **Then** `historyMessages` é `0` e a mensagem atual ainda é processada normalmente.

---

### User Story 3 - Validar store e chat sem rede (Priority: P2)

Como pessoa desenvolvedora, preciso de testes determinísticos do `ConversationStore` em `:memory:` (`create`, `append`, `lastMessages`) e do fluxo HTTP `/chat` com estratégia fake, sem modelo de linguagem nem arquivo em disco.

**Why this priority**: Garante regressão segura alinhada às features `004-http-chat-endpoint` e `005-persistencia-sqlite`.

**Independent Test**: Rodar os testes do store com `:memory:` e os testes de `/chat` com registro fake e store de conversa em memória.

**Acceptance Scenarios**:

1. **Given** um `ConversationStore` em `:memory:`, **When** uma conversa é criada, mensagens são anexadas e `lastMessages` é consultado, **Then** a ordem e o limite (até 12) estão corretos.
2. **Given** o endpoint `/chat` com estratégia fake, **When** os testes de integração rodam, **Then** validam `conversationId` opcional na entrada, presença na resposta e `historyMessages` em `metrics`, sem acesso à rede.
3. **Given** a implementação SQLite do store de conversa, **When** os testes usam `:memory:`, **Then** o DDL idempotente e as operações passam sem depender de um arquivo em disco.

### Edge Cases

- `conversationId` enviado, mas inexistente: erro de cliente claro na borda HTTP, sem executar a estratégia.
- `conversationId` com formato inválido (tipo errado ou string vazia): falha na validação do corpo (400) com issues do Zod.
- A estratégia falha depois de a mensagem do usuário já ter sido anexada: o turno é tratado de forma atômica (ou a resposta do agente não é persistida em caso de falha), sem gerar uma resposta 200 parcial ou enganosa.
- Conversa com exatamente 12 mensagens ao receber a 13ª: apenas as 12 mais recentes anteriores ao turno atual entram no prompt, com `historyMessages === 12`.
- Cliente omite `conversationId` em toda chamada: cada chamada cria uma conversa nova; não há sessão implícita por outro mecanismo (como IP ou cookie).

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: O sistema MUST oferecer um `ConversationStore` com as operações `create`, `append` e `lastMessages` (obter as N mensagens mais recentes de uma conversa).
- **FR-002**: A persistência MUST incluir uma tabela de mensagens (e o necessário para conversas) no mesmo estilo do `SqliteOpsStore`: SQLite nativo, DDL idempotente na construção, apenas prepared statements.
- **FR-003**: Os testes do store MUST usar `:memory:`; em produção, o store MUST reutilizar o mesmo mecanismo de localização de dados já usado pelo projeto (mesma variável de ambiente ou arquivo do store operacional), sem introduzir um segundo arquivo de dados não documentado.
- **FR-004**: `POST /chat` MUST aceitar `conversationId` opcional no corpo, validado com Zod junto aos campos já existentes.
- **FR-005**: Em caso de sucesso, a resposta 200 MUST incluir `conversationId` (criado ou reutilizado) além de `answer`, `trace` e `metrics`.
- **FR-006**: Quando `conversationId` estiver ausente, o sistema MUST criar uma nova conversa antes de executar o turno.
- **FR-007**: Quando `conversationId` estiver presente e for válido, o sistema MUST carregar essa conversa; um identificador desconhecido MUST resultar em erro de cliente, sem executar a estratégia.
- **FR-008**: A composição do endpoint `/chat` MUST injetar no prompt as até 12 mensagens mais recentes da conversa (via `lastMessages`) antes de executar a estratégia.
- **FR-009**: `metrics` MUST incluir `historyMessages` (inteiro ≥ 0), indicando quantas mensagens de histórico foram injetadas no prompt naquela execução.
- **FR-010**: Após uma execução bem-sucedida, o sistema MUST persistir o turno (mensagem do usuário e resposta do agente) via `append`.
- **FR-011**: DEVE existir suíte de testes do `ConversationStore` em `:memory:` cobrindo `create`, `append` e `lastMessages` (incluindo o limite de 12).
- **FR-012**: DEVE existir teste de integração de `/chat` com estratégia fake e store de conversa em memória, sem acesso à rede, cobrindo criação, continuação e a métrica `historyMessages`.

### Key Entities

- **Conversa**: fio de diálogo com o cliente do endpoint. Atributos: identificador estável (`conversationId`), momento de criação.
- **Mensagem**: turno persistido numa conversa. Atributos: identificador, `conversationId`, papel (usuário ou agente), conteúdo textual, ordem/timestamp para recuperar as mais recentes.
- **Requisição de chat** (estendida): campos existentes (`message`, `strategy?`, `reflect?`) mais `conversationId?`.
- **Resposta de chat** (estendida): campos existentes (`answer`, `trace`, `metrics`) mais `conversationId`; `metrics` passa a incluir `historyMessages`.
- **ConversationStore**: contrato de persistência de conversas e mensagens (`create`, `append`, `lastMessages`).

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Um cliente completa dois turnos reutilizando o `conversationId` da primeira resposta e observa continuidade (histórico injetado, mesma conversa) em 100% dos testes de integração cobertos.
- **SC-002**: Com mais de 12 mensagens na conversa, exatamente 12 entram no prompt e `historyMessages` reporta `12` nos testes.
- **SC-003**: Uma requisição sem `conversationId` sempre devolve um novo identificador; uma requisição com identificador inválido ou inexistente não executa a estratégia, nos testes.
- **SC-004**: A suíte `:memory:` do store e os testes de `/chat` com estratégia fake passam sem rede; `npm test` e `npm run typecheck` permanecem verdes após a feature.
- **SC-005**: Um cliente consegue retomar o contexto operacional entre turnos HTTP sem reenviar o histórico manualmente no corpo da requisição.

## Assumptions

- Esta feature estende o contrato da feature `004-http-chat-endpoint` (campos e status existentes permanecem; autenticação e streaming continuam fora de escopo).
- Os papéis de mensagem se alinham ao domínio de mensagens já usado pelas estratégias (mensagem de usuário / mensagem do agente); o plano fixa o mapeamento exato.
- O limite de histórico é fixo em 12 mensagens nesta feature (não configurável).
- `historyMessages` conta apenas mensagens de histórico injetadas, não a mensagem atual da requisição.
- A implementação SQLite do `ConversationStore` segue o mesmo padrão do `SqliteOpsStore` (sem ORM); pode viver no mesmo arquivo de banco ou em um módulo irmão — decisão de co-localização cabe ao plano.
- A estratégia fake determinística dos testes de `/chat` (já usada na feature `004`) continua sendo o harness de testes HTTP; não exige provedor de modelo real.
- CLI, arena e bench não precisam expor `conversationId` nesta feature — apenas o endpoint HTTP `/chat`, o store e a composição.
- A política de falha no meio do turno (mensagem do usuário já persistida versus turno atômico) é detalhe de plano; a spec exige apenas comportamento testável e nenhuma resposta 200 parcial enganosa.
