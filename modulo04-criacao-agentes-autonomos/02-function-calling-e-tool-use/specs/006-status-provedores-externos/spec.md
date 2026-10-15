# Feature Specification: Status de Provedores Externos

**Feature Branch**: `006-status-provedores-externos`

**Created**: 2026-10-14

**Status**: Implemented

**Input**: User description: "Tool de status de provedores externos: check_provider_status em src/services/tools.ts consulta a statuspage pública do provedor via API statuspage.io (sem chave): github -> https://www.githubstatus.com/api/v2/status.json; cloudflare -> https://www.cloudflarestatus.com/api/v2/status.json. Parâmetro provider (enum github | cloudflare, default github, com .describe). Descrição orientada a quando usar: suspeita de problema externo, 'é o nosso ou do provedor?', dependência fora do ar. Resiliência: timeout de 5s via AbortSignal.timeout; falha de rede ou 5xx, uma nova tentativa; resposta validada com zod ({ status: { indicator, description } }); qualquer falha final retorna string de erro legível como resultado da tool (erro é observação, nunca lançar exceção para fora da tool). Retorno compacto (indicador + descrição, uma linha). Teste: a função de fetch é injetável; testes cobrem sucesso, timeout e resposta inválida sem uso de rede (fake fetch)."

## Escopo

Adicionar ao catálogo de tools do OpsPilot uma nova capacidade de diagnóstico: consultar o status público de um provedor externo (GitHub ou Cloudflare) para ajudar o plantonista a distinguir uma falha interna de uma indisponibilidade de terceiros durante um incidente. A tool consulta a statuspage pública do provedor (API `statuspage.io`, sem autenticação), aplica uma política curta de resiliência (timeout + uma retentativa) e sempre devolve uma observação em texto — nunca lança exceção para fora da tool.

Fora de escopo: provedores além de GitHub e Cloudflare; autenticação ou chave de API; normalização do indicador para um enum interno do projeto; alteração do contrato HTTP `/chat` ou do contrato das demais tools.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Distinguir falha interna de falha do provedor (Priority: P1)

Como plantonista (via agente), durante um incidente eu quero perguntar se a causa pode ser externa (GitHub ou Cloudflare fora do ar) e receber o status oficial do provedor em uma linha, para decidir se a investigação deve continuar internamente ou aguardar o provedor.

**Why this priority**: É o valor central da feature — reduzir investigação de falso positivo interno quando a causa real é uma dependência externa.

**Independent Test**: Invocar `check_provider_status` com uma função de fetch fake que devolve um payload válido de status; verificar que o retorno é uma linha compacta com indicador e descrição, sem acesso à rede real.

**Acceptance Scenarios**:

1. **Given** a tool registrada e um provedor com statuspage saudável, **When** o plantonista pergunta sobre o GitHub (ou omite o provedor), **Then** a tool usa `github` por padrão e retorna uma linha com indicador e descrição oficiais.
2. **Given** a mesma tool, **When** o plantonista pede o status da Cloudflare (`provider: "cloudflare"`), **Then** a consulta usa a statuspage da Cloudflare e o retorno reflete o indicador/descrição dessa página.
3. **Given** a descrição da tool, **When** o modelo decide se deve chamá-la, **Then** a descrição deixa claro o gatilho de uso: suspeita de problema externo, dúvida "é o nosso ou do provedor?", dependência fora do ar.

---

### User Story 2 - Falha externa se torna observação, não quebra o raciocínio (Priority: P1)

Como sistema, quando a consulta ao provedor externo falha (timeout, erro de rede, 5xx ou payload inválido), eu quero continuar o raciocínio do agente normalmente, recebendo uma string de erro legível como resultado da tool.

**Why this priority**: Ferramentas que lançam exceção para fora interrompem a execução do agente; erro como observação já é o padrão do restante do catálogo de tools do projeto.

**Independent Test**: Com fetch fake, simular timeout, HTTP 5xx e payload fora do schema esperado; verificar que a tool sempre retorna uma string de erro legível e nenhuma exceção escapa dela.

**Acceptance Scenarios**:

1. **Given** a consulta demora mais que 5 segundos, **When** a tool é invocada, **Then** ela aborta por timeout, faz no máximo uma nova tentativa e, se ainda falhar, retorna uma string de erro legível sem lançar exceção.
2. **Given** falha de rede ou resposta HTTP 5xx na primeira tentativa, **When** a tool roda, **Then** há exatamente uma nova tentativa; se a segunda também falhar, o resultado é um erro legível.
3. **Given** uma resposta HTTP bem-sucedida cujo corpo não corresponde ao schema esperado (`status.indicator` e `status.description`), **When** a tool valida o payload, **Then** retorna um erro legível, sem propagar dados inválidos para o contexto do agente.

---

### User Story 3 - Testar sem rede (Priority: P2)

Como pessoa desenvolvedora, quero cobrir sucesso, timeout e resposta inválida com uma função de fetch injetável (fake), sem depender das statuspages reais na suíte de testes.

**Why this priority**: Statuspages públicas são infraestrutura externa fora do controle do projeto; depender delas tornaria os testes instáveis.

**Independent Test**: Executar os testes unitários da tool com fetch fake; a suíte passa inteiramente sem acesso à rede.

**Acceptance Scenarios**:

1. **Given** um fetch fake que devolve um JSON válido no formato statuspage.io, **When** o teste de sucesso roda, **Then** o resultado é a linha compacta com indicador e descrição.
2. **Given** um fetch fake que simula timeout, **When** o teste de timeout roda, **Then** a tool retorna um erro legível após a política de retentativa, sem acesso à rede.
3. **Given** um fetch fake que devolve corpo inválido, **When** o teste de validação roda, **Then** a tool retorna um erro legível e o teste não usa rede.

### Edge Cases

- `provider` omitido: usa o padrão `github`.
- `provider` com valor fora do enum (`github` | `cloudflare`): falha na validação de entrada da tool, antes de qualquer chamada de rede.
- Primeira tentativa falha e a segunda sucede: retorna o status compacto da segunda tentativa, sem mencionar a retentativa no retorno de sucesso.
- JSON de resposta com campos extras além de `status.indicator`/`status.description`: aceitos (validação cobre apenas os campos obrigatórios); o retorno expõe só indicador e descrição.
- Resposta HTTP 4xx (por exemplo, 404): tratada como falha final legível, sem retentativa (retentativa é só para falha de rede ou 5xx).
- Indicador oficial fora do vocabulário comum (`none`, `minor`, `major`, `critical`): repassado como veio, sem normalização para outro enum interno nesta feature.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: O sistema MUST expor a tool `check_provider_status` no catálogo de tools (`src/services/tools.ts`).
- **FR-002**: A tool MUST aceitar o parâmetro `provider` com enum `github` | `cloudflare`, padrão `github`, com `.describe()` explicando o parâmetro.
- **FR-003**: A descrição da tool MUST orientar o modelo sobre quando usá-la: suspeita de problema externo, dúvida "é o nosso ou do provedor?", dependência fora do ar.
- **FR-004**: Para `provider: "github"`, a tool MUST consultar `https://www.githubstatus.com/api/v2/status.json`.
- **FR-005**: Para `provider: "cloudflare"`, a tool MUST consultar `https://www.cloudflarestatus.com/api/v2/status.json`.
- **FR-006**: Cada tentativa de consulta MUST respeitar um timeout de 5 segundos (`AbortSignal.timeout` ou equivalente).
- **FR-007**: Em falha de rede ou resposta HTTP 5xx, a tool MUST realizar exatamente uma nova tentativa; demais falhas (timeout esgotado, 4xx, validação de payload) MUST resultar em erro legível sem retentativas adicionais.
- **FR-008**: A resposta HTTP bem-sucedida MUST ser validada com um schema Zod no formato `{ status: { indicator, description } }` (campos string); payload fora do schema MUST se tornar erro legível.
- **FR-009**: Em caso de sucesso, a tool MUST retornar uma string compacta de uma linha com indicador e descrição (por exemplo, `none — All Systems Operational`).
- **FR-010**: Qualquer falha final MUST retornar uma string de erro legível como resultado da tool; a tool MUST NÃO lançar exceção para fora de si mesma.
- **FR-011**: A função de fetch usada pela tool MUST ser injetável, para permitir testes sem acesso à rede.
- **FR-012**: DEVE existir cobertura de testes para sucesso, timeout e resposta inválida, usando fetch fake, sem acesso à rede real.

### Key Entities

- **Status do provedor**: resultado oficial de status de um provedor suportado (`provider`, `indicator`, `description`).
- **Consulta de status**: pedido do agente com `provider` opcional (padrão `github`), mapeado para a URL pública da statuspage correspondente.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Em cenário com statuspage saudável (simulada), o agente obtém indicador e descrição oficiais em uma única linha de observação em 100% das invocações bem-sucedidas da tool, nos testes.
- **SC-002**: Quando a consulta externa falha (timeout, 5xx ou payload inválido), 100% das invocações nos testes devolvem erro legível como observação — nenhuma exceção escapa da tool.
- **SC-003**: Após uma falha transitória seguida de sucesso na segunda tentativa, o retorno é o status compacto da segunda tentativa, sem necessidade de nova pergunta do plantonista.
- **SC-004**: A suíte de testes da tool cobre sucesso, timeout e resposta inválida e passa integralmente sem acesso à rede externa.

## Assumptions

- Apenas GitHub e Cloudflare entram no escopo desta feature; outros provedores (AWS, Stripe, etc.) ficam para uma feature futura, se necessário.
- As statuspages públicas seguem o contrato `statuspage.io` `/api/v2/status.json`, com `status.indicator` e `status.description` como strings.
- Não há autenticação; o uso é somente leitura de uma página pública.
- Retentativa: exatamente uma nova tentativa, apenas para falha de rede, timeout ou HTTP 5xx; respostas 4xx não são retentadas.
- A descrição da tool segue o mesmo padrão já adotado nas demais tools do projeto (quando usar, o que retorna, defaults, enums com `.describe()`, erro como observação).
- A tool integra-se ao catálogo já usado pelas estratégias de raciocínio (ReAct / Plan-and-Execute); não altera o contrato HTTP `/chat`.
- O retorno compacto é texto livre de uma linha; não precisa ser um objeto estruturado no resultado da tool.
