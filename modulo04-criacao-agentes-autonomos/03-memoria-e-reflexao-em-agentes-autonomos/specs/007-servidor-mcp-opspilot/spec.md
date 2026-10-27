# Feature Specification: Servidor MCP do OpsPilot

**Feature Branch**: `007-servidor-mcp-opspilot`

**Created**: 2026-10-14

**Status**: Implemented

**Input**: User description: "Servidor MCP do OpsPilot: src/mcp/server.ts com @modelcontextprotocol/sdk, transport stdio, expondo list_alerts, open_incident e resolve_incident, reutilizando o mesmo OpsStore e os mesmos schemas zod das tools existentes (uma única fonte de verdade). Nome do server: opspilot. Script npm: mcp = tsx src/mcp/server.ts (carregar env antes se precisar). Regra crítica: nenhum console.log no server — no stdio o stdout é o canal do protocolo; diagnóstico vai para o stderr. Teste sobre o server valida o list de tools."

## Escopo

Expor um subconjunto das operações já existentes do OpsPilot (listar alertas, abrir incidente, resolver incidente) por um servidor MCP (Model Context Protocol) próprio, com transporte stdio, para que clientes compatíveis com MCP (IDEs, outros agentes) consultem e ajam sobre o mesmo estado operacional já usado pelo agente LangGraph — sem duplicar schema ou lógica de domínio.

Fora de escopo: as demais tools do catálogo (`list_incidents`, `consultar_runbook`, `check_provider_status`) nesta primeira versão do servidor MCP; transporte HTTP/SSE; autenticação/autorização de clientes MCP; qualquer mudança no contrato HTTP `/chat` ou nas estratégias de raciocínio.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Cliente MCP descobre as ferramentas operacionais (Priority: P1)

Como operador (ou IDE compatível com MCP), eu me conecto ao OpsPilot via stdio e, ao listar as ferramentas disponíveis, vejo exatamente `list_alerts`, `open_incident` e `resolve_incident`, com o servidor se identificando como `opspilot`.

**Why this priority**: Sem descoberta confiável das três tools, o servidor MCP não entrega valor; é o contrato mínimo de integração.

**Independent Test**: Instanciar o servidor em teste, solicitar a listagem de tools e verificar o nome do servidor e o conjunto exato das três tools, sem depender de modelo de linguagem ou rede externa.

**Acceptance Scenarios**:

1. **Given** o servidor MCP do OpsPilot iniciado, **When** o cliente pede a listagem de tools, **Then** a resposta contém exatamente `list_alerts`, `open_incident` e `resolve_incident`.
2. **Given** o mesmo servidor, **When** o cliente inspeciona a identidade do servidor, **Then** o nome reportado é `opspilot`.
3. **Given** a suíte de testes do servidor, **When** o teste de listagem roda, **Then** valida o catálogo das três tools sem chamar modelos externos.

---

### User Story 2 - Mesmas operações do plantão via MCP (Priority: P1)

Como cliente MCP, eu invoco `list_alerts`, `open_incident` e `resolve_incident` e obtenho o mesmo comportamento operacional das tools já usadas pelo agente — mesmos filtros/defaults, mesma validação de entrada e o mesmo estado operacional compartilhado.

**Why this priority**: Uma única fonte de verdade evita divergência de estado entre o agente e o servidor MCP.

**Independent Test**: Com o store de operações populado (seed), invocar as três operações via handlers MCP e comparar o resultado com o das tools equivalentes do agente sobre o mesmo store.

**Acceptance Scenarios**:

1. **Given** o store com alertas em curso e resolvidos, **When** o cliente MCP chama `list_alerts` (com filtro padrão ou explícito), **Then** o resultado espelha o da tool `list_alerts` do agente sobre o mesmo store.
2. **Given** o mesmo store, **When** o cliente MCP chama `open_incident` com título, serviço e severidade válidos, **Then** o incidente é criado no store operacional e o retorno confirma identificador e status, como na tool do agente.
3. **Given** um incidente aberto conhecido, **When** o cliente MCP chama `resolve_incident` com o identificador, **Then** o incidente passa a resolvido no mesmo store; um identificador inexistente devolve um erro descritivo como observação, sem interromper o servidor.

---

### User Story 3 - Stdio seguro para diagnóstico (Priority: P2)

Como pessoa que opera o servidor, preciso de mensagens de diagnóstico sem corromper o protocolo MCP: qualquer log vai para stderr; stdout permanece exclusivo do protocolo.

**Why this priority**: Em transporte stdio, qualquer escrita em stdout fora do protocolo quebra clientes MCP de forma silenciosa e difícil de depurar.

**Independent Test**: Revisar o código do servidor em busca de escrita em stdout fora do transporte (por exemplo, `console.log`); confirmar que diagnósticos usam `console.error` ou equivalente, direcionado a stderr.

**Acceptance Scenarios**:

1. **Given** o código de `src/mcp/server.ts` e dos módulos exclusivos do servidor MCP, **When** se busca por escrita acidental em stdout, **Then** não há ocorrências fora do transporte do protocolo.
2. **Given** o servidor em execução via script npm, **When** ocorre um evento digno de log (por exemplo, falha ao abrir o store), **Then** a mensagem aparece em stderr e o canal stdout do protocolo permanece intacto.

---

### User Story 4 - Subir o servidor com um comando (Priority: P3)

Como pessoa desenvolvedora, subo o servidor MCP com um único comando npm, com variáveis de ambiente carregadas quando necessário, no mesmo padrão dos demais entrypoints do projeto.

**Why this priority**: Facilita integração local; não é o valor operacional central, mas desbloqueia o uso.

**Independent Test**: Verificar a presença do script no `package.json` e que o entrypoint aponta para `src/mcp/server.ts`.

**Acceptance Scenarios**:

1. **Given** o repositório com a feature implementada, **When** se inspeciona `package.json`, **Then** existe um script `mcp` que inicia `src/mcp/server.ts`, carregando variáveis de ambiente no mesmo padrão já usado pelos outros scripts do projeto, quando o store em uso exigir.

### Edge Cases

- Cliente pede uma tool fora do catálogo desta feature (`list_incidents`, `consultar_runbook`, `check_provider_status`): não aparece na listagem; chamar um nome desconhecido segue o comportamento padrão do SDK MCP para tool inexistente.
- Entrada de uma tool falha na validação Zod: rejeitada na fronteira com erro descritivo ao cliente MCP, sem alterar o store.
- `resolve_incident` recebe identificador inexistente: devolve uma mensagem de erro legível (mesmo espírito da tool do agente), sem derrubar o processo do servidor.
- Variável de ambiente de localização dos dados não carregada e o servidor depende de um arquivo: o script npm carrega o `.env` no mesmo padrão dos outros scripts; falha ao abrir o store vai para stderr e encerra o processo de forma clara.
- Os schemas Zod e a lógica de domínio das três tools são a mesma fonte usada pelo agente — sem duplicar schemas só para o servidor MCP.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: O sistema MUST expor um servidor MCP em `src/mcp/server.ts`, usando o SDK oficial do Model Context Protocol, com transporte stdio.
- **FR-002**: O nome do servidor MCP MUST ser `opspilot`.
- **FR-003**: O servidor MUST registrar exatamente três tools nesta feature: `list_alerts`, `open_incident` e `resolve_incident`.
- **FR-004**: As três tools MUST reutilizar o mesmo store operacional (mesma interface/implementação já usada pelo agente) e os mesmos schemas Zod das tools existentes — uma única fonte de verdade, sem schemas paralelos.
- **FR-005**: O comportamento funcional (defaults, filtros, criação/resolução, mensagens de sucesso e erro) MUST permanecer alinhado ao das tools equivalentes do agente sobre o mesmo store.
- **FR-006**: O código do servidor MCP MUST NÃO usar `console.log` nem qualquer escrita em stdout fora do transporte do protocolo; diagnóstico MUST ir para stderr.
- **FR-007**: DEVE existir um script npm que inicie o servidor (`src/mcp/server.ts`); se o servidor precisar de variáveis de ambiente (por exemplo, a localização do arquivo de dados), o script MUST carregá-las antes, no mesmo padrão já usado pelos outros scripts do projeto.
- **FR-008**: DEVE existir teste focado no servidor que valide a listagem de tools (catálogo igual às três tools esperadas e nome do servidor igual a `opspilot`).
- **FR-009**: A dependência do SDK de Model Context Protocol MUST ser adicionada ao projeto conforme necessário para o servidor e os testes.
- **FR-010**: Tools do catálogo além das três (`list_incidents`, `consultar_runbook`, `check_provider_status`) ficam fora do escopo do catálogo MCP desta feature.

### Key Entities

- **Servidor MCP do OpsPilot**: processo nomeado `opspilot`, transporte stdio, que publica o catálogo operacional mínimo.
- **Vínculo de tool MCP**: ligação entre o nome MCP (`list_alerts` | `open_incident` | `resolve_incident`), o schema Zod compartilhado e o handler que opera sobre o store.
- **Store operacional** (existente): fonte de verdade de serviços, alertas e incidentes; compartilhada conceitualmente com o agente.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Um cliente MCP consegue listar as tools e encontra exatamente as três operações de plantão em 100% das listagens bem-sucedidas, nos testes.
- **SC-002**: Com o store populado (seed), invocar as três operações via MCP produz o mesmo efeito observável que as tools do agente sobre o mesmo estado, em 100% dos casos de teste cobertos.
- **SC-003**: 100% das mensagens de diagnóstico emitidas pelo servidor em cenários de falha vão para stderr; não há ocorrência de `console.log` no código do servidor.
- **SC-004**: A suíte de testes do servidor valida a listagem de tools e passa sem rede externa e sem modelo de linguagem.
- **SC-005**: Uma pessoa desenvolvedora sobe o servidor com um único comando npm documentado.

## Assumptions

- O escopo desta feature é apenas as três tools pedidas; as demais tools do catálogo entram em uma feature futura, se necessário.
- "Mesma fonte de verdade" pode exigir extrair schemas/handlers hoje fechados em `src/services/tools.ts` para um ponto compartilhado; esse detalhe de reorganização cabe ao plano, desde que não haja duplicação de schema.
- O servidor MCP é um entrypoint separado do HTTP `/chat`; não substitui o grafo de raciocínio nesta feature.
- Persistência: o servidor usa o mesmo store real (SQLite) já injetado na composição de produção, ou um store injetável para testes; os testes preferem um store efêmero/isolado por execução.
- Transporte apenas stdio nesta feature, sem HTTP/SSE.
- Autenticação/autorização de clientes MCP fica fora de escopo (stdio local confiável).
