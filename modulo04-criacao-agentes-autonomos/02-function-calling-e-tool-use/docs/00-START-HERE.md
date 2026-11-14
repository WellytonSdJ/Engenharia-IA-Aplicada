# Por onde começar

Este é o segundo projeto do módulo 04. O anterior foi [`01-padroes-de-raciocinio-e-execucao`](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md): lá nascem o OpsPilot, as três estratégias de raciocínio, o contrato `ReasoningStrategy`/`TraceEvent`, a arena e o bench. Este projeto começa como uma **cópia** daquele (mesmo `src/`, specs 001–003) e acrescenta cinco specs.

---

## O que estamos construindo e por quê

O 01 respondeu "como o agente pensa?". Esta unidade pergunta:

> **"Com que mãos ele age de verdade, e por quais portas os outros falam com ele?"**

No fim do 01, o OpsPilot raciocinava bem, mas:
- o que ele fazia **sumia** quando o processo terminava;
- só dava para falar com ele **pelo terminal**;
- ele só enxergava **o lado de dentro** (nossos alertas), nunca o mundo externo;
- cada pergunta começava do **zero**, sem lembrar a anterior.

| Spec | O que acrescenta | Por quê |
| --- | --- | --- |
| [004-http-chat-endpoint](../specs/004-http-chat-endpoint/spec.md) | `POST /chat` com Zod, registry de estratégias, `reflect` e timeout | Uma porta HTTP: a API que é, ela mesma, um agente |
| [005-persistencia-sqlite](../specs/005-persistencia-sqlite/spec.md) | `SqliteOpsStore` (`node:sqlite`), tools `list_incidents` e `consultar_runbook`, descrições pelas 6 regras | Ações que ficam + tools que o modelo entende |
| [006-status-provedores-externos](../specs/006-status-provedores-externos/spec.md) | Tool `check_provider_status` com timeout, retry e erro como observação | Responder "o problema é nosso ou do provedor?" |
| [007-servidor-mcp-opspilot](../specs/007-servidor-mcp-opspilot/spec.md) | Servidor MCP `opspilot` (stdio) com 3 tools | Outros agentes (Copilot, IDEs) operam o mesmo banco |
| [008-conversa-persistente](../specs/008-conversa-persistente/spec.md) | `conversationId` e as últimas 12 mensagens no prompt | Continuidade entre requisições (memória de curto prazo) |

A próxima unidade, [`03-memoria-e-reflexao-em-agentes-autonomos`](../../03-memoria-e-reflexao-em-agentes-autonomos/docs/00-START-HERE.md), parte daqui e acrescenta memória de **longo prazo** por usuário.

---

## Trilha de leitura

Cada doc tem um bloco **"Em 30 segundos"** no topo e **checkpoints** ao longo do texto. Com pouco tempo, leia só esses dois.

| # | Documento | Por que ler |
| --- | --- | --- |
| 0 | Trilha do [01](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md) | Estratégias, trace, tools básicas e `.env` não são repetidos aqui |
| 1 | [01-function-calling-e-design-de-tools.md](./01-function-calling-e-design-de-tools.md) | O coração da unidade: como o modelo escolhe e chama tools, e como escrever tools que ele entende |
| 2 | [02-persistencia-sqlite.md](./02-persistencia-sqlite.md) | Onde as tools passam a escrever de verdade |
| 3 | [03-tool-externa-resiliente.md](./03-tool-externa-resiliente.md) | Uma tool que sai para a internet sem travar o agente |
| 4 | [04-api-chat-http.md](./04-api-chat-http.md) | A porta HTTP para tudo isso |
| 5 | [05-servidor-mcp.md](./05-servidor-mcp.md) | A porta MCP: o OpsPilot oferecendo tools a outros agentes |
| 6 | [06-conversa-persistente.md](./06-conversa-persistente.md) | O fio da conversa entre requisições |
| 7 | [07-glossario.md](./07-glossario.md) | Consulta rápida dos termos novos |

---

## Mapa do código

Só o que é **novo ou mudou** nesta unidade, em ordem de leitura. O resto está no mapa do [01](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md).

| Arquivo | Spec | O que faz |
| --- | --- | --- |
| [`src/services/tools.ts`](../src/services/tools.ts) | 005 / 006 | `OpsToolDefinition`, as 6 tools, descrições pelas 6 regras, `fetchProviderStatus` |
| [`src/models/in-memory-store.ts`](../src/models/in-memory-store.ts) | 005 | Contrato `OpsStore` ganha `listIncidents` e `findRunbook` |
| [`src/models/domain.ts`](../src/models/domain.ts) | 005 / 008 | + runbook, filtro de incidentes, conversa e mensagem |
| [`src/models/sqlite-store.ts`](../src/models/sqlite-store.ts) | 005 | `SqliteOpsStore`: DDL, `CHECK`, prepared statements, transações, seed |
| [`src/services/seed.ts`](../src/services/seed.ts) | 005 | `npm run seed` → `reset()` no SQLite |
| [`src/services/errors.ts`](../src/services/errors.ts) | 004 / 008 | + `UnknownStrategyError`, `ExecutionTimeoutError`, `UnknownConversationError` |
| [`src/agents/index.ts`](../src/agents/index.ts) | 004 | Registry de estratégias e `resolveStrategy` |
| [`src/services/chat.ts`](../src/services/chat.ts) | 004 / 008 | `runChat`, `withTimeout`, `composeInput`, janela de 12 |
| [`src/controllers/chat-controller.ts`](../src/controllers/chat-controller.ts) | 004 / 008 | `ChatRequestSchema` e erro → status HTTP |
| [`src/http/server.ts`](../src/http/server.ts) | 004 | `createServer(deps)` com Express 5 |
| [`src/index.ts`](../src/index.ts) | 004 | Entry point: sobe o HTTP na `PORT` |
| [`src/models/conversation-store.ts`](../src/models/conversation-store.ts) | 008 | `ConversationStore`: `conversations` + `messages` |
| [`src/mcp/server.ts`](../src/mcp/server.ts) | 007 | `createMcpServer`, allowlist de 3 tools, stdio |
| [`src/controllers/bench-controller.ts`](../src/controllers/bench-controller.ts), [`src/bench.ts`](../src/bench.ts) | 005 | O bench continua no `InMemoryOpsStore` |
| Testes: [`tools`](../src/services/tools.test.ts), [`sqlite-store`](../src/models/sqlite-store.test.ts), [`chat`](../src/services/chat.test.ts), [`server`](../src/http/server.test.ts), [`agents/index`](../src/agents/index.test.ts), [`conversation-store`](../src/models/conversation-store.test.ts), [`mcp/server`](../src/mcp/server.test.ts) | 004–008 | Tudo sem rede: SQLite `:memory:`, estratégia fake, `fetch` fake, MCP em memória |

Saíram do projeto: `src/models/json-store.ts`, `src/models/sequelize-store.ts` e `src/data/ops-store.json`. A única persistência real agora é o SQLite.

---

## O fluxo em uma linha

```text
POST /chat {message, conversationId?}
  → Zod → registry → histórico (12 msgs) → estratégia
      → tools → SqliteOpsStore (ou a internet, via check_provider_status)
  → grava o turno → 200 { answer, trace, metrics, conversationId }

npm run mcp → Copilot/IDE → 3 tools → o mesmo SqliteOpsStore
```

---

## Como rodar e ver o que importa

Requisitos e `.env` iguais aos do [01](../../01-padroes-de-raciocinio-e-execucao/docs/09-variaveis-de-ambiente-com-env-file.md), mais `OPSPILOT_DB` (opcional; default `./data/opspilot.db`).

```powershell
npm test       # 99 testes, sem chave e sem rede
npm run seed   # cria/zera o banco com o cenário Mercadinho
npm run dev    # POST /chat em http://localhost:3000
npm run mcp    # servidor MCP via stdio (espera um cliente)
```

Abra um incidente e continue a conversa:

```bash
curl -X POST http://localhost:3000/chat -H "content-type: application/json" \
  -d '{"message": "abra um incidente sev2 para o checkout"}'
# copie o conversationId da resposta
curl -X POST http://localhost:3000/chat -H "content-type: application/json" \
  -d '{"message": "agora liste os incidentes abertos", "conversationId": "conversation-..."}'
```

Reinicie o `npm run dev` e liste de novo: o incidente continua lá (SQLite).

> ⚠️ O `npm run seed` **zera** o banco: os incidentes abertos pelo `/chat` somem.

---

## 🧪 Exercícios

- **Prova circular com MCP:** registre o `npm run mcp` no seu cliente MCP, peça ao Copilot para abrir um incidente e veja o incidente aparecer no `/chat`. O passo a passo está em [05-servidor-mcp.md](./05-servidor-mcp.md).
- **Ampliar o catálogo MCP:** coloque `list_incidents` e `consultar_runbook` na allowlist (`MCP_TOOL_NAMES`) e ajuste o teste que confere o catálogo.
- **Revisar uma descrição:** escolha uma tool, aplique as 6 regras de [01-function-calling-e-design-de-tools.md](./01-function-calling-e-design-de-tools.md) e compare o comportamento na arena antes e depois.

---

⬅️ Anterior: [Documentação](./README.md) · **➡️ Próximo passo: [01 · Function calling e design de tools](./01-function-calling-e-design-de-tools.md)**
