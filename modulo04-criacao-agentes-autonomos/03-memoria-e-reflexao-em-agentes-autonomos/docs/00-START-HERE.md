# Por onde começar

Este é o segundo projeto do módulo 04. O anterior foi [`01-padroes-de-raciocinio-e-execucao`](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md) — leia-o primeiro se ainda não leu: lá nasce o OpsPilot, as três estratégias de raciocínio (ReAct, Plan-and-Execute, Reflection), o contrato `ReasoningStrategy`/`TraceEvent`, a arena e o bench. Este projeto começa como uma **cópia exata** daquele — mesmo `src/`, mesmo `package.json` — e evolui a partir daí.

---

## O que muda nesta unidade

O `01` respondia "como o agente pensa?". Esta unidade pergunta **"com que mãos ele age de verdade, e como outros sistemas conversam com ele?"** — hoje as tools do OpsPilot escreviam num store em memória (ou um JSON) que não sobrevivia a um reinício, o agente só era acessível por CLI, e não havia como consultar nada fora da base própria de alertas/incidentes.

Specs implementadas nesta unidade:

| Spec | Mudança | Por quê |
| --- | --- | --- |
| [004-http-chat-endpoint](../specs/004-http-chat-endpoint/spec.md) | `POST /chat` (Express 5), registro de estratégias em `src/agents/index.ts`, `reflect` no corpo, timeout de 60 s | Expõe o agente por HTTP, não só por CLI |
| [005-persistencia-sqlite](../specs/005-persistencia-sqlite/spec.md) | `SqliteOpsStore` via `node:sqlite` (`DatabaseSync`), substituindo o mock em memória e o JSON em arquivo; tool nova `consultar_runbook(service)` | Persistência real, self-contained no processo (sem MySQL); fecha a lacuna entre "constatar o problema" e "agir sobre ele" |
| [006-status-provedores-externos](../specs/006-status-provedores-externos/spec.md) | Tool nova `check_provider_status(provider)`, consultando a statuspage pública de GitHub/Cloudflare, com timeout e retentativa | Ajuda a distinguir falha interna de indisponibilidade de terceiros |
| [007-servidor-mcp-opspilot](../specs/007-servidor-mcp-opspilot/spec.md) | Servidor MCP (`src/mcp/server.ts`, stdio) expondo `list_alerts`/`open_incident`/`resolve_incident` para clientes compatíveis com Model Context Protocol | Permite que outros agentes/IDEs operem sobre o mesmo store, via protocolo padrão |
| [008-conversa-persistente](../specs/008-conversa-persistente/spec.md) | `conversationId` opcional em `/chat`, histórico de até 12 mensagens injetado no prompt, métrica `historyMessages` | Dá continuidade ao diálogo entre requisições, sem o cliente reenviar o histórico manualmente |

Esta é a última feature planejada desta unidade.

---

## Trilha de leitura

1. Leia (ou revise) a trilha completa do [`01`](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md#trilha-de-leitura) — ela cobre spec-driven, contrato de estratégia/trace, tools/store, ReAct, Plan-and-Execute, Reflection, arena/bench, `.env` e glossário. Nada disso é repetido aqui.
2. [specs/004-http-chat-endpoint/spec.md](../specs/004-http-chat-endpoint/spec.md) — o endpoint HTTP de chat.
3. [specs/005-persistencia-sqlite/spec.md](../specs/005-persistencia-sqlite/spec.md) — persistência SQLite e a tool `consultar_runbook`.
4. [specs/006-status-provedores-externos/spec.md](../specs/006-status-provedores-externos/spec.md) — a tool `check_provider_status`.
5. [specs/007-servidor-mcp-opspilot/spec.md](../specs/007-servidor-mcp-opspilot/spec.md) — o servidor MCP.
6. [specs/008-conversa-persistente/spec.md](../specs/008-conversa-persistente/spec.md) — a última spec desta unidade: conversa persistente em `/chat`.

---

## Como rodar

Igual ao [`01`](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md#como-rodar-e-ver-o-que-importa), com `npm run seed` agora populando `OPSPILOT_DB` (default `./data/opspilot.db`, fora do versionamento) em vez do JSON, e com o servidor HTTP disponível via `npm run dev`:

```powershell
npm test                     # testes sem chave e sem rede (SqliteOpsStore via :memory:)
npm run seed                 # popula o OPSPILOT_DB (serviços, alertas e runbooks)
npm run dev                  # inicia o servidor POST /chat (padrão PORT=3000)
npm run arena -- "quantos alertas criticos estão disparando" --strategies react
npm run bench                 # C1/C2/C3 × react e plan-and-execute (InMemoryOpsStore)
```

Detalhes de uso do `POST /chat` (exemplo `curl`, tabela de status) estão em [README.md](./README.md#post-chat).
