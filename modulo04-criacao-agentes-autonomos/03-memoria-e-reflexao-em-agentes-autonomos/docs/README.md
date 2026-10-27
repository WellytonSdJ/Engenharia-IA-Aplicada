# Documentação — 02-function-calling-e-tool-use

Este projeto nasce como uma cópia de [`01-padroes-de-raciocinio-e-execucao`](../../01-padroes-de-raciocinio-e-execucao/) — mesmo OpsPilot, mesmo `src/`, mesmas três estratégias de raciocínio (ReAct, Plan-and-Execute, Reflection), mesmo contrato `ReasoningStrategy`/`TraceEvent`, mesma arena e o mesmo bench. Para tudo isso, a documentação do `01` continua valendo e **não é repetida aqui** — veja a tabela abaixo.

O que esta unidade acrescenta é a parte de **function calling e tool use** de verdade: o endpoint `POST /chat`, tools com persistência real (SQLite via `node:sqlite`, sem servidor externo), uma tool nova de runbook, uma tool de status de provedores externos e, em seguida, um servidor MCP e conversa persistente.

> **Estado atual do nosso projeto:** herdado do `01` (specs 001–003 implementadas; ver [docs do 01](../../01-padroes-de-raciocinio-e-execucao/docs/README.md)). Neste projeto já foram implementadas: `004-http-chat-endpoint` (`POST /chat`), `005-persistencia-sqlite` (`SqliteOpsStore` substitui Sequelize/MySQL e o store em JSON; tool `consultar_runbook`), `006-status-provedores-externos` (tool `check_provider_status`), `007-servidor-mcp-opspilot` (servidor MCP `opspilot` com `list_alerts`, `open_incident`, `resolve_incident` via stdio) e `008-conversa-persistente` (`conversationId` opcional em `/chat`, histórico injetado no prompt, métrica `historyMessages`). Esta é a última feature planejada desta unidade.

---

## O que já está documentado no `01` (não repetido aqui)

| Documento (no `01`) | Conteúdo |
| --- | --- |
| [00-START-HERE.md](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md) | O que é o OpsPilot, trilha de leitura, mapa do código |
| [spec-driven-com-spec-kit.md](../../01-padroes-de-raciocinio-e-execucao/docs/spec-driven-com-spec-kit.md) | Spec-Driven Development com o Spec Kit |
| [contrato-de-estrategia-e-trace.md](../../01-padroes-de-raciocinio-e-execucao/docs/contrato-de-estrategia-e-trace.md) | `ReasoningStrategy`, `TraceEvent`, métricas |
| [tools-e-store-deterministico.md](../../01-padroes-de-raciocinio-e-execucao/docs/tools-e-store-deterministico.md) | Tools com `tool()` + Zod, store injetado, seed, testes sem rede |
| [react.md](../../01-padroes-de-raciocinio-e-execucao/docs/react.md) | O padrão ReAct |
| [plan-and-execute.md](../../01-padroes-de-raciocinio-e-execucao/docs/plan-and-execute.md) | Plan-and-Execute como `StateGraph` |
| [reflection.md](../../01-padroes-de-raciocinio-e-execucao/docs/reflection.md) | A camada de Reflection |
| [arena-e-bench.md](../../01-padroes-de-raciocinio-e-execucao/docs/arena-e-bench.md) | Arena e bench, cenários C1/C2/C3 |
| [variaveis-de-ambiente-com-env-file.md](../../01-padroes-de-raciocinio-e-execucao/docs/variaveis-de-ambiente-com-env-file.md) | Como o `.env` chega ao `process.env` |
| [roteiro-de-implementacao.md](../../01-padroes-de-raciocinio-e-execucao/docs/roteiro-de-implementacao.md) | Ponte `tasks.md` ↔ código de referência |
| [glossario.md](../../01-padroes-de-raciocinio-e-execucao/docs/glossario.md) | Termos de raciocínio/Spec Kit já cobertos |

## Documentos deste projeto

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Por que este projeto existe, o que ele herda do `01` e o que as specs 004–008 acrescentam |

## `POST /chat`

A spec `004-http-chat-endpoint` ([specs/004-http-chat-endpoint/](../specs/004-http-chat-endpoint/)) expõe as estratégias de raciocínio por HTTP.

```powershell
npm run dev                  # inicia o servidor (padrão PORT=3000)
```

```bash
curl -X POST http://localhost:3000/chat \
  -H "content-type: application/json" \
  -d '{"message": "quantos alertas criticos estao disparando?", "strategy": "react", "reflect": false}'
```

| Status | Quando |
| --- | --- |
| 200 | sucesso: `{ answer, trace, metrics }` |
| 400 | corpo inválido (JSON malformado, `message` vazio, tipos errados) |
| 422 | `strategy` fora do registro (`react`, `plan-and-execute`) |
| 504 | execução excedeu 60 s |
| 500 | erro interno inesperado |

Contrato completo em [contracts/chat-http.md](../specs/004-http-chat-endpoint/contracts/chat-http.md).

---

## Servidor MCP

A spec `007-servidor-mcp-opspilot` ([specs/007-servidor-mcp-opspilot/](../specs/007-servidor-mcp-opspilot/)) expõe um subconjunto das tools operacionais — `list_alerts`, `open_incident` e `resolve_incident` — por um servidor [MCP](https://modelcontextprotocol.io/) próprio (`src/mcp/server.ts`), com transporte stdio, para clientes compatíveis (IDEs, outros agentes). O servidor se identifica como `opspilot` e reutiliza o mesmo `SqliteOpsStore` e os mesmos schemas Zod das tools do agente — `createOpsToolDefinitions(store)` em `src/services/tools.ts` é a única fonte de verdade, consumida tanto por `createOpsTools` (LangChain) quanto pelo servidor MCP.

```powershell
npm run mcp                  # inicia o servidor MCP via stdio (aguarda um cliente conectar)
```

Regra crítica de stdio: nenhuma linha de `src/mcp/server.ts` usa `console.log` — no transporte stdio, o stdout é o canal do protocolo; qualquer diagnóstico vai para `console.error` (stderr).

Fora do escopo desta feature: as demais tools do catálogo (`list_incidents`, `consultar_runbook`, `check_provider_status`), transporte HTTP/SSE e autenticação de clientes MCP.

---

## Conversa persistente

A spec `008-conversa-persistente` ([specs/008-conversa-persistente/](../specs/008-conversa-persistente/)) estende `POST /chat` (spec `004-http-chat-endpoint`) com continuidade de conversa entre requisições. O corpo aceita um `conversationId` opcional:

```bash
curl -X POST http://localhost:3000/chat \
  -H "content-type: application/json" \
  -d '{"message": "abra um incidente sev2 para o checkout", "conversationId": "conversation-..."}'
```

- Sem `conversationId`: o `ConversationStore` cria uma conversa nova antes de executar a estratégia.
- Com `conversationId` existente: a conversa é reutilizada; um id bem formado mas inexistente resulta em `422 UNKNOWN_CONVERSATION`, sem executar a estratégia.
- Antes de cada execução, até 12 mensagens mais recentes da conversa (`ConversationStore.lastMessages`) são compostas no prompt enviado à estratégia; a resposta 200 inclui `conversationId` e `metrics.historyMessages` (quantas mensagens de histórico entraram naquele turno).
- Depois de uma execução bem-sucedida, o turno completo (mensagem do usuário + resposta do agente) é persistido via `ConversationStore.append`; se a estratégia falhar, nada é persistido — nunca há uma resposta 200 com histórico parcial.

O `ConversationStore` (`src/models/conversation-store.ts`) segue o mesmo estilo do `SqliteOpsStore`: SQLite nativo (`node:sqlite`/`DatabaseSync`), DDL idempotente, apenas prepared statements, tabelas `conversations` e `messages` (com `FOREIGN KEY` e `CHECK (role IN ('user','assistant'))`). Em produção ele abre o mesmo arquivo apontado por `OPSPILOT_DB` já usado pelo `SqliteOpsStore` (conexão própria sobre o mesmo caminho); os testes usam `:memory:`.

| Status | Quando (além dos já documentados em `POST /chat`) |
| --- | --- |
| 422 | `UNKNOWN_CONVERSATION`: `conversationId` bem formado mas inexistente |
| 400 | `conversationId` com formato inválido (ex.: string vazia) |

---

## Contexto do projeto

Igual ao [`01`](../../01-padroes-de-raciocinio-e-execucao/docs/README.md#contexto-do-projeto), com as mudanças já implementadas neste projeto: **Sequelize + MySQL** e o store em JSON saíram do projeto — a persistência real passa a ser **SQLite** via `node:sqlite` (`DatabaseSync`), self-contained no processo, sem servidor externo; e as estratégias de raciocínio agora também são acessíveis por `POST /chat` (Express 5), além da CLI (arena/bench).
