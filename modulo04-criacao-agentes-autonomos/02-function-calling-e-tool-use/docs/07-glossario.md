# Glossário — 02-function-calling-e-tool-use

Termos já cobertos no [glossário do projeto 01](../../01-padroes-de-raciocinio-e-execucao/docs/10-glossario.md) (estratégias, trace, `createReactAgent`, catálogo de tools, `.env`, seed, `InMemoryOpsStore`) não são repetidos aqui. O vocabulário base de MCP (`McpServer`, `registerTool`, `StdioServerTransport`, `Client`) está no [glossário do módulo 03 / 05-mcps-do-zero](../../../modulo03-mcp-na-pratica/05-mcps-do-zero/docs/04-glossario.md).

---

## Function calling e design de tools

| Termo | Definição |
| --- | --- |
| **Function calling / tool calling** | O modelo devolve um pedido estruturado ("chame X com estes argumentos") em vez de texto. Quem executa é o código. |
| **"O schema é o prompt"** | Nome, descrição e `.describe()` de uma tool são tudo o que o modelo vê dela, então funcionam como prompt. |
| **6 regras de descrição** | Checklist do projeto: quando usar, mapear vocabulário, `.describe()` em todo campo, enums por extenso, sem overclaim, erros antecipados. |
| **Overclaim** | Descrição que promete um efeito que a tool não garante. O modelo repete a promessa ao usuário. |
| **`OpsToolDefinition`** | Definição neutra de tool (`name`, `description`, `inputSchema`, `run`), adaptada para LangChain e para MCP. |
| **Fonte única de verdade** | Uma só lista de definições (`createOpsToolDefinitions`) alimenta todos os consumidores. |
| **`list_incidents`** | Tool nova: lista incidentes por status (`open` por padrão), para não abrir duplicado. |
| **`consultar_runbook`** | Tool nova: devolve o procedimento de um serviço, ou `found: false` quando o serviço existe mas não tem runbook. |
| **Runbook** | Passo a passo de investigação e mitigação de um serviço. |

---

## API HTTP

| Termo | Definição |
| --- | --- |
| **`POST /chat`** | Porta HTTP do agente: `{ message, strategy?, reflect?, conversationId? }` → `{ answer, trace, metrics, conversationId }`. |
| **Validação na fronteira** | Checar a entrada (Zod) no ponto onde ela entra no sistema, antes de qualquer lógica. |
| **Registry de estratégias** | Dicionário nome → estratégia (`strategyRegistry`). A rota resolve pelo nome, sem `if/else`. |
| **`resolveStrategy`** | Acha a estratégia no registry (default `react`) e aplica `withReflection` se `reflect: true`. |
| **`withTimeout`** | `Promise.race` entre o agente e um cronômetro; ao perder, lança `ExecutionTimeoutError` (→ 504). |
| **400 × 422 × 504** | Corpo mal escrito × pedido bem escrito de algo que não existe × agente lento demais. |
| **Express 5** | Framework HTTP usado no servidor (`createServer`, `express.json()`). |

---

## Persistência

| Termo | Definição |
| --- | --- |
| **`node:sqlite` / `DatabaseSync`** | SQLite embutido no Node 22+, com API síncrona. Sem pacote nem servidor externo. |
| **`SqliteOpsStore`** | Implementação de `OpsStore` sobre SQLite: tabelas `services`, `alerts`, `incidents` e `runbooks`. |
| **`OPSPILOT_DB`** | Caminho do arquivo do banco (default `./data/opspilot.db`). |
| **`:memory:`** | Banco SQLite só em memória, descartável. Usado nos testes. |
| **DDL idempotente** | `CREATE TABLE IF NOT EXISTS`: rodar várias vezes não muda nem apaga nada. |
| **`CHECK`** | Restrição no banco que recusa valores fora do domínio (ex.: `severity`). |
| **Prepared statement** | SQL com `?` preenchidos como dados (`prepare(...).run(...)`). Impede SQL injection. |
| **SQL injection** | Ataque em que texto de entrada vira código SQL. Relevante porque argumentos de tool vêm do modelo. |
| **`BEGIN IMMEDIATE`** | Abre uma transação que já reserva a escrita, para que ler e atualizar aconteçam sem interferência. |
| **`reset()`** | Apaga as tabelas e reinsere o cenário Mercadinho numa transação (é o que o `npm run seed` faz). |

---

## Tool externa

| Termo | Definição |
| --- | --- |
| **`check_provider_status`** | Tool que consulta a status page pública do GitHub ou da Cloudflare. |
| **Status page (`/api/v2/status.json`)** | Endpoint público com o estado do provedor: `indicator` (`none`, `minor`, `major`, `critical`) + `description`. |
| **`AbortSignal.timeout(ms)`** | Sinal que cancela o `fetch` depois de `ms` milissegundos. |
| **Retry seletivo** | Tentar de novo só em falhas passageiras (rede, 5xx), nunca em 4xx. |
| **Erro como observação** | A tool devolve uma frase de erro em vez de lançar exceção, para o modelo seguir raciocinando. |
| **`fetchFn` injetável** | O `fetch` entra por parâmetro, para os testes usarem um fake sem rede. |
| **`.passthrough()`** | Opção do Zod que aceita campos extras no objeto, validando só os exigidos. |

---

## MCP

| Termo | Definição |
| --- | --- |
| **Servidor MCP do OpsPilot** | `createMcpServer(store)`: expõe `list_alerts`, `open_incident` e `resolve_incident` com o nome `opspilot`. |
| **`MCP_TOOL_NAMES`** | Allowlist das tools que saem pelo MCP. |
| **Regra do stdout** | Em stdio, o stdout é o canal do protocolo: `console.log` é proibido e o diagnóstico vai para `console.error`. |
| **`InMemoryTransport.createLinkedPair()`** | Par de transportes ligados em memória, para testar cliente e servidor MCP sem processo. |
| **Prova circular** | Exercício: um cliente MCP abre um incidente e o `/chat` o enxerga, porque os dois compartilham o banco. |

---

## Conversa

| Termo | Definição |
| --- | --- |
| **`conversationId`** | Id da conversa. Omitido = conversa nova; inexistente = 422 `UNKNOWN_CONVERSATION`. |
| **`ConversationStore`** | Store SQLite de `conversations` e `messages`, no mesmo `OPSPILOT_DB`. |
| **Janela de histórico** | As últimas `MAX_HISTORY_MESSAGES` (12) mensagens, reinjetadas no prompt. |
| **`composeInput`** | Monta o prompt: `Conversation so far:` + histórico + a mensagem nova. |
| **`seq`** | Número de ordem da mensagem na conversa. Mais confiável que o timestamp. |
| **`historyMessages`** | Métrica: quantas mensagens antigas entraram no turno. |
| **Memória de curto prazo** | O fio de uma conversa (aqui). A de longo prazo, por usuário, vem no [projeto 03](../../03-memoria-e-reflexao-em-agentes-autonomos/docs/06-glossario.md). |

---

⬅️ Anterior: [06 · Conversa persistente](./06-conversa-persistente.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: projeto 03, [03-memoria-e-reflexao-em-agentes-autonomos](../../03-memoria-e-reflexao-em-agentes-autonomos/docs/00-START-HERE.md)**
