# Documentação — 02-function-calling-e-tool-use

Este projeto nasce como cópia do [`01-padroes-de-raciocinio-e-execucao`](../../01-padroes-de-raciocinio-e-execucao/): mesmo OpsPilot, mesmas estratégias (ReAct, Plan-and-Execute, Reflection), mesmo contrato `ReasoningStrategy`/`TraceEvent`, mesma arena e mesmo bench. Nada disso é re-explicado aqui.

O que esta unidade acrescenta são **mãos de verdade** para o agente: tools bem descritas que gravam num banco real, uma tool que consulta a internet com segurança, uma porta HTTP (`POST /chat`), uma porta MCP para outros agentes usarem as mesmas tools e uma conversa que continua entre requisições.

> **Estado atual:** specs 001–003 herdadas do 01; `004` a `008` implementadas. `npm test`: 99 testes, sem rede.

Comece por [00-START-HERE.md](./00-START-HERE.md).

---

## Documentos deste projeto

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | O que muda nesta unidade, trilha de leitura, mapa do código e como rodar |
| [01-function-calling-e-design-de-tools.md](./01-function-calling-e-design-de-tools.md) | O modelo pede e o código executa; "o schema é o prompt"; as 6 regras; `list_incidents`, `consultar_runbook`; fonte única `OpsToolDefinition` |
| [04-api-chat-http.md](./04-api-chat-http.md) | `POST /chat`: Zod na fronteira, registry de estratégias, `reflect`, `withTimeout`, erro de domínio → 400/422/504/500 |
| [02-persistencia-sqlite.md](./02-persistencia-sqlite.md) | `node:sqlite`, DDL idempotente, `CHECK`, prepared statements, `BEGIN IMMEDIATE`, seed, `:memory:` e por que o bench fica em memória |
| [03-tool-externa-resiliente.md](./03-tool-externa-resiliente.md) | `check_provider_status`: timeout, retry seletivo, Zod na resposta, erro como observação, `fetch` injetável |
| [05-servidor-mcp.md](./05-servidor-mcp.md) | O OpsPilot como provedor MCP: 3 tools da mesma fonte, stdio sem `console.log`, testes em memória, 🧪 prova circular |
| [06-conversa-persistente.md](./06-conversa-persistente.md) | `conversationId`, janela de 12 mensagens, gravar só depois do sucesso, 422 `UNKNOWN_CONVERSATION` |
| [07-glossario.md](./07-glossario.md) | Termos novos desta unidade |

## O que já está documentado antes (não repetido aqui)

| Documento | Conteúdo |
| --- | --- |
| [01 / 00-START-HERE.md](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md) | O que é o OpsPilot, trilha e mapa do código base |
| [01 / 01-spec-driven-com-spec-kit.md](../../01-padroes-de-raciocinio-e-execucao/docs/01-spec-driven-com-spec-kit.md) | O processo Spec Kit e a constituição |
| [01 / 02-contrato-de-estrategia-e-trace.md](../../01-padroes-de-raciocinio-e-execucao/docs/02-contrato-de-estrategia-e-trace.md) | `ReasoningStrategy`, `TraceEvent` e métricas (a resposta do `/chat`) |
| [01 / 03-tools-e-store-deterministico.md](../../01-padroes-de-raciocinio-e-execucao/docs/03-tools-e-store-deterministico.md) | `tool()` + Zod, store injetado, `ToolBoundaryError`, testes sem rede |
| [01 / 04-react.md](../../01-padroes-de-raciocinio-e-execucao/docs/04-react.md) | O loop ReAct que chama as tools |
| [01 / 05-plan-and-execute.md](../../01-padroes-de-raciocinio-e-execucao/docs/05-plan-and-execute.md) | Plan-and-Execute como `StateGraph` |
| [01 / 06-reflection.md](../../01-padroes-de-raciocinio-e-execucao/docs/06-reflection.md) | O decorator que o `reflect: true` aplica |
| [01 / 07-arena-e-bench.md](../../01-padroes-de-raciocinio-e-execucao/docs/07-arena-e-bench.md) | Arena e bench (seguem no store em memória) |
| [01 / 09-variaveis-de-ambiente-com-env-file.md](../../01-padroes-de-raciocinio-e-execucao/docs/09-variaveis-de-ambiente-com-env-file.md) | Como o `.env` (e o `OPSPILOT_DB`) chega ao processo |
| [01 / 10-glossario.md](../../01-padroes-de-raciocinio-e-execucao/docs/10-glossario.md) | Termos de raciocínio e Spec Kit |
| [módulo 03 / 01-construindo-mcp-server-do-zero.md](../../../modulo03-mcp-na-pratica/05-mcps-do-zero/docs/01-construindo-mcp-server-do-zero.md) | O básico de servidor MCP |

---

## Contexto do projeto

| Tecnologia | Papel neste projeto |
| --- | --- |
| `node:sqlite` (`DatabaseSync`) | **Nova.** Persistência real embutida no Node, no lugar de Sequelize/MySQL e do JSON em arquivo |
| Express 5 | **Nova.** Servidor HTTP do `POST /chat` |
| `@modelcontextprotocol/sdk` | **Nova.** Servidor MCP `opspilot` (stdio) |
| `fetch` + `AbortSignal.timeout` | **Novo uso.** A primeira tool que chama a internet |
| Zod | Já existia; agora valida também o corpo HTTP e a resposta da API externa |
| LangChain / LangGraph | Já existiam; as estratégias não mudaram, só ganharam mais tools |

### Contrato HTTP desta unidade

| Rota | Corpo | Respostas |
| --- | --- | --- |
| `POST /chat` | `{ message, strategy?, reflect?, conversationId? }` | 200 `{ answer, trace, metrics, conversationId }`; 400 corpo inválido; 422 estratégia ou conversa desconhecida; 504 timeout (60 s); 500 erro interno |

Os detalhes estão em [04-api-chat-http.md](./04-api-chat-http.md) e [06-conversa-persistente.md](./06-conversa-persistente.md).

---

**➡️ Próximo passo: [00 · Por onde começar](./00-START-HERE.md)**
