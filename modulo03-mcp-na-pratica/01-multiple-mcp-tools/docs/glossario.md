# Glossário

Referência rápida. Para profundidade, vá ao documento específico de cada conceito.

Termos de MCP (MCP, MCP Server, MCP Client, STDIO transport, Tool, Tool calling, Agente, Lazy initialization, Escopo MCP) já cobertos no [glossário de `05-safeguard-prompt-injection`](../../../modulo02-integracao-apis-llms/05-safeguard-prompt-injection/docs/glossario.md) não são repetidos aqui. Termos de LangGraph/LangChain/Zod (StateGraph, State, Node, Edge, Structured Output, `providerStrategy`, `createAgent`, Zod, `z.infer`) já cobertos ao longo do módulo 02 (ver [glossário de `06-rag-neo4j-students`](../../../modulo02-integracao-apis-llms/06-rag-neo4j-students/docs/glossario.md) e [`structured-output.md` de `03-medical-appointment`](../../../modulo02-integracao-apis-llms/03-medical-appointment/docs/structured-output.md)) também não são repetidos.

---

## MCP: múltiplos servidores

| Termo | Definição |
| --- | --- |
| **`MultiServerMCPClient.mcpServers`** | Objeto de configuração onde cada chave é o nome de um servidor MCP e o valor descreve seu transporte. Permite conectar N servidores no mesmo cliente. |
| **`mongodb-mcp-server`** | Servidor MCP oficial do MongoDB. Expõe tools que traduzem para operações do driver (inserir, consultar, agregar, deletar documentos). |
| **`MDB_MCP_CONNECTION_STRING`** | Variável de ambiente passada ao subprocesso do `mongodb-mcp-server` com a connection string do MongoDB. |
| **Tool nativa (não-MCP)** | Tool criada com `tool()` de `@langchain/core/tools`, executada in-process — sem subprocesso, sem transporte STDIO. Neste projeto: `csv_to_json`. |
| **`onMessage`** | Callback do `MultiServerMCPClient` que recebe logs de cada servidor conectado, identificado por `source.server`. |

---

## Agente autônomo e tool calling loop

| Termo | Definição |
| --- | --- |
| **Tool calling loop** | Ciclo interno do `createAgent`: o modelo recebe o histórico, decide se chama uma tool, a tool executa, o resultado volta para o histórico, e o modelo decide de novo — até responder em texto final sem novas tool calls. |
| **Orquestração explícita** | Padrão dos projetos anteriores (03, 06 do módulo 02): cada etapa é um nó de grafo escrito em código, com transições fixas ou condicionais definidas no `StateGraph`. |
| **Orquestração autônoma (delegada ao modelo)** | Padrão deste projeto: uma única chamada a `createAgent` com tools reais, e o modelo decide a sequência de execução guiado só por instruções em linguagem natural no system prompt. |
| **Prompt com passos numerados** | Técnica de instruir o modelo com uma sequência explícita ("Step 0... Step 5") dentro do system prompt, na tentativa de aproximar o comportamento do agente de um fluxo determinístico — sem, no entanto, garantir isso estruturalmente. |
| **`agentConfig` ternário** | Padrão no `OpenRouterService`: o mesmo método (`generateStructured`) monta `{ responseFormat, tools: [] }` quando recebe um schema Zod, ou `{ tools }` quando não recebe — dois comportamentos de `createAgent` a partir do mesmo código. |

---

## Observabilidade

| Termo | Definição |
| --- | --- |
| **Callback (LangChain)** | Função-gancho registrada em `agent.invoke(..., { callbacks: [...] })` que dispara em pontos específicos do ciclo de execução (início de chamada ao LLM, fim da chamada, início/fim de tool). |
| **`handleChatModelStart`** | Callback disparado antes de cada chamada ao modelo — mostra a última mensagem do contexto enviado. |
| **`handleLLMEnd`** | Callback disparado após a resposta do modelo — permite inspecionar se ele decidiu chamar alguma tool (`tool_calls`). |
| **`handleToolStart`** / **`handleToolEnd`** | Callbacks disparados antes e depois da execução de uma tool — mostram nome, input e output. |
| **`ChatGeneration`** | Tipo do LangChain usado para tipar a saída bruta do modelo dentro de `handleLLMEnd` (`output.generations`). |

---

## Domínio (pipeline de dados de vendas)

| Termo | Definição |
| --- | --- |
| **`IntentSchema`** | Schema Zod deste projeto: `intent` (string livre, não enum), `fileContent`, `fileName`, `fileType` (`csv`/`json`/`unknown`). Diferente do `IntentSchema` de `03-medical-appointment`, que usava um enum fixo de ações. |
| **`sales.csv` / `sales-complete.csv`** | Arquivos de exemplo com dados de vendas (produto, preço, data) usados como entrada do pipeline. |
| **`reports/`** | Diretório onde o próprio agente escreve o relatório final via `write_file`, no último passo do prompt (Step 5). |
| **`mongo-express`** | UI web (porta 8081) para inspecionar visualmente as coleções inseridas no MongoDB pelo agente. Não faz parte do fluxo do agente — é só uma ferramenta de inspeção manual. |
