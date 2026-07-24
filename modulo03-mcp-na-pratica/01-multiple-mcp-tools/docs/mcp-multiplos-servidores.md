# MCP: Múltiplos Servidores + Tool Customizada

> Este projeto **não repete** os fundamentos de MCP — isso já foi coberto em [`05-safeguard-prompt-injection/docs/mcp.md`](../../../modulo02-integracao-apis-llms/05-safeguard-prompt-injection/docs/mcp.md) (o que é MCP, transporte STDIO, `MultiServerMCPClient`, tool calling, lazy initialization). Leia aquele documento primeiro se esses termos são novos para você. Aqui o foco é só no que muda: **dois servidores MCP ao mesmo tempo** e **uma tool nativa do LangChain misturada com tools vindas de MCP**.

## O que já era conhecido vs. o que é novo

| | 05-safeguard-prompt-injection | 01-multiple-mcp-tools (aqui) |
| --- | --- | --- |
| Servidores MCP conectados | 1 (`filesystem`) | 2 (`filesystem` + `MongoDB`) |
| Origem das tools do agente | 100% MCP | MCP (2 servidores) + 1 tool nativa LangChain |
| Papel do agente | Responder perguntas usando 1 ferramenta | Executar um pipeline de dados de ponta a ponta usando 5+ ferramentas |

## Conectando dois servidores no mesmo cliente

`MultiServerMCPClient` aceita um objeto `mcpServers` onde cada chave é o nome de um servidor. O projeto monta esse objeto combinando dois helpers via spread:

```typescript
// src/services/mcpService.ts
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { getMongoDBTool } from "../tools/mongodbTool.ts";
import { getCSVTOJSONTool } from "../tools/csvToJSONTool.ts";
import { getFSTool } from "../tools/fsTool.ts";

export const getMCPTools = async () => {
  const client = new MultiServerMCPClient({
    mcpServers: {
      ...getMongoDBTool(),
      ...getFSTool(),
    },
    onMessage: (log, source) => {
      console.log(`[${source.server}] ${log.data}`)
    }
  })

  const mcpTools = await client.getTools()

  return [
    ...mcpTools,
    getCSVTOJSONTool()
  ];
};
```

Cada helper (`getFSTool`, `getMongoDBTool`) retorna um objeto com uma única chave — o nome do servidor — e sua configuração de transporte:

```typescript
// src/tools/fsTool.ts
export const getFSTool = () => {
  return {
    "filesystem": {
      transport: 'stdio' as const,
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", `${process.cwd()}`],
    }
  }
}
```

```typescript
// src/tools/mongodbTool.ts
// https://github.com/mongodb-js/mongodb-mcp-server
export const getMongoDBTool = () => {
  return {
    "MongoDB": {
      transport: 'stdio' as const,
      "command": "npx",
      "args": ["-y", "mongodb-mcp-server@latest"],
      "env": {
        "MDB_MCP_CONNECTION_STRING": "mongodb://localhost:27017/dataprocessing"
      }
    }
  }
}
```

O `...spread` dentro de `mcpServers` é só açúcar sintático para "juntar dois objetos de configuração em um só". Na prática, o `MultiServerMCPClient` sobe **dois subprocessos** (`npx @modelcontextprotocol/server-filesystem` e `npx mongodb-mcp-server`), cada um falando STDIO com o cliente, e agrega as tools de ambos num único array via `client.getTools()`.

## O servidor MCP do MongoDB

`mongodb-mcp-server` é um pacote oficial do time do MongoDB. Diferente do `server-filesystem` (que expõe operações de arquivo), ele expõe tools que traduzem para operações do driver: inserir documentos, rodar queries/agregações, listar coleções, etc. A conexão é configurada via variável de ambiente passada ao subprocesso:

```typescript
"env": {
  "MDB_MCP_CONNECTION_STRING": "mongodb://localhost:27017/dataprocessing"
}
```

Esse Mongo local é o mesmo subido pelo `docker-compose.yaml` do projeto:

```yaml
# docker-compose.yaml
services:
  mongodb:
    image: mongo:8
    ports:
      - "27017:27017"
  mongo-express:
    image: mongo-express:1.0.2
    ports:
      - "8081:8081"
```

`mongo-express` é só uma UI web (porta 8081) para inspecionar visualmente o que o agente inseriu no banco — útil durante o desenvolvimento, não faz parte do fluxo do agente.

## Misturando tool nativa do LangChain com tools de MCP

A parte mais importante deste documento: o array final de tools **não vem só de MCP**. O `csv_to_json` é uma tool comum do LangChain (`tool()` de `@langchain/core/tools`), executada dentro do próprio processo Node — sem subprocesso, sem STDIO:

```typescript
// src/tools/csvToJSONTool.ts
import { tool } from "@langchain/core/tools";
import csvtojson from 'csvtojson'
import { z } from 'zod/v3'

export function getCSVTOJSONTool() {
  return tool(
    async ({ csvText }) => {
      const result = await csvtojson().fromString(csvText)
      return JSON.stringify(result)
    },
    {
      name: 'csv_to_json',
      description: 'Convert CSV to JSON formart',
      schema: z.object({
        csvText: z.string().describe('CSV data to be converted to JSON formart')
      })
    }
  )
}
```

Do ponto de vista do agente, não existe diferença entre uma tool MCP e uma tool nativa — as duas chegam ao `createAgent` como itens do mesmo array `tools`, cada uma com `name`, `description` e `schema`. O modelo escolhe qual chamar olhando a descrição, não a origem:

```typescript
// src/services/openRouterService.ts
const agentConfig = schema
  ? { responseFormat: providerStrategy(schema), tools: [] }
  : { tools: await this.#getTools() };   // aqui entram MCP + csv_to_json juntas

const agent = createAgent({ ...agentConfig, model: this.llmClient });
```

**Por que isso importa:** nem toda capacidade precisa ser um servidor MCP. Rodar um subprocesso via STDIO tem custo (start do processo, serialização JSON-RPC) que só compensa quando a ferramenta precisa de algo fora do processo Node (sistema de arquivos, outro banco, uma API de terceiros). Conversão de CSV para JSON é pura computação local — não ganha nada rodando em outro processo, então vira uma tool LangChain comum. A decisão de "isso deveria ser um MCP server ou uma function local?" é uma escolha de arquitetura, não uma obrigação.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Composição dos servidores MCP | [src/services/mcpService.ts](../src/services/mcpService.ts) | `mcpServers: { ...getMongoDBTool(), ...getFSTool() }` |
| Servidor filesystem | [src/tools/fsTool.ts](../src/tools/fsTool.ts) | Igual ao já visto em `05-safeguard-prompt-injection`, escopo `process.cwd()` |
| Servidor MongoDB | [src/tools/mongodbTool.ts](../src/tools/mongodbTool.ts) | `MDB_MCP_CONNECTION_STRING`, pacote `mongodb-mcp-server` |
| Tool nativa (não-MCP) | [src/tools/csvToJSONTool.ts](../src/tools/csvToJSONTool.ts) | `tool()` do `@langchain/core/tools`, roda in-process |
| Infra do MongoDB | [docker-compose.yaml](../docker-compose.yaml) | `mongodb` (27017) + `mongo-express` (8081, UI) |
| Onde as tools chegam ao agente | [src/services/openRouterService.ts](../src/services/openRouterService.ts) | `tools: await this.#getTools()` passado ao `createAgent` |
