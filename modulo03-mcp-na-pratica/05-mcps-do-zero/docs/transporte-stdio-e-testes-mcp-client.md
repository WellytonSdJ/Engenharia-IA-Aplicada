# Transporte STDIO e testes via MCP client

## O servidor como subprocesso

`src/index.ts` não abre uma porta HTTP — ele conecta o `McpServer` a um `StdioServerTransport`, que fala o protocolo MCP (JSON-RPC) através de stdin/stdout:

```typescript
// src/index.ts
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { server } from "./mcp.ts";

async function main() {
   const transport = new StdioServerTransport()
    await server.connect(transport)
    console.error('Encrypt MCP Server running on stdio')
}
```

Isso só faz sentido porque quem sobe esse processo é sempre outro processo: o VS Code, o MCP Inspector, ou (nos testes) o próprio `Client` do SDK. Não existe um "servidor rodando sozinho" esperando requisições — ele é um subprocesso descartável, iniciado sob demanda por quem precisa das tools.

**Por que `console.error` e não `console.log`**: stdout está reservado para as mensagens JSON-RPC do protocolo. Qualquer `console.log` ali corromperia a comunicação — por isso o log de diagnóstico vai para stderr.

```
processo cliente (VS Code / Inspector / tests/helpers.ts)
      │
      │ spawna: node --experimental-strip-types src/index.ts
      │ stdin/stdout (JSON-RPC)
      ▼
processo do servidor (este projeto)
      │
      ▼
McpServer despacha para a tool/resource/prompt certa
```

Esse mesmo padrão aparece em `.vscode/mcp.json` (o VS Code sobe o processo com `command: "node"`, `args: [...]`) e em `tests/helpers.ts` (o `Client` de teste faz o mesmo, programaticamente).

---

## Testes que conversam com o protocolo, não com as funções

A suíte de testes (`tests/mcp.test.ts`) não importa `encrypt`/`decrypt` de `service.ts` para testar unitariamente. Em vez disso, ela abre um `Client` MCP real, conectado ao servidor real via STDIO, e chama tools/resources/prompts exatamente como um agente faria:

```typescript
// tests/helpers.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

export async function createTestClient () {
    const transport = new StdioClientTransport({
        command: 'node',
        args: ['--experimental-strip-types', 'src/index.ts']   // sobe o servidor como subprocesso real
    })

    const client = new Client(
        { name: 'test-client', version: '1.0.1' },
        { capabilities: {} }                                    // cliente não anuncia capacidades extras — só consome
    )

    await client.connect(transport)
    return client
}
```

Cada teste chama o servidor pelo mesmo caminho que um cliente de produção usaria:

```typescript
// tests/mcp.test.ts
const result = await client.callTool({
    name: 'encrypt_message',
    arguments: { message, encryptionKey }
})
```

O motivo de testar dessa forma, mais cara que um teste unitário direto: valida o contrato inteiro de ponta a ponta — nomes registrados batendo com os esperados, schemas de entrada aceitando os argumentos, `structuredContent` saindo no formato certo, `listResources`/`getPrompt` respondendo pelo protocolo real. Um teste unitário de `encrypt()`/`decrypt()` não pegaria, por exemplo, um erro de nome de tool digitado errado em `registerTool`.

**`before`/`after` sobem e derrubam o subprocesso uma única vez por suíte** (`client.connect` no `before`, `client.close` no `after`), não a cada `it` — subir um processo Node novo por teste seria desnecessariamente lento para uma suíte pequena como esta.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Servidor conectado ao transporte STDIO | [src/index.ts](../src/index.ts) | `new StdioServerTransport()` + `server.connect(transport)` |
| Cliente de teste conectado via STDIO | [tests/helpers.ts](../tests/helpers.ts) | `StdioClientTransport({ command: 'node', args: [...] })` sobe o mesmo `src/index.ts` usado em produção |
| Chamadas ao protocolo nos testes | [tests/mcp.test.ts](../tests/mcp.test.ts) | `client.callTool`, `client.listResources`, `client.getPrompt` |
| Configuração equivalente para o VS Code | [.vscode/mcp.json](../.vscode/mcp.json) | Mesmo `command`/`args` usado pelo cliente de teste, mas para o Copilot Chat |
