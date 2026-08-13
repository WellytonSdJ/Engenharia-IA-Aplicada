import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

export async function createTestClient () {
    // sobe o MESMO entry point usado em produção (.vscode/mcp.json aponta pro mesmo comando) —
    // os testes validam o servidor real via protocolo, não as funções internas de service.ts
    const transport = new StdioClientTransport({
        command: 'node',
        args: [
            '--experimental-strip-types',
            'src/index.ts'
        ]
    })

    const client = new Client({
        name: 'test-client',
        version: '1.0.1'
    }, {
        capabilities: {} // cliente só consome tools/resources/prompts, não anuncia capacidades extras
    })

    await client.connect(transport)
    return client
}