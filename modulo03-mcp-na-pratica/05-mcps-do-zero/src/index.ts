import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { server } from "./mcp.ts";

async function main() {
   // stdin/stdout, não HTTP: este processo é sempre spawnado sob demanda por um cliente MCP
   // (VS Code, MCP Inspector, ou tests/helpers.ts) — não existe servidor "de pé" esperando requisições
   const transport = new StdioServerTransport()
    await server.connect(transport)
    console.error('Encrypt MCP Server running on stdio') // stderr, nunca stdout: stdout é reservado ao JSON-RPC do protocolo
}

main().catch((error) => {
    console.error("Fatal error in main():", error);
    process.exit(1);
});