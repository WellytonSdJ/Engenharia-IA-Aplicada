import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { server } from "./mcp/server.ts";

// Entry point idêntico ao do projeto anterior (05-mcps-do-zero).
// O servidor não abre porta HTTP — conecta-se ao processo cliente via stdin/stdout (STDIO transport).
// Quem sobe este processo: VS Code (.vscode/mcp.json), MCP Inspector, ou os testes (tests/helpers.ts).
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    // console.error, não console.log: stdout está reservado para o protocolo JSON-RPC.
    console.error("Customers MCP Server running on stdio");
}

main().catch((error) => {
    console.error("Fatal error in main():", error);
    process.exit(1);
});
