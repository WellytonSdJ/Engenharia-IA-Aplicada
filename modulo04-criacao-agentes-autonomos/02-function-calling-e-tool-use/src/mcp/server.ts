import { pathToFileURL } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createOpsToolDefinitions } from "../services/tools.js";
import { type OpsStore } from "../models/in-memory-store.js";
import { SqliteOpsStore } from "../models/sqlite-store.js";

const MCP_TOOL_NAMES = ["list_alerts", "open_incident", "resolve_incident"] as const;

export function createMcpServer(store: OpsStore): McpServer {
  const server = new McpServer({ name: "opspilot", version: "1.0.0" });
  const definitions = createOpsToolDefinitions(store).filter((definition) =>
    (MCP_TOOL_NAMES as readonly string[]).includes(definition.name),
  );
  for (const definition of definitions) {
    server.registerTool(
      definition.name,
      {
        description: definition.description,
        inputSchema: definition.inputSchema,
      },
      async (input) => ({ content: [{ type: "text", text: await definition.run(input) }] }),
    );
  }
  return server;
}

export async function startStdioServer(): Promise<void> {
  const store = new SqliteOpsStore();
  const server = createMcpServer(store);
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

// Ponto de entrada da CLI. Rode via `npm run mcp`: o script é
// `node --env-file-if-exists=.env --import tsx src/mcp/server.ts`, que carrega o .env
// (ex.: OPSPILOT_DB) antes deste arquivo executar.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  startStdioServer().catch((error: unknown) => {
    console.error("OpsPilot MCP server failed:", error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
