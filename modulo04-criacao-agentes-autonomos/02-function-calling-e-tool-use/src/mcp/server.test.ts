import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { SqliteOpsStore } from "../models/sqlite-store.js";
import { createOpsTools } from "../services/tools.js";
import { createMcpServer } from "./server.js";

async function connectedClient(store: SqliteOpsStore) {
  const server = createMcpServer(store);
  const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "opspilot-test-client", version: "0.0.0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  return client;
}

async function seededStore() {
  const store = new SqliteOpsStore(":memory:");
  await store.reset();
  return store;
}

function textOf(result: Awaited<ReturnType<Client["callTool"]>>): string {
  const content = result.content as Array<{ type: string; text: string }>;
  assert.equal(content.length, 1);
  assert.equal(content[0].type, "text");
  return content[0].text;
}

test("lists exactly list_alerts, open_incident, resolve_incident, and names the server opspilot", async () => {
  const store = await seededStore();
  const client = await connectedClient(store);

  const { tools } = await client.listTools();
  assert.deepEqual(tools.map((tool) => tool.name).sort(), ["list_alerts", "open_incident", "resolve_incident"]);

  const names = tools.map((tool) => tool.name);
  for (const outOfScope of ["list_incidents", "consultar_runbook", "check_provider_status"]) {
    assert.ok(!names.includes(outOfScope), `${outOfScope} should not be registered on the MCP server`);
  }

  assert.deepEqual(client.getServerVersion(), { name: "opspilot", version: "1.0.0" });
});

test("list_alerts via MCP mirrors the agent tool over the same store", async () => {
  const store = await seededStore();
  const client = await connectedClient(store);
  const agentTools = createOpsTools(store);

  const mcpResult = await client.callTool({ name: "list_alerts", arguments: { status: "firing" } });
  const agentResult = await agentTools[0].invoke({ status: "firing" });

  assert.equal(textOf(mcpResult), agentResult);
});

test("open_incident via MCP creates an incident in the shared store", async () => {
  const store = await seededStore();
  const client = await connectedClient(store);

  const result = await client.callTool({
    name: "open_incident",
    arguments: { title: "Queue delay", service: "notifications", severity: "medium" },
  });
  const opened = JSON.parse(textOf(result));
  assert.equal(opened.status, "open");
  assert.equal(opened.severity, "medium");

  const incidents = await store.listIncidents("all");
  assert.equal(incidents.length, 1);
  assert.equal(incidents[0].id, opened.id);
  assert.equal(incidents[0].status, "open");
  assert.equal(incidents[0].service.name, "notifications");
});

test("open_incident via MCP rejects an unknown service without altering the store", async () => {
  const store = await seededStore();
  const client = await connectedClient(store);

  const result = await client.callTool({
    name: "open_incident",
    arguments: { title: "Invalid target", service: "missing", severity: "high" },
  });

  assert.equal(result.isError, true);
  assert.equal((await store.readSummary()).incidents, 0);
});

test("resolve_incident via MCP transitions a known incident to resolved in the shared store", async () => {
  const store = await seededStore();
  const client = await connectedClient(store);
  const created = await store.createIncident({ title: "Checkout down", service: "checkout", severity: "critical" });

  const result = await client.callTool({ name: "resolve_incident", arguments: { id: created.id } });
  const resolved = JSON.parse(textOf(result));
  assert.equal(resolved.status, "resolved");

  const [incident] = await store.listIncidents("all");
  assert.equal(incident.status, "resolved");
});

test("resolve_incident via MCP returns a readable error for an unknown id without throwing or crashing", async () => {
  const store = await seededStore();
  const client = await connectedClient(store);

  const result = await client.callTool({ name: "resolve_incident", arguments: { id: "missing" } });

  assert.equal(result.isError, true);
  assert.match(textOf(result), /missing/);
});

test("src/mcp/server.ts never writes to stdout outside the protocol transport (no console.log)", () => {
  const serverPath = fileURLToPath(new URL("./server.ts", import.meta.url));
  const source = readFileSync(serverPath, "utf8");
  assert.ok(!source.includes("console.log"), "src/mcp/server.ts must not contain console.log");
});
