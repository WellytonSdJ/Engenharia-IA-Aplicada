import assert from "node:assert/strict";
import test from "node:test";
import { SqliteMemoryStore } from "../models/memory-store.js";
import { SqliteOpsStore } from "../models/sqlite-store.js";
import { FakeEmbedder } from "./fake-embedder.js";
import {
  createForgetPreferenceTool,
  createOpsTools,
  ForgetPreferenceInputSchema,
  ListAlertsInputSchema,
  ListIncidentsInputSchema,
  OpenIncidentInputSchema,
  ResolveIncidentInputSchema,
} from "./tools.js";
import { ToolBoundaryError } from "./errors.js";

async function seededTools() {
  const store = new SqliteOpsStore(":memory:");
  await store.reset();
  return { store, tools: createOpsTools(store) };
}

test("tools validate inputs and return schema-checked JSON outputs", async () => {
  const { tools } = await seededTools();
  const alerts = JSON.parse(await tools[0].invoke({ status: "firing" }));
  assert.equal(alerts.length, 3);
  assert.ok(alerts.every((alert: { status: string }) => alert.status === "firing"));

  const opened = JSON.parse(await tools[1].invoke({ title: "Queue delay", service: "notifications", severity: "medium" }));
  assert.equal(opened.status, "open");
  assert.equal(OpenIncidentInputSchema.safeParse({ title: "", service: "api", severity: "low" }).success, false);
  assert.equal(OpenIncidentInputSchema.safeParse({ title: "bad severity", service: "api", severity: "urgent" }).success, false);
  assert.equal(ListAlertsInputSchema.safeParse({ status: "unknown" }).success, false);
  assert.equal(ListIncidentsInputSchema.safeParse({ status: "unknown" }).success, false);
  assert.equal(ResolveIncidentInputSchema.safeParse({ id: "" }).success, false);
});

test("domain errors are surfaced from the tool boundary without partial changes", async () => {
  const { store, tools } = await seededTools();
  await assert.rejects(
    tools[1].invoke({ title: "Invalid target", service: "missing", severity: "high" }),
    ToolBoundaryError,
  );
  assert.equal((await store.readSummary()).incidents, 0);
  await assert.rejects(tools[2].invoke({ id: "missing" }));
});

test("consultar_runbook returns runbook, found:false, or rejects for unknown service", async () => {
  const { tools } = await seededTools();
  const found = JSON.parse(await tools[4].invoke({ service: "payments" }));
  assert.equal(found.service, "payments");
  assert.ok(found.title);
  assert.ok(found.content);

  const notFound = JSON.parse(await tools[4].invoke({ service: "notifications" }));
  assert.deepEqual(notFound, { service: "notifications", found: false });

  await assert.rejects(tools[4].invoke({ service: "unknown-service" }), ToolBoundaryError);
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

test("check_provider_status returns a compact line on success, defaulting to github", async () => {
  let calls = 0;
  const fetchFn = (async (url: string | URL) => {
    calls++;
    assert.ok(String(url).includes("githubstatus.com"));
    return jsonResponse({ status: { indicator: "none", description: "All Systems Operational" } });
  }) as typeof fetch;
  const { store } = await seededTools();
  const tools = createOpsTools(store, { fetchFn });

  const result = await tools[5].invoke({});
  assert.equal(result, "none — All Systems Operational");
  assert.equal(calls, 1);
});

test("check_provider_status queries cloudflare when requested", async () => {
  const fetchFn = (async (url: string | URL) => {
    assert.ok(String(url).includes("cloudflarestatus.com"));
    return jsonResponse({ status: { indicator: "minor", description: "Partial outage" } });
  }) as typeof fetch;
  const { store } = await seededTools();
  const tools = createOpsTools(store, { fetchFn });

  const result = await tools[5].invoke({ provider: "cloudflare" });
  assert.equal(result, "minor — Partial outage");
});

test("check_provider_status retries once on network failure then succeeds", async () => {
  let calls = 0;
  const fetchFn = (async () => {
    calls++;
    if (calls === 1) {
      throw new Error("network down");
    }
    return jsonResponse({ status: { indicator: "none", description: "All Systems Operational" } });
  }) as typeof fetch;
  const { store } = await seededTools();
  const tools = createOpsTools(store, { fetchFn });

  const result = await tools[5].invoke({});
  assert.equal(result, "none — All Systems Operational");
  assert.equal(calls, 2);
});

test("check_provider_status returns a readable error after persistent failure, without throwing", async () => {
  const fetchFn = (async () => {
    throw new Error("timeout");
  }) as typeof fetch;
  const { store } = await seededTools();
  const tools = createOpsTools(store, { fetchFn });

  const result = await tools[5].invoke({});
  assert.match(result, /Não foi possível consultar o status de github/);
});

test("check_provider_status does not retry on 4xx responses", async () => {
  let calls = 0;
  const fetchFn = (async () => {
    calls++;
    return jsonResponse({}, 404);
  }) as typeof fetch;
  const { store } = await seededTools();
  const tools = createOpsTools(store, { fetchFn });

  const result = await tools[5].invoke({});
  assert.match(result, /HTTP 404/);
  assert.equal(calls, 1);
});

test("check_provider_status returns a readable error for an invalid payload", async () => {
  const fetchFn = (async () => jsonResponse({ unexpected: true })) as typeof fetch;
  const { store } = await seededTools();
  const tools = createOpsTools(store, { fetchFn });

  const result = await tools[5].invoke({});
  assert.match(result, /formato inesperado/);
});

test("forget_preference forgets only the most similar memory of the bound user", async () => {
  const embedder = new FakeEmbedder()
    .setAxis("O plantonista prioriza o checkout", 0)
    .setAxis("Prefere resumos curtos", 1)
    .setAxis("priorizar checkout", 0);
  const memoryStore = new SqliteMemoryStore(":memory:", embedder);
  await memoryStore.remember("ana", "O plantonista prioriza o checkout");
  await memoryStore.remember("ana", "Prefere resumos curtos");
  await memoryStore.remember("bia", "O plantonista prioriza o checkout");
  const forget = createForgetPreferenceTool(memoryStore, "ana");

  const output = await forget.invoke({ query: "priorizar checkout" });

  assert.equal(output, "Preferência esquecida: O plantonista prioriza o checkout");
  assert.deepEqual(await memoryStore.recall("ana", "priorizar checkout"), []);
  assert.equal((await memoryStore.recall("ana", "Prefere resumos curtos")).length, 1);
  assert.equal((await memoryStore.recall("bia", "priorizar checkout")).length, 1);
});

test("forget_preference without a similar memory answers safely and changes nothing", async () => {
  const embedder = new FakeEmbedder().setAxis("Prefere resumos curtos", 1).setAxis("esqueça o checkout", 2);
  const memoryStore = new SqliteMemoryStore(":memory:", embedder);
  await memoryStore.remember("ana", "Prefere resumos curtos");
  const forget = createForgetPreferenceTool(memoryStore, "ana");

  const output = await forget.invoke({ query: "esqueça o checkout" });

  assert.match(output, /Nenhuma preferência parecida/);
  assert.equal((await memoryStore.recall("ana", "Prefere resumos curtos")).length, 1);
});

test("forget_preference schema has no userId field and rejects an empty query", () => {
  assert.deepEqual(Object.keys(ForgetPreferenceInputSchema.shape), ["query"]);
  assert.equal(ForgetPreferenceInputSchema.safeParse({ query: "  " }).success, false);
});
