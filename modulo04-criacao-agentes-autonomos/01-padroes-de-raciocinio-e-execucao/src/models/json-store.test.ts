import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { ValidationError } from "../services/errors.js";
import { createOpsTools } from "../services/tools.js";
import { JsonOpsStore } from "./json-store.js";

async function createTemporaryStore(t: test.TestContext): Promise<JsonOpsStore> {
  const directory = await mkdtemp(join(tmpdir(), "opspilot-json-store-"));
  t.after(async () => rm(directory, { recursive: true, force: true }));
  return new JsonOpsStore(join(directory, "ops-store.json"));
}

test("JSON store reset writes the seed and mutations persist across store instances", async (t) => {
  const store = await createTemporaryStore(t);
  await store.reset();
  const tools = createOpsTools(store);
  assert.deepEqual(await store.readSummary(), {
    services: 6,
    alerts: 6,
    firing: 3,
    resolved: 3,
    incidents: 0,
  });

  const [firstJson, secondJson] = await Promise.all([
    tools[1].invoke({ title: "Payment timeout", service: "payments", severity: "critical" }),
    tools[1].invoke({ title: "API errors", service: "api", severity: "high" }),
  ]);
  const first = JSON.parse(firstJson) as { id: string; status: string };
  const second = JSON.parse(secondJson) as { id: string; status: string };
  const reopened = new JsonOpsStore(store.filePath);
  assert.equal((await reopened.readSummary()).incidents, 2);
  const openIncidents = await createOpsTools(reopened)[3].invoke({ status: "open" });
  assert.equal(JSON.parse(openIncidents).length, 2);
  assert.equal((await reopened.resolveIncident(first.id)).status, "resolved");
  assert.equal((await reopened.resolveIncident(second.id)).status, "resolved");
  assert.equal((await store.readSummary()).incidents, 2);
  assert.equal((await reopened.listIncidents("resolved")).length, 2);
});

test("JSON store reset is repeatable and malformed JSON reports a validation error", async (t) => {
  const store = await createTemporaryStore(t);
  await store.reset();
  const first = await readFile(store.filePath, "utf8");
  await store.createIncident({ title: "Temporary", service: "api", severity: "low" });
  await store.reset();
  assert.equal(await readFile(store.filePath, "utf8"), first);
  assert.equal((await store.readSummary()).incidents, 0);

  await writeFile(store.filePath, "{ invalid", "utf8");
  await assert.rejects(store.readSummary(), ValidationError);
});
