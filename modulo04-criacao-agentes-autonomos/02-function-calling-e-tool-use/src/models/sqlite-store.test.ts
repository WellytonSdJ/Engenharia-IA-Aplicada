import assert from "node:assert/strict";
import test from "node:test";
import { DatabaseSync } from "node:sqlite";
import { SqliteOpsStore } from "./sqlite-store.js";

async function seededStore(): Promise<SqliteOpsStore> {
  const store = new SqliteOpsStore(":memory:");
  await store.reset();
  return store;
}

test("reset seeds the Mercadinho scenario and is idempotent", async () => {
  const store = await seededStore();
  const expected = { services: 6, alerts: 6, firing: 3, resolved: 3, incidents: 0 };
  assert.deepEqual(await store.readSummary(), expected);
  await store.reset();
  assert.deepEqual(await store.readSummary(), expected);
});

test("createIncident validates service and persists a new open incident", async () => {
  const store = await seededStore();
  await assert.rejects(
    store.createIncident({ title: "Test", service: "payment", severity: "high" }),
    /Service payment was not found\. Known services: api, catalog, checkout, identity, notifications, payments\./,
  );
  assert.equal((await store.readSummary()).incidents, 0);

  const incident = await store.createIncident({ title: "Payment errors", service: "payments", severity: "critical" });
  assert.equal(incident.status, "open");
  assert.equal(incident.summary, null);
  assert.equal((await store.listIncidents("open")).length, 1);
});

test("listIncidents filters by open, resolved and all", async () => {
  const store = await seededStore();
  const first = await store.createIncident({ title: "Checkout slow", service: "checkout", severity: "medium" });
  await store.createIncident({ title: "Catalog down", service: "catalog", severity: "low" });
  await store.resolveIncident(first.id);

  assert.equal((await store.listIncidents("open")).length, 1);
  assert.equal((await store.listIncidents("resolved")).length, 1);
  assert.equal((await store.listIncidents("all")).length, 2);
});

test("resolveIncident resolves once, rejects a second resolution and unknown ids", async () => {
  const store = await seededStore();
  const incident = await store.createIncident({ title: "Payment errors", service: "payments", severity: "critical" });
  const resolved = await store.resolveIncident(incident.id);
  assert.equal(resolved.status, "resolved");
  assert.ok(resolved.resolvedAt instanceof Date);

  await assert.rejects(store.resolveIncident(incident.id), /already resolved/);
  await assert.rejects(store.resolveIncident("missing"), /was not found/);
  assert.equal((await store.readSummary()).incidents, 1);
});

test("findRunbook differentiates found, missing-runbook and unknown service", async () => {
  const store = await seededStore();
  const runbook = await store.findRunbook("payments");
  assert.ok(runbook);
  assert.equal(runbook?.serviceId, "svc-payments");
  assert.match(runbook?.content ?? "", /pagamento/i);

  assert.equal(await store.findRunbook("notifications"), null);

  await assert.rejects(
    store.findRunbook("unknown-service"),
    /Service unknown-service was not found\. Known services: /,
  );
});

test("CHECK constraints reject out-of-domain severity and status at the SQL layer", async () => {
  const db = new DatabaseSync(":memory:");
  db.exec(`
    CREATE TABLE incidents (
      id          TEXT PRIMARY KEY,
      title       TEXT NOT NULL,
      service_id  TEXT NOT NULL,
      severity    TEXT NOT NULL CHECK (severity IN ('low','medium','high','critical')),
      status      TEXT NOT NULL CHECK (status IN ('open','resolved')),
      created_at  TEXT NOT NULL,
      resolved_at TEXT,
      summary     TEXT
    );
  `);
  assert.throws(() => {
    db.prepare(
      "INSERT INTO incidents (id, title, service_id, severity, status, created_at, resolved_at, summary) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
    ).run("incident-x", "Bad severity", "svc-api", "sev1", "open", new Date().toISOString());
  }, /CHECK constraint failed/);
  assert.throws(() => {
    db.prepare(
      "INSERT INTO incidents (id, title, service_id, severity, status, created_at, resolved_at, summary) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
    ).run("incident-y", "Bad status", "svc-api", "high", "pending", new Date().toISOString());
  }, /CHECK constraint failed/);
});
