import assert from "node:assert/strict";
import test from "node:test";
import { InMemoryOpsStore } from "./in-memory-store.js";

test("default seed has five services and a 3/3 alert status split", async () => {
  const store = new InMemoryOpsStore();
  assert.deepEqual(await store.readSummary(), {
    services: 5,
    alerts: 6,
    firing: 3,
    resolved: 3,
    incidents: 0,
  });
  assert.equal((await store.listAlerts("firing")).length, 3);
  assert.equal((await store.listAlerts("resolved")).length, 3);
  assert.equal((await store.listAlerts("all")).length, 6);
});

test("incident lifecycle is atomic, validates service, and prevents repeated resolution", async () => {
  const store = new InMemoryOpsStore();
  await assert.rejects(store.createIncident({ title: "Test", service: "unknown", severity: "high" }));
  assert.equal((await store.readSummary()).incidents, 0);
  assert.deepEqual(await store.listIncidents(), []);

  const incident = await store.createIncident({ title: "Payment errors", service: "payments", severity: "critical" });
  assert.equal(incident.status, "open");
  assert.equal((await store.listIncidents("open")).length, 1);
  assert.equal((await store.resolveIncident(incident.id)).status, "resolved");
  assert.equal((await store.listIncidents("resolved")).length, 1);
  await assert.rejects(store.resolveIncident(incident.id), /already resolved/);
  await assert.rejects(store.resolveIncident("missing"), /was not found/);
  assert.equal((await store.readSummary()).incidents, 1);
});

test("reset is idempotent and restores the seeded state", async () => {
  const store = new InMemoryOpsStore();
  await store.createIncident({ title: "Temporary", service: "api", severity: "low" });
  await store.reset();
  await store.reset();
  assert.deepEqual(await store.readSummary(), {
    services: 5,
    alerts: 6,
    firing: 3,
    resolved: 3,
    incidents: 0,
  });
});
