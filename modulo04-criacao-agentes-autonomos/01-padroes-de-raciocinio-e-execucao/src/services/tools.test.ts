import assert from "node:assert/strict";
import test from "node:test";
import { InMemoryOpsStore } from "../models/in-memory-store.js";
import {
  createOpsTools,
  ListAlertsInputSchema,
  ListIncidentsInputSchema,
  OpenIncidentInputSchema,
  ResolveIncidentInputSchema,
} from "./tools.js";
import { ToolBoundaryError } from "./errors.js";

test("tools validate inputs and return schema-checked JSON outputs", async () => {
  const tools = createOpsTools(new InMemoryOpsStore());
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
  const store = new InMemoryOpsStore();
  const tools = createOpsTools(store);
  await assert.rejects(
    tools[1].invoke({ title: "Invalid target", service: "missing", severity: "high" }),
    ToolBoundaryError,
  );
  assert.equal((await store.readSummary()).incidents, 0);
  await assert.rejects(tools[2].invoke({ id: "missing" }));
});
