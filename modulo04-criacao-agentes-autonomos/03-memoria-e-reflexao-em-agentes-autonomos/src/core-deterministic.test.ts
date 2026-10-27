import assert from "node:assert/strict";
import test from "node:test";
import { formatTrace } from "./agents/trace.js";
import { InMemoryOpsStore } from "./models/in-memory-store.js";

test("seed summary and trace output remain stable over repeated resets", async () => {
  const store = new InMemoryOpsStore();
  const trace = [
    { type: "observation" as const, content: JSON.stringify(await store.listAlerts("firing")) },
    { type: "answer" as const, content: "Three alerts are firing." },
  ];
  const firstSummary = await store.readSummary();
  const firstTrace = formatTrace(trace);
  await store.reset();
  assert.deepEqual(await store.readSummary(), firstSummary);
  assert.equal(formatTrace([
    { type: "observation", content: JSON.stringify(await store.listAlerts("firing")) },
    { type: "answer", content: "Three alerts are firing." },
  ]), firstTrace);
});
