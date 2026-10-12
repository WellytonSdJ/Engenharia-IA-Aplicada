import assert from "node:assert/strict";
import test from "node:test";
import { formatTrace, parseTrace } from "./trace.js";

test("trace parser accepts all supported event types and preserves order", () => {
  const events = [
    { type: "thought", content: "Inspect the incident." },
    { type: "action", content: "List alerts.", tool: "list_alerts", args: { status: "firing" } },
    { type: "observation", content: "Two alerts are firing." },
    { type: "plan", content: "Review and answer.", steps: ["review", "answer"] },
    { type: "critique", content: "The evidence is sufficient." },
    { type: "answer", content: "The API is degraded." },
  ];
  assert.deepEqual(parseTrace(events), events);
  assert.equal(formatTrace(events).split("\n").map((line) => line[0]).join(""), "123456");
});

test("trace parser rejects missing action fields, unknown types, and plans over eight steps", () => {
  assert.throws(() => parseTrace([{ type: "action", content: "missing fields" }]));
  assert.throws(() => parseTrace([{ type: "action", content: "missing args", tool: "list_alerts" }]));
  assert.throws(() => parseTrace([{ type: "hidden", content: "unknown" }]));
  assert.throws(() => parseTrace([{ type: "plan", content: "too long", steps: Array(9).fill("step") }]));
});
