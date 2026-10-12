import assert from "node:assert/strict";
import test from "node:test";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { SimpleChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseMessage } from "@langchain/core/messages";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { InMemoryOpsStore } from "../models/in-memory-store.js";
import { IterationLimitError } from "../services/errors.js";
import { createReactStrategy, traceFromMessages } from "./react.js";

class NoopChatModel extends SimpleChatModel {
  _llmType(): string {
    return "noop-test-model";
  }

  async _call(
    _messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<string> {
    throw new Error("The injected test agent must not invoke the model.");
  }
}

test("ReAct trace captures thought, tool action, observation, and final answer", () => {
  const trace = traceFromMessages([
    new HumanMessage("List active alerts."),
    new AIMessage({
      content: "I should inspect the alert store.",
      tool_calls: [{ name: "list_alerts", args: { status: "firing" }, id: "call-1", type: "tool_call" }],
    }),
    new ToolMessage({ content: '[{"status":"firing"}]', tool_call_id: "call-1", name: "list_alerts" }),
    new AIMessage({ content: "One alert is firing." }),
  ]);
  assert.deepEqual(trace.map(({ type }) => type), ["thought", "action", "observation", "answer"]);
  assert.deepEqual(trace[1], {
    type: "action",
    content: "Calling list_alerts",
    tool: "list_alerts",
    args: { status: "firing" },
  });
});

test("ReAct strategy consumes a controlled agent result and accounts for model turns", async () => {
  const model = new NoopChatModel({});
  let configuredRecursionLimit = 0;
  const strategy = createReactStrategy((_model, tools, recursionLimit) => {
    assert.deepEqual(tools.map((item) => item.name), ["list_alerts", "open_incident", "resolve_incident", "list_incidents"]);
    configuredRecursionLimit = recursionLimit;
    return {
      async invoke() {
        return {
          messages: [
            new HumanMessage("List active alerts."),
            new AIMessage({
              content: "",
              tool_calls: [{ name: "list_alerts", args: { status: "firing" }, id: "call-1", type: "tool_call" }],
            }),
            new ToolMessage({ content: '[{"status":"firing"}]', tool_call_id: "call-1", name: "list_alerts" }),
            new AIMessage({ content: "One alert is firing." }),
          ],
        };
      },
    };
  });

  const result = await strategy.run("List active alerts.", {
    model,
    store: new InMemoryOpsStore(),
    maxIterations: 1,
  });
  assert.equal(configuredRecursionLimit, 3);
  assert.equal(result.metrics.llmCalls, 2);
  assert.equal(result.metrics.iterations, 1);
  assert.deepEqual(result.trace.map((event) => event.type), ["action", "observation", "answer"]);
});

test("ReAct strategy rejects results that exceed the requested iteration cap", async () => {
  const strategy = createReactStrategy(() => ({
    async invoke() {
      return {
        messages: [
          new AIMessage({ content: "Intermediate response." }),
          new AIMessage({ content: "Another intermediate response." }),
          new AIMessage({ content: "Final response." }),
        ],
      };
    },
  }));
  await assert.rejects(
    strategy.run("Check alerts.", { model: new NoopChatModel({}), maxIterations: 1 }),
    IterationLimitError,
  );
});
