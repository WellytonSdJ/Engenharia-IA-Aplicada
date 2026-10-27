import assert from "node:assert/strict";
import test from "node:test";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { BaseMessage } from "@langchain/core/messages";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import { SimpleChatModel } from "@langchain/core/language_models/chat_models";
import { RunnableLambda, type Runnable } from "@langchain/core/runnables";
import { z } from "zod";
import type { ReasoningOptions, ReasoningStrategy } from "./strategy.js";
import type { ReasoningResult } from "./trace.js";
import { withReflection } from "./reflection.js";

class ControlledCriticModel extends SimpleChatModel {
  readonly inputs: BaseLanguageModelInput[] = [];

  constructor(private readonly outputs: Array<unknown | Error>) {
    super({});
  }

  _llmType(): string {
    return "controlled-critic-model";
  }

  async _call(
    _messages: BaseMessage[],
    _options: this["ParsedCallOptions"],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<string> {
    throw new Error("The structured-output test double should not make text model calls.");
  }

  withStructuredOutput<RunOutput extends Record<string, unknown>>(
    outputSchema: z.ZodType<RunOutput> | Record<string, unknown>,
  ): Runnable<BaseLanguageModelInput, RunOutput> {
    return RunnableLambda.from(async (input: BaseLanguageModelInput) => {
      this.inputs.push(input);
      const next = this.outputs.shift();
      if (next instanceof Error) {
        throw next;
      }
      if (next === undefined) {
        throw new Error("No controlled critic response remains.");
      }
      if (outputSchema instanceof z.ZodType) {
        return outputSchema.parse(next) as RunOutput;
      }
      return next as RunOutput;
    });
  }
}

function controlledStrategy(
  answers: string[],
  seenInputs: string[],
  seenModels: Array<ReasoningOptions["model"]>,
): ReasoningStrategy {
  return {
    name: "controlled",
    async run(input, options) {
      seenInputs.push(input);
      seenModels.push(options?.model);
      const answer = answers.shift();
      if (answer === undefined) {
        throw new Error("No controlled strategy answer remains.");
      }
      const result: ReasoningResult = {
        answer,
        trace: [
          { type: "observation", content: `Observed evidence for ${answer}.` },
          { type: "answer", content: answer },
        ],
        metrics: { llmCalls: 2, latencyMs: 5, iterations: 1 },
      };
      return result;
    },
  };
}

function rejection(feedback: string): { aproved: false; feedback: string } {
  return { aproved: false, feedback };
}

test("reflection returns an initially approved answer without regenerating", async () => {
  const model = new ControlledCriticModel([{ aproved: true, feedback: "Supported by the observations." }]);
  const seenInputs: string[] = [];
  const seenModels: Array<ReasoningOptions["model"]> = [];
  const strategy = withReflection(controlledStrategy(["Approved answer."], seenInputs, seenModels));

  const result = await strategy.run("Original request.", { model });

  assert.equal(strategy.name, "reflect:controlled");
  assert.equal(result.answer, "Approved answer.");
  assert.deepEqual(seenInputs, ["Original request."]);
  assert.deepEqual(seenModels, [model]);
  assert.equal(model.inputs.length, 1);
});

test("reflection regenerates a rejected answer with its feedback and stops on approval", async () => {
  const model = new ControlledCriticModel([
    rejection("Cite the observed service status specifically."),
    { aproved: true, feedback: "The revised answer matches the observations." },
  ]);
  const seenInputs: string[] = [];
  const seenModels: Array<ReasoningOptions["model"]> = [];
  const strategy = withReflection(controlledStrategy(["First answer.", "Revised answer."], seenInputs, seenModels));

  const result = await strategy.run("Original request.", { model });

  assert.equal(result.answer, "Revised answer.");
  assert.equal(seenInputs.length, 2);
  assert.equal(seenInputs[0], "Original request.");
  assert.match(seenInputs[1], /Original request\./);
  assert.match(seenInputs[1], /Cite the observed service status specifically\./);
  assert.deepEqual(seenModels, [model, model]);
  assert.equal(model.inputs.length, 2);
});

test("reflection defaults to at most two regenerations", async () => {
  const model = new ControlledCriticModel([
    rejection("First correction."),
    rejection("Second correction."),
    rejection("Final correction."),
  ]);
  const seenInputs: string[] = [];
  const strategy = withReflection(controlledStrategy(["Answer one.", "Answer two.", "Answer three."], seenInputs, []));

  const result = await strategy.run("Original request.", { model });

  assert.equal(result.answer, "Answer three.");
  assert.equal(seenInputs.length, 3);
  assert.equal(model.inputs.length, 3);
  const critiques = result.trace.filter((event) => event.type === "critique");
  assert.equal(critiques.length, 3);
  assert.ok(critiques.every((event) => JSON.parse(event.content).aproved === false));
  assert.equal(result.metrics.llmCalls, 9);
  assert.ok(result.metrics.latencyMs >= 15);
});

test("reflection records critique decisions and aggregates attempt and critic metrics", async () => {
  const model = new ControlledCriticModel([
    rejection("Use only the observed service status."),
    { aproved: true, feedback: "The revised answer is supported." },
  ]);
  const seenInputs: string[] = [];
  const strategy: ReasoningStrategy = {
    name: "trace-fixture",
    async run(input) {
      seenInputs.push(input);
      const attempt = seenInputs.length;
      return {
        answer: `Answer ${attempt}.`,
        trace: [
          { type: "thought", content: "Private reasoning not available as critic evidence." },
          { type: "action", content: "Unnecessary critic evidence", tool: "hidden_tool", args: {} },
          { type: "observation", content: `Observed status for attempt ${attempt}.` },
          { type: "answer", content: `Answer ${attempt}.` },
        ],
        metrics: { llmCalls: attempt + 1, latencyMs: attempt * 5, iterations: attempt },
      };
    },
  };

  const result = await withReflection(strategy).run("Original request.", { model });
  const critiqueEvents = result.trace.filter((event) => event.type === "critique");
  const criticInputs = JSON.stringify(model.inputs);

  assert.equal(result.answer, "Answer 2.");
  assert.deepEqual(
    result.trace.map((event) => event.type),
    ["thought", "action", "observation", "answer", "critique", "thought", "action", "observation", "answer", "critique"],
  );
  assert.deepEqual(
    critiqueEvents.map((event) => JSON.parse(event.content)),
    [
      { aproved: false, feedback: "Use only the observed service status." },
      { aproved: true, feedback: "The revised answer is supported." },
    ],
  );
  assert.equal(result.metrics.llmCalls, 7);
  assert.ok(result.metrics.latencyMs >= 15);
  assert.equal(result.metrics.iterations, 3);
  assert.match(criticInputs, /Original request\./);
  assert.match(criticInputs, /Observed status for attempt/);
  assert.doesNotMatch(criticInputs, /Private reasoning not available/);
  assert.doesNotMatch(criticInputs, /Unnecessary critic evidence/);
  assert.deepEqual(seenInputs, [
    "Original request.",
    "Original request.\n\nCritic feedback for revising the previous answer:\nUse only the observed service status.",
  ]);
});

test("maxReflections zero returns the base result without invoking the critic", async () => {
  const model = new ControlledCriticModel([]);
  const seenInputs: string[] = [];
  const strategy = withReflection(controlledStrategy(["Base answer."], seenInputs, []), { maxReflections: 0 });

  const result = await strategy.run("Original request.", { model });

  assert.equal(result.answer, "Base answer.");
  assert.deepEqual(seenInputs, ["Original request."]);
  assert.equal(model.inputs.length, 0);
});

test("reflection rejects invalid limits and propagates strategy and critic errors", async () => {
  assert.throws(() => withReflection(controlledStrategy([], [], []), { maxReflections: -1 }));

  const strategyError = new Error("Strategy failed.");
  const failingStrategy: ReasoningStrategy = {
    name: "failing",
    async run() {
      throw strategyError;
    },
  };
  await assert.rejects(
    withReflection(failingStrategy, { maxReflections: 0 }).run("Original request."),
    (error: unknown) => error === strategyError,
  );

  const criticError = new Error("Critic failed.");
  const model = new ControlledCriticModel([criticError]);
  const seenModels: Array<ReasoningOptions["model"]> = [];
  await assert.rejects(
    withReflection(controlledStrategy(["Answer."], [], seenModels)).run("Original request.", { model }),
    (error: unknown) => error === criticError,
  );

  const invalidModel = new ControlledCriticModel([{ approved: true, feedback: "Wrong field name." }]);
  await assert.rejects(
    withReflection(controlledStrategy(["Answer."], [], [])).run("Original request.", { model: invalidModel }),
    z.ZodError,
  );
});
