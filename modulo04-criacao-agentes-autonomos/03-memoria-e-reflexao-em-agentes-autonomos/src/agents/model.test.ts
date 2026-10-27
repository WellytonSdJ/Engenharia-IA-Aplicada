import assert from "node:assert/strict";
import test from "node:test";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AgentConfigurationError } from "../services/errors.js";
import { createModel, resolveModel } from "./model.js";

test("model factory reports missing provider configuration without making a request", () => {
  const apiKey = process.env.OPENROUTER_API_KEY;
  const modelName = process.env.OPENROUTER_MODEL;
  delete process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_MODEL;
  try {
    assert.throws(createModel, AgentConfigurationError);
  } finally {
    if (apiKey === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = apiKey;
    if (modelName === undefined) delete process.env.OPENROUTER_MODEL;
    else process.env.OPENROUTER_MODEL = modelName;
  }
});

test("injected model doubles bypass the real provider factory", () => {
  const fake = {} as BaseChatModel;
  assert.equal(resolveModel(fake, () => {
    throw new Error("Provider factory must not run for injected test models.");
  }), fake);
});
