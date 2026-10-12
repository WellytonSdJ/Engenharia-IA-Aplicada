import { ChatOpenAI } from "@langchain/openai";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AgentConfigurationError } from "../services/errors.js";

export function createModel(): ChatOpenAI {
  const apiKey = process.env.OPENROUTER_API_KEY;
  const model = process.env.OPENROUTER_MODEL;
  if (!apiKey || !model) {
    throw new AgentConfigurationError(
      "OPENROUTER_API_KEY and OPENROUTER_MODEL are required to run an LLM strategy.",
    );
  }
  return new ChatOpenAI({
    model,
    apiKey,
    configuration: { baseURL: "https://openrouter.ai/api/v1" },
    temperature: 0,
  });
}

export function resolveModel(
  injectedModel?: BaseChatModel,
  factory: () => BaseChatModel = createModel,
): BaseChatModel {
  return injectedModel ?? factory();
}
