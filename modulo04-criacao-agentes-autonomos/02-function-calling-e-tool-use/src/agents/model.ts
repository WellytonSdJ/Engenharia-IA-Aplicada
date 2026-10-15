import { ChatOpenAI } from "@langchain/openai";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AgentConfigurationError } from "../services/errors.js";

// As variáveis abaixo NÃO são lidas do arquivo .env por este código: quem as coloca em
// process.env é o próprio Node, via `node --env-file-if-exists=.env` nos scripts do
// package.json (a constituição proíbe `dotenv`). Se rodar `tsx src/arena.ts` direto,
// o .env é ignorado e esta função lança AgentConfigurationError.
// Estudo: docs/variaveis-de-ambiente-com-env-file.md
export function createModel(): ChatOpenAI {
  const apiKey = process.env.OPENROUTER_API_KEY;
  const model = process.env.OPENROUTER_MODEL;
  // `!valor` (e não `valor === undefined`) também barra string vazia — o caso de copiar
  // o .env.example sem preencher, que faria o erro aparecer só no provedor.
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
