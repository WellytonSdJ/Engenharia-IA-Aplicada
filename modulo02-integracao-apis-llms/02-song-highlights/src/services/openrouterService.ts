import { ChatOpenAI } from '@langchain/openai';
import { config, type ModelConfig } from '../config.ts';
import { SystemMessage, HumanMessage } from '@langchain/core/messages';
import type { z } from 'zod/v3';

export type LLMResponse = {
  model: string;
  content: string;
};

export class OpenRouterService {
  private llmClient: ChatOpenAI;
  private config: ModelConfig;

  constructor(configOverride?: ModelConfig) {
    this.config = configOverride ?? config;

    this.llmClient = new ChatOpenAI({
      apiKey: this.config.apiKey,
      modelName: this.config.models[0],
      temperature: this.config.temperature,
      configuration: {
        baseURL: 'https://openrouter.ai/api/v1',
        defaultHeaders: {
          'HTTP-Referer': this.config.httpReferer,
          'X-Title': this.config.xTitle,
        },
      },

      modelKwargs: {
        models: this.config.models,
        provider: this.config.provider,
      },
    });
  }

  async generateStructured<T>(
    systemPrompt: string,
    userPrompt: string,
    schema: z.ZodSchema<T>,
  ) {
    try {
      const structuredLlm = this.llmClient.withStructuredOutput(schema as z.ZodSchema);

      const messages = [
        new SystemMessage(systemPrompt),
        new HumanMessage(userPrompt),
      ];

      const data = await structuredLlm.invoke(messages);

      return {
        success: true,
        data: data as T,
      };

    } catch (error) {
      console.error('🔴 LLM Error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
