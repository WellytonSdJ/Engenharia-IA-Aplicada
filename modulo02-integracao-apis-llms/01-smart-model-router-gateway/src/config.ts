console.assert(
  process.env.OPENROUTER_API_KEY,
  "OPENROUTER_API_KEY is not set in env variables",
);

export type ModelConfig = {
  apiKey: string;
  httpReferer: string;
  xTitle: string;
  port: number;
  models: string[];
  temperature: number;
  maxTokens: number;
  systemPrompt: string;

  provider: {
    sort: {
      by: string;
      partition: string;
    };
  };
};

export const config: ModelConfig = {
  apiKey: process.env.OPENROUTER_API_KEY!,
  httpReferer: "http://pos-ia.com",
  xTitle: "SmartModelRouterGateway",
  port: 3000,
  models: [
    // top 4 para a listagem ordenada por preço
    "arcee-ai/trinity-large-preview:free",
    // teste
    "nvidia/nemotron-3-ultra-550b-a55b:free",
  ],
  temperature: 0.2,
  maxTokens: 100,
  systemPrompt:
    "Voce é um assistente inteligente que responde perguntas de forma clara e objetiva.",
  provider: {
    sort: {
      by: "throughput",
      // by: 'latency',
      // by: 'price',
      partition: "none",
    },
  },
};
