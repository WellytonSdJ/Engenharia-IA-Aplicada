export type ModelConfig = {
  apiKey: string;
  httpReferer: string;
  xTitle: string;

  provider: {
    sort: {
      by: string;
      partition: string;
    };
  };

  models: string[];
  temperature: number;

  memory: {
    dbUri: string;
  };
  maxMessagesToSummary: number;
};

console.assert(
  process.env.OPENROUTER_API_KEY,
  "OPENROUTER_API_KEY is not set in environment variables",
);

export const config: ModelConfig = {
  apiKey: process.env.OPENROUTER_API_KEY!,
  httpReferer: "",
  xTitle: "IA Devs - Prompt Chaining Article Generator",
  models: [
    // "arcee-ai/trinity-large-preview:free",
    // top 4 para a listagem ordenada por preço
    // teste
    // "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-xs.2:free",
  ],
  provider: {
    sort: {
      by: "throughput", // Route to model with highest throughput (fastest response)
      partition: "none",
    },
  },
  temperature: 0.7,
  memory: {
    dbUri:
      "postgresql://postgres:mysecretpassword@localhost:5432/song_recommender",
  },
  maxMessagesToSummary: 2,
};
