export type ModelConfig = {
  apiKey: string;
  httpReferer: string;
  xTitle: string;

  provider: {
    sort: {
      // OpenRouter aceita "throughput", "price", "latency" como critérios de roteamento
      by: string;
      partition: string;
    };
  };

  // Lista de model IDs do OpenRouter — o primeiro é o primário, os demais são fallback
  models: string[];
  temperature: number;

  memory: {
    // URI do Postgres usado pelo LangGraph para checkpointer e store
    dbUri: string;
  };
  // Quantidade de mensagens no state que dispara a sumarização automática
  maxMessagesToSummary: number;
};

// Falha rápido na inicialização se a variável obrigatória não estiver definida
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
