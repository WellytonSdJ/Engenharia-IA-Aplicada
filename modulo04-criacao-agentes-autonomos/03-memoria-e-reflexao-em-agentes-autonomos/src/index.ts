import { pathToFileURL } from "node:url";
import { z } from "zod";
import { createModel } from "./agents/model.js";
import { startServer } from "./http/server.js";
import { createLLMLearningReflector } from "./services/learning-reflector.js";

const PortSchema = z.coerce.number().int().positive().default(3000);

export async function main(): Promise<void> {
  const port = PortSchema.parse(process.env.PORT);
  // O refletor de aprendizado (spec 010) só é ligado aqui, no servidor de verdade. Testes de
  // /chat não o recebem e por isso nunca chamam o LLM em segundo plano.
  await startServer(port, { learningReflector: createLLMLearningReflector(createModel) });
  console.log(`opsPilot HTTP listening on http://localhost:${port}`);
}

// Ponto de entrada do servidor HTTP. Rode via `npm run dev`: o script é
// `node --env-file-if-exists=.env --import tsx src/index.ts`, que carrega o .env
// (credenciais do OpenRouter) antes deste arquivo executar. `tsx src/index.ts` puro não carrega.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error: unknown) => {
    console.error("opsPilot server failed:", error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
