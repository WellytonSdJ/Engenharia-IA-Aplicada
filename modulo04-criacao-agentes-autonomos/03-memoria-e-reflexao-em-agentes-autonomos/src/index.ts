import { pathToFileURL } from "node:url";
import { z } from "zod";
import { startServer } from "./http/server.js";

const PortSchema = z.coerce.number().int().positive().default(3000);

export async function main(): Promise<void> {
  const port = PortSchema.parse(process.env.PORT);
  await startServer(port);
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
