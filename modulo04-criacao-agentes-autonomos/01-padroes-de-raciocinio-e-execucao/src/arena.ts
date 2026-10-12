import { pathToFileURL } from "node:url";
import { formatArenaRuns, runArena } from "./controllers/arena-controller.js";
import { planAndExecuteStrategy } from "./agents/plan-and-execute.js";
import { reactStrategy } from "./agents/react.js";

export async function main(args = process.argv.slice(2)): Promise<void> {
  const runs = await runArena(args, {
    react: reactStrategy,
    "plan-and-execute": planAndExecuteStrategy,
  });
  console.log(formatArenaRuns(runs));
}

// Ponto de entrada da CLI. Rode via `npm run arena`: o script é
// `node --env-file-if-exists=.env --import tsx src/arena.ts`, que carrega o .env
// (credenciais do OpenRouter) antes deste arquivo executar. `tsx src/arena.ts` puro não carrega.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error: unknown) => {
    console.error("Arena failed:", error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
