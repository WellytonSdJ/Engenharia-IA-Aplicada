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

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error: unknown) => {
    console.error("Arena failed:", error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
