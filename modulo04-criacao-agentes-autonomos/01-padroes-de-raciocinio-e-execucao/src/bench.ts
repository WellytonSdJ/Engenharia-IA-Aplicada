import { pathToFileURL } from "node:url";
import { formatBenchMisses, formatBenchTable, runBench } from "./controllers/bench-controller.js";
import { createPlanAndExecuteStrategy } from "./agents/plan-and-execute.js";
import { reactStrategy } from "./agents/react.js";
import type { ReasoningStrategy } from "./agents/strategy.js";

/** As 2 estratégias do bench; `--no-replanner` troca o plan-and-execute pela versão em linha reta. */
export function benchStrategies(noReplanner: boolean): ReasoningStrategy[] {
  return [reactStrategy, createPlanAndExecuteStrategy({ replanner: !noReplanner })];
}

export async function main(args = process.argv.slice(2)): Promise<void> {
  const rows = await runBench(args, benchStrategies, undefined, (row) => {
    console.log(`${row.scenario} / ${row.strategy}: ${row.correct ? "acertou" : "errou"}`);
  });
  console.log(`\n${formatBenchTable(rows)}`);
  const misses = formatBenchMisses(rows);
  if (misses) {
    console.log(`\nPor que errou:\n${misses}`);
  }
}

// Rode via `npm run bench` (carrega o .env). Exemplos:
//   npm run bench
//   npm run bench -- --scenario C2
//   npm run bench -- --scenario c1,c3 --no-replanner
// Cada célula usa um InMemoryOpsStore novo: o bench nunca altera src/data/ops-store.json.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error: unknown) => {
    console.error("Bench failed:", error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
