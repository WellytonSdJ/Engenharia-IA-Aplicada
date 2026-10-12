import { JsonOpsStore } from "../models/json-store.js";
import { pathToFileURL } from "node:url";

export async function seed(): Promise<void> {
  const store = new JsonOpsStore();
  await store.reset();
  const summary = await store.readSummary();
  console.log(
    `Seed ready: ${summary.services} services, ${summary.alerts} alerts (${summary.firing} firing, ${summary.resolved} resolved).`,
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  seed().catch((error: unknown) => {
    console.error("Seed failed:", error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
