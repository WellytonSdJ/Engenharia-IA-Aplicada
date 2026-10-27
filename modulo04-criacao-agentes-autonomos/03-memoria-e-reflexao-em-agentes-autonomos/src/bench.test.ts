import assert from "node:assert/strict";
import test from "node:test";
import type { OpsStore } from "./models/in-memory-store.js";
import type { ReasoningStrategy } from "./agents/strategy.js";
import {
  answerMentionsCount,
  formatBenchMisses,
  formatBenchTable,
  parseBenchArgs,
  runBench,
} from "./controllers/bench-controller.js";
import { benchStrategies } from "./bench.js";

// Estratégia falsa: executa `act` contra o store injetado (como um agente faria via tools)
// e devolve `answer`. Sem modelo e sem rede.
function fakeStrategy(
  name: string,
  act: (store: OpsStore, input: string) => Promise<string>,
): ReasoningStrategy {
  return {
    name,
    async run(input, options) {
      const store = options?.store;
      if (!store) throw new Error("bench must inject a store");
      const answer = await act(store, input);
      return {
        answer,
        trace: [{ type: "answer", content: answer }],
        metrics: { llmCalls: 2, latencyMs: 5, iterations: 1 },
      };
    },
  };
}

// Faz exatamente o que cada cenário pede.
const perfect = fakeStrategy("perfect", async (store, input) => {
  if (input.startsWith("quantos")) return "Há 1 alerta crítico disparando.";
  if (input.startsWith("abra três")) {
    const first = await store.createIncident({ title: "Checkout", service: "checkout", severity: "high" });
    await store.createIncident({ title: "Payments", service: "payments", severity: "high" });
    await store.createIncident({ title: "Catalog", service: "catalog", severity: "high" });
    await store.resolveIncident(first.id);
    return "Feito.";
  }
  await store.createIncident({ title: "API error rate", service: "api", severity: "high" });
  return "Incidente aberto; sobraram 2 alertas disparando.";
});

// Responde de forma convincente, mas deixa o store errado.
const sloppy = fakeStrategy("sloppy", async (store, input) => {
  if (input.startsWith("quantos")) {
    await store.createIncident({ title: "Oops", service: "api", severity: "low" });
    return "Há 1 alerta crítico.";
  }
  if (input.startsWith("abra três")) {
    await store.createIncident({ title: "Payments", service: "payments", severity: "medium" });
    await store.createIncident({ title: "Checkout", service: "checkout", severity: "high" });
    return "Abri os três incidentes e resolvi o primeiro.";
  }
  await store.createIncident({ title: "Checkout", service: "checkout", severity: "high" });
  return "Sobraram 2.";
});

const broken = fakeStrategy("broken", async () => {
  throw new Error("model returned undefined");
});

test("bench flags: --scenario (case-insensitive, list), --no-replanner, defaults and invalid values", () => {
  assert.deepEqual(parseBenchArgs([]), { scenarios: ["C1", "C2", "C3"], noReplanner: false, maxIterations: 8 });
  assert.deepEqual(parseBenchArgs(["--scenario", "c2", "--no-replanner"]).scenarios, ["C2"]);
  assert.deepEqual(parseBenchArgs(["--scenario=c1,C3"]).scenarios, ["C1", "C3"]);
  assert.equal(parseBenchArgs(["--no-replanner"]).noReplanner, true);
  assert.throws(() => parseBenchArgs(["--scenario", "C9"]), /Invalid --scenario/);
  assert.throws(() => parseBenchArgs(["--scenario"]), /Missing value/);
  assert.throws(() => parseBenchArgs(["--max-iterations", "0"]));
  assert.throws(() => parseBenchArgs(["--input", "x"]), /Unknown option/);
});

test("bench judges correctness by store state, isolates each cell and survives strategy errors", async () => {
  const rows = await runBench([], () => [perfect, sloppy, broken]);
  assert.equal(rows.length, 9);
  const verdict = Object.fromEntries(rows.map((row) => [`${row.scenario}/${row.strategy}`, row.correct]));
  assert.deepEqual(verdict, {
    "C1/perfect": true, "C1/sloppy": false, "C1/broken": false,
    "C2/perfect": true, "C2/sloppy": false, "C2/broken": false,
    "C3/perfect": true, "C3/sloppy": false, "C3/broken": false,
  });
  // A resposta do "sloppy" no C2 parece certa; quem reprova é o estado.
  const sloppyC2 = rows.find((row) => row.scenario === "C2" && row.strategy === "sloppy");
  assert.ok(sloppyC2?.reasons.some((reason) => reason.includes("esperava 3 incidentes")));
  const brokenRow = rows.find((row) => row.strategy === "broken");
  assert.equal(brokenRow?.llmCalls, null);
  assert.match(brokenRow?.reasons[0] ?? "", /erro: model returned undefined/);
});

test("bench table has the expected columns and misses list their reasons", async () => {
  const rows = await runBench(["--scenario", "C1"], () => [perfect, broken]);
  const table = formatBenchTable(rows);
  assert.match(table.split("\n")[0], /\| cenário \| estratégia \| acerto \| llmCalls \| latencyMs \|/);
  assert.match(table, /\| C1 +\| perfect +\| sim +\| 2 +\| 5 +\|/);
  assert.match(table, /\| C1 +\| broken +\| não +\| - +\|/);
  assert.match(formatBenchMisses(rows), /C1 \/ broken:\n  - erro:/);
});

test("count matcher accepts digits and number words without matching inside other numbers", () => {
  assert.ok(answerMentionsCount("Há 1 alerta crítico.", 1));
  assert.ok(answerMentionsCount("Sobraram duas.", 2));
  assert.ok(answerMentionsCount("restam três alertas", 3));
  assert.equal(answerMentionsCount("há 10 alertas", 1), false);
  assert.equal(answerMentionsCount("abri um incidente", 2), false);
});

test("bench registers react and plan-and-execute, swapping in the no-replanner variant", () => {
  assert.deepEqual(benchStrategies(false).map((strategy) => strategy.name), ["react", "plan-and-execute"]);
  assert.deepEqual(benchStrategies(true).map((strategy) => strategy.name), ["react", "plan-and-execute:no-replanner"]);
});
