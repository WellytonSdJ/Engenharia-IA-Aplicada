import { z } from "zod";
import { InMemoryOpsStore, type OpsStore } from "../models/in-memory-store.js";
import type { ReasoningStrategy } from "../agents/strategy.js";
import { ReasoningResultSchema } from "../agents/trace.js";

// O bench responde "acertou?" olhando o ESTADO do store depois da execução, não o texto da
// resposta. Uma resposta bonita com o incidente errado é erro; o store não mente.
// A arena mostra COMO cada estratégia pensou; o bench mede SE ela fez o certo.

export const ScenarioIdSchema = z.enum(["C1", "C2", "C3"]);
export type ScenarioId = z.infer<typeof ScenarioIdSchema>;

export interface BenchScenario {
  id: ScenarioId;
  label: string;
  prompt: string;
  /** Lista os motivos do erro; lista vazia = acerto. */
  check(store: OpsStore, answer: string): Promise<string[]>;
}

export interface BenchArguments {
  scenarios: ScenarioId[];
  noReplanner: boolean;
  maxIterations: number;
}

export interface BenchRow {
  scenario: ScenarioId;
  strategy: string;
  correct: boolean;
  /** `null` quando a estratégia lançou erro e não devolveu métricas. */
  llmCalls: number | null;
  latencyMs: number;
  reasons: string[];
}

/** sev2 → high: o mesmo mapeamento que a descrição da tool `open_incident` ensina ao modelo. */
const SEV2 = "high";
/** Ordem pedida no C2. O usuário escreve "payment", mas o serviço do seed é `payments`. */
const C2_EXPECTED_SERVICE_IDS = ["svc-checkout", "svc-payments", "svc-catalog"];

const NUMBER_WORDS: Record<number, string[]> = {
  0: ["zero", "nenhum", "nenhuma", "none"],
  2: ["dois", "duas", "two"],
  3: ["três", "tres", "three"],
  4: ["quatro", "four"],
  5: ["cinco", "five"],
};

/**
 * A resposta cita a contagem esperada? Aceita o dígito isolado (não casa "1" dentro de "10")
 * e o número por extenso. "um"/"one" ficam de fora: aparecem em qualquer frase ("um incidente").
 */
export function answerMentionsCount(answer: string, count: number): boolean {
  const text = answer.toLowerCase();
  if (new RegExp(`(?<!\\d)${count}(?!\\d)`).test(text)) {
    return true;
  }
  return (NUMBER_WORDS[count] ?? []).some((word) => new RegExp(`(?<![\\p{L}])${word}(?![\\p{L}])`, "u").test(text));
}

export const BENCH_SCENARIOS: readonly BenchScenario[] = [
  {
    id: "C1",
    label: "direto",
    prompt: "quantos alertas críticos estão disparando?",
    async check(store, answer) {
      // Pergunta só de leitura: o estado não pode mudar e a resposta precisa da contagem real.
      const reasons: string[] = [];
      const incidents = await store.listIncidents("all");
      const expected = (await store.listAlerts("firing")).filter((alert) => alert.severity === "critical").length;
      if (incidents.length !== 0) reasons.push(`esperava 0 incidentes (pergunta só de leitura), veio ${incidents.length}`);
      if (!answerMentionsCount(answer, expected)) reasons.push(`resposta não cita a contagem esperada (${expected})`);
      return reasons;
    },
  },
  {
    id: "C2",
    label: "estruturado",
    prompt: "abra três incidentes de sev2 para checkout, payment e catalog, nessa mesma ordem e resolva o primeiro.",
    async check(store) {
      // Tudo verificado no store: quantidade, ordem de criação, severidade e status.
      const reasons: string[] = [];
      const incidents = await store.listIncidents("all");
      const serviceIds = incidents.map((incident) => incident.serviceId);
      if (incidents.length !== 3) reasons.push(`esperava 3 incidentes, veio ${incidents.length}`);
      if (C2_EXPECTED_SERVICE_IDS.some((id, index) => serviceIds[index] !== id)) {
        reasons.push(`serviços/ordem: [${serviceIds.join(", ")}] (esperado: ${C2_EXPECTED_SERVICE_IDS.join(", ")})`);
      }
      const wrongSeverity = incidents.filter((incident) => incident.severity !== SEV2);
      if (wrongSeverity.length > 0) {
        reasons.push(`severidade diferente de ${SEV2} (sev2): ${wrongSeverity.map((item) => `${item.service.name}=${item.severity}`).join(", ")}`);
      }
      if (incidents[0] && incidents[0].status !== "resolved") reasons.push(`primeiro incidente está ${incidents[0].status} (esperado resolved)`);
      const notOpen = incidents.slice(1).filter((incident) => incident.status !== "open");
      if (notOpen.length > 0) reasons.push(`deveriam continuar open: ${notOpen.map((item) => item.service.name).join(", ")}`);
      return reasons;
    },
  },
  {
    id: "C3",
    label: "dinâmico",
    prompt: "dos alertas disparando, abra um incidente para o mais antigo e diga quantos sobraram",
    async check(store, answer) {
      // "Dinâmico": o alvo (o mais antigo) só é conhecido depois de consultar o store.
      // "Sobraram" = alertas disparando que continuam sem incidente aberto.
      const reasons: string[] = [];
      const firing = await store.listAlerts("firing");
      const oldest = [...firing].sort((a, b) => a.createdAt.getTime() - b.createdAt.getTime())[0];
      const incidents = await store.listIncidents("all");
      const remaining = firing.length - 1;
      if (incidents.length !== 1) reasons.push(`esperava 1 incidente, veio ${incidents.length}`);
      const [incident] = incidents;
      if (incident && oldest && incident.serviceId !== oldest.serviceId) {
        reasons.push(`incidente em ${incident.service.name} (o alerta mais antigo é de ${oldest.service.name})`);
      }
      if (incident && incident.status !== "open") reasons.push(`incidente está ${incident.status} (esperado open)`);
      if (!answerMentionsCount(answer, remaining)) reasons.push(`resposta não cita quantos sobraram (${remaining})`);
      return reasons;
    },
  },
];

export function parseBenchArgs(args: readonly string[]): BenchArguments {
  let scenarios: ScenarioId[] = [...ScenarioIdSchema.options];
  let noReplanner = false;
  let maxIterations = 8;

  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];
    const [flag, inlineValue] = value.includes("=") ? value.split("=", 2) : [value, undefined];
    if (flag === "--no-replanner") {
      noReplanner = true;
    } else if (flag === "--scenario" || flag === "--max-iterations") {
      const next = inlineValue ?? args[index + 1];
      if (!next || next.startsWith("--")) {
        throw new Error(`Missing value for ${flag}.`);
      }
      if (inlineValue === undefined) index += 1;
      if (flag === "--scenario") {
        // Aceita um ou vários: --scenario C2 ou --scenario c1,c3
        const parsed = z.array(ScenarioIdSchema).safeParse(next.split(",").map((item) => item.trim().toUpperCase()));
        if (!parsed.success) {
          throw new Error(`Invalid --scenario "${next}". Use: ${ScenarioIdSchema.options.join(", ")}.`);
        }
        scenarios = [...new Set(parsed.data)];
      } else {
        const parsed = Number(next);
        if (!Number.isInteger(parsed) || parsed <= 0) {
          throw new Error("--max-iterations must be a positive integer.");
        }
        maxIterations = parsed;
      }
    } else {
      throw new Error(`Unknown option "${value}".`);
    }
  }
  return { scenarios, noReplanner, maxIterations };
}

export async function runBench(
  args: readonly string[],
  strategiesFor: (noReplanner: boolean) => readonly ReasoningStrategy[],
  createStore: () => OpsStore = () => new InMemoryOpsStore(),
  onRow: (row: BenchRow) => void = () => {},
): Promise<BenchRow[]> {
  const parsed = parseBenchArgs(args);
  const scenarios = BENCH_SCENARIOS.filter((scenario) => parsed.scenarios.includes(scenario.id));
  const strategies = strategiesFor(parsed.noReplanner);
  const rows: BenchRow[] = [];

  for (const scenario of scenarios) {
    for (const strategy of strategies) {
      // Store novo (seed limpo) por célula: o C2 de uma estratégia não pode contaminar a outra.
      const store = createStore();
      const startedAt = performance.now();
      let row: BenchRow;
      try {
        const result = ReasoningResultSchema.parse(
          await strategy.run(scenario.prompt, { store, maxIterations: parsed.maxIterations }),
        );
        const reasons = await scenario.check(store, result.answer);
        row = {
          scenario: scenario.id,
          strategy: strategy.name,
          correct: reasons.length === 0,
          llmCalls: result.metrics.llmCalls,
          latencyMs: result.metrics.latencyMs,
          reasons,
        };
      } catch (error) {
        // Uma estratégia que quebra conta como erro, mas o bench segue para a próxima célula.
        row = {
          scenario: scenario.id,
          strategy: strategy.name,
          correct: false,
          llmCalls: null,
          latencyMs: performance.now() - startedAt,
          reasons: [`erro: ${error instanceof Error ? error.message.replace(/\s+/g, " ").slice(0, 200) : String(error)}`],
        };
      }
      rows.push(row);
      onRow(row);
    }
  }
  return rows;
}

export function formatBenchTable(rows: readonly BenchRow[]): string {
  const headers = ["cenário", "estratégia", "acerto", "llmCalls", "latencyMs"];
  const cells = rows.map((row) => [
    row.scenario,
    row.strategy,
    row.correct ? "sim" : "não",
    row.llmCalls === null ? "-" : String(row.llmCalls),
    String(Math.round(row.latencyMs)),
  ]);
  const widths = headers.map((header, column) => Math.max(header.length, ...cells.map((cell) => cell[column].length)));
  const line = (parts: string[]) => `| ${parts.map((part, column) => part.padEnd(widths[column])).join(" | ")} |`;
  const separator = `|${widths.map((width) => "-".repeat(width + 2)).join("|")}|`;
  return [line(headers), separator, ...cells.map(line)].join("\n");
}

export function formatBenchMisses(rows: readonly BenchRow[]): string {
  return rows
    .filter((row) => !row.correct)
    .map((row) => [`${row.scenario} / ${row.strategy}:`, ...row.reasons.map((reason) => `  - ${reason}`)].join("\n"))
    .join("\n");
}
