import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  AlertFilterSchema,
  AlertWithServiceSchema,
  IncidentFilterSchema,
  IncidentSchema,
  IncidentWithServiceSchema,
  RunbookSchema,
  SeveritySchema,
} from "../models/domain.js";
import { type OpsStore } from "../models/in-memory-store.js";
import { SqliteOpsStore } from "../models/sqlite-store.js";
import { OpsPilotError, ToolBoundaryError, ValidationError } from "./errors.js";

async function executeTool<T>(operation: () => Promise<unknown>, schema: z.ZodType<T>): Promise<string> {
  try {
    return JSON.stringify(schema.parse(await operation()));
  } catch (error) {
    if (error instanceof OpsPilotError) {
      throw new ToolBoundaryError(error.code, error.message, { cause: error });
    }
    if (error instanceof z.ZodError) {
      throw new ValidationError("Tool output did not match its contract.", { cause: error });
    }
    throw error;
  }
}

export const ListAlertsInputSchema = z.object({
  status: AlertFilterSchema.default("firing").describe(
    "Filtra alertas por status; 'firing' = ainda ativos, 'resolved' = já resolvidos, 'all' = todos. Default: firing.",
  ),
});
export const ListAlertsOutputSchema = z.array(AlertWithServiceSchema);

export const OpenIncidentInputSchema = z.object({
  title: z.string().trim().min(1).describe("Título curto e descritivo do incidente, não vazio."),
  service: z.string().trim().min(1).describe(
    "Nome exato do serviço afetado (ex.: 'checkout', 'payments'); use o nome como aparece na lista de serviços conhecida, não um nome aproximado.",
  ),
  severity: SeveritySchema.describe(
    "Severidade do incidente: critical, high, medium ou low. Se o usuário falar em sevN (plantão/ITIL), mapeie sev1=critical, sev2=high, sev3=medium, sev4=low antes de preencher este campo.",
  ),
});
export const OpenIncidentOutputSchema = IncidentSchema;

export const ResolveIncidentInputSchema = z.object({
  id: z.string().trim().min(1).describe(
    "Identificador exato do incidente a resolver, no formato retornado por open_incident/list_incidents (ex.: 'incident-...').",
  ),
});
export const ResolveIncidentOutputSchema = IncidentSchema;

export const ListIncidentsInputSchema = z.object({
  status: IncidentFilterSchema.default("open").describe(
    "Filtra incidentes por status; 'open' = ainda em andamento, 'resolved' = já encerrados, 'all' = todos. Default: open.",
  ),
});
export const ListIncidentsOutputSchema = z.array(IncidentWithServiceSchema);

export const ConsultarRunbookInputSchema = z.object({
  service: z.string().trim().min(1).describe(
    "Nome exato do serviço para buscar o runbook (ex.: 'checkout', 'payments', 'identity'); use o nome como aparece na lista de serviços conhecida.",
  ),
});
export const ConsultarRunbookOutputSchema = z.union([
  RunbookSchema.pick({ title: true, content: true }).extend({ service: z.string() }),
  z.object({ service: z.string(), found: z.literal(false) }),
]);

export const CheckProviderStatusInputSchema = z.object({
  provider: z.enum(["github", "cloudflare"]).default("github").describe(
    "Provedor externo a consultar: github ou cloudflare. Default: github.",
  ),
});

const ProviderStatusResponseSchema = z.object({
  status: z.object({
    indicator: z.string(),
    description: z.string(),
  }),
}).passthrough();

const PROVIDER_STATUS_URLS: Record<"github" | "cloudflare", string> = {
  github: "https://www.githubstatus.com/api/v2/status.json",
  cloudflare: "https://www.cloudflarestatus.com/api/v2/status.json",
};

async function fetchProviderStatus(
  provider: "github" | "cloudflare",
  fetchFn: typeof fetch,
): Promise<string> {
  const url = PROVIDER_STATUS_URLS[provider];
  let lastError = "unknown error";

  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const response = await fetchFn(url, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) {
        lastError = `HTTP ${response.status}`;
        if (response.status >= 500 && attempt === 1) {
          continue;
        }
        return `Não foi possível consultar o status de ${provider}: ${lastError}.`;
      }
      const parsed = ProviderStatusResponseSchema.safeParse(await response.json());
      if (!parsed.success) {
        return `Não foi possível consultar o status de ${provider}: resposta em formato inesperado.`;
      }
      const { indicator, description } = parsed.data.status;
      return `${indicator} — ${description}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      if (attempt === 1) {
        continue;
      }
    }
  }

  return `Não foi possível consultar o status de ${provider}: ${lastError}.`;
}

export interface OpsToolDefinition<TSchema extends z.ZodType = z.ZodType> {
  name: string;
  description: string;
  inputSchema: TSchema;
  run: (input: z.infer<TSchema>) => Promise<string>;
}

export function createOpsToolDefinitions(
  store: OpsStore,
  options: { fetchFn?: typeof fetch } = {},
): OpsToolDefinition[] {
  const fetchFn = options.fetchFn ?? fetch;
  return [
    {
      name: "list_alerts",
      description: "Use para inspecionar a saúde atual dos serviços ou ver o que está tocando o plantão agora, antes de decidir se abre um incidente. Não altera nenhum dado. Status: firing (ativos), resolved (resolvidos) ou all (todos); default firing.",
      inputSchema: ListAlertsInputSchema,
      run: async ({ status }) => executeTool(() => store.listAlerts(status), ListAlertsOutputSchema),
    },
    {
      name: "open_incident",
      description: "Use quando o usuário pedir explicitamente para abrir/registrar um incidente para um serviço — não para apenas relatar um alerta já existente sem pedido de abertura. Severidade: critical, high, medium ou low; mapeie sev1=critical, sev2=high, sev3=medium, sev4=low. Cria um incidente no store. Serviço inexistente é rejeitado sem alterar o estado, e a mensagem de erro lista os serviços conhecidos para tentar de novo com o nome certo.",
      inputSchema: OpenIncidentInputSchema,
      run: async (input) => executeTool(() => store.createIncident(input), OpenIncidentOutputSchema),
    },
    {
      name: "resolve_incident",
      description: "Use quando o usuário confirmar que um incidente específico (já com id conhecido, normalmente obtido de open_incident ou list_incidents) foi mitigado/encerrado. Altera somente um incidente aberto; id inexistente ou já resolvido é rejeitado sem alterar o estado, não sobrescreve.",
      inputSchema: ResolveIncidentInputSchema,
      run: async ({ id }) => executeTool(() => store.resolveIncident(id), ResolveIncidentOutputSchema),
    },
    {
      name: "list_incidents",
      description: "Use para ver o que já está em andamento ou já foi resolvido antes de abrir um incidente novo (evita duplicar) ou para responder 'o que está aberto agora'. Não altera nenhum dado. Status: open (em andamento), resolved (encerrados) ou all (todos); default open.",
      inputSchema: ListIncidentsInputSchema,
      run: async ({ status }) => executeTool(() => store.listIncidents(status), ListIncidentsOutputSchema),
    },
    {
      name: "consultar_runbook",
      description: "Use quando o usuário pedir os passos de investigação/mitigação recomendados para um serviço (ex.: 'qual o runbook de payments?', 'como eu investigo isso?') — não serve para listar alertas ou incidentes. Não altera nenhum dado. Serviço existente sem runbook cadastrado retorna 'found: false' (não é erro); serviço inexistente é rejeitado com a mesma mensagem de serviços conhecidos usada por open_incident.",
      inputSchema: ConsultarRunbookInputSchema,
      run: async ({ service }) => executeTool(async () => {
        const runbook = await store.findRunbook(service);
        return runbook ? { service, title: runbook.title, content: runbook.content } : { service, found: false };
      }, ConsultarRunbookOutputSchema),
    },
    {
      name: "check_provider_status",
      description: "Use quando houver suspeita de problema externo, dúvida se a causa é nossa ou do provedor (GitHub ou Cloudflare), ou dependência externa possivelmente fora do ar. Não altera nenhum dado. Consulta a página pública de status do provedor e retorna indicador + descrição oficiais em uma linha. Em caso de falha na consulta (timeout, indisponibilidade ou resposta inesperada), retorna uma mensagem de erro legível em vez de travar.",
      inputSchema: CheckProviderStatusInputSchema,
      run: async ({ provider }) => fetchProviderStatus(provider, fetchFn),
    },
  ];
}

function toLangChainTool<TSchema extends z.ZodType>(definition: OpsToolDefinition<TSchema>) {
  return tool(definition.run, {
    name: definition.name,
    description: definition.description,
    schema: definition.inputSchema,
  });
}

export function createOpsTools(store: OpsStore, options: { fetchFn?: typeof fetch } = {}) {
  const [listAlerts, openIncident, resolveIncident, listIncidents, consultarRunbook, checkProviderStatus] =
    createOpsToolDefinitions(store, options);
  return [
    toLangChainTool(listAlerts),
    toLangChainTool(openIncident),
    toLangChainTool(resolveIncident),
    toLangChainTool(listIncidents),
    toLangChainTool(consultarRunbook),
    toLangChainTool(checkProviderStatus),
  ] as const;
}

export type OpsTools = ReturnType<typeof createOpsTools>;

export const defaultOpsStore = new SqliteOpsStore();
export const opsTools = createOpsTools(defaultOpsStore);
