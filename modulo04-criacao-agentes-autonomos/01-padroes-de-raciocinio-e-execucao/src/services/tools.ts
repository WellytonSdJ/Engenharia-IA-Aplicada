import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  AlertFilterSchema,
  AlertWithServiceSchema,
  IncidentFilterSchema,
  IncidentSchema,
  IncidentWithServiceSchema,
  SeveritySchema,
} from "../models/domain.js";
import { type OpsStore } from "../models/in-memory-store.js";
import { JsonOpsStore } from "../models/json-store.js";
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
  status: AlertFilterSchema.default("firing"),
});
export const ListAlertsOutputSchema = z.array(AlertWithServiceSchema);
export const OpenIncidentInputSchema = z.object({
  title: z.string().trim().min(1),
  service: z.string().trim().min(1),
  severity: SeveritySchema,
});
export const OpenIncidentOutputSchema = IncidentSchema;
export const ResolveIncidentInputSchema = z.object({ id: z.string().trim().min(1) });
export const ResolveIncidentOutputSchema = IncidentSchema;
export const ListIncidentsInputSchema = z.object({
  status: IncidentFilterSchema.default("open"),
});
export const ListIncidentsOutputSchema = z.array(IncidentWithServiceSchema);

export function createOpsTools(store: OpsStore) {
  const listAlerts = tool(
    async ({ status }) => executeTool(() => store.listAlerts(status), ListAlertsOutputSchema),
    {
      name: "list_alerts",
      description: "List monitoring alerts by status (firing, resolved, or all; defaults to firing) to inspect service health or current on-call alerts.",
      schema: ListAlertsInputSchema,
    },
  );
  const openIncident = tool(
    async (input) => executeTool(() => store.createIncident(input), OpenIncidentOutputSchema),
    {
      name: "open_incident",
      description: "Open an incident for an existing service with a title and supported severity.",
      schema: OpenIncidentInputSchema,
    },
  );
  const resolveIncident = tool(
    async ({ id }) => executeTool(() => store.resolveIncident(id), ResolveIncidentOutputSchema),
    {
      name: "resolve_incident",
      description: "Resolve an open incident by its identifier.",
      schema: ResolveIncidentInputSchema,
    },
  );
  const listIncidents = tool(
    async ({ status }) => executeTool(() => store.listIncidents(status), ListIncidentsOutputSchema),
    {
      name: "list_incidents",
      description: "List current incidents by status (open, resolved, or all; defaults to open) to inspect existing operational work.",
      schema: ListIncidentsInputSchema,
    },
  );
  return [listAlerts, openIncident, resolveIncident, listIncidents] as const;
}

export type OpsTools = ReturnType<typeof createOpsTools>;

export const defaultOpsStore = new JsonOpsStore();
export const opsTools = createOpsTools(defaultOpsStore);
