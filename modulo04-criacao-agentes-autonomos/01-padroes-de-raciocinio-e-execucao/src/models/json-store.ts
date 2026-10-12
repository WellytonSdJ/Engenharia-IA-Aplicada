import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { basename, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  AlertFilterSchema,
  AlertSchema,
  AlertWithServiceSchema,
  IncidentSchema,
  IncidentFilterSchema,
  IncidentWithServiceSchema,
  ServiceSchema,
  SeveritySchema,
  StoreSummarySchema,
  type Alert,
  type AlertFilter,
  type AlertWithService,
  type Incident,
  type IncidentFilter,
  type IncidentWithService,
  type Service,
  type Severity,
  type StoreSummary,
} from "./domain.js";
import { DEFAULT_ALERTS, DEFAULT_SERVICES, type OpsStore } from "./in-memory-store.js";
import { InvalidTransitionError, ResourceNotFoundError, ValidationError } from "../services/errors.js";

const JsonDate = (value: unknown): Date => {
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) {
    throw new Error("Invalid date value in JSON store.");
  }
  return date;
};

const documentSchema = {
  parse(value: unknown): OpsStoreDocument {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      throw new ValidationError("JSON store must contain an object.");
    }
    const record = value as Record<string, unknown>;
    if (!Array.isArray(record.services) || !Array.isArray(record.alerts) || !Array.isArray(record.incidents)) {
      throw new ValidationError("JSON store must contain services, alerts, and incidents arrays.");
    }
    try {
      const services = record.services.map((item) => ServiceSchema.parse(item));
      const alerts = record.alerts.map((item) => {
        if (typeof item !== "object" || item === null || Array.isArray(item)) {
          throw new Error("Alert record must be an object.");
        }
        const alert = item as Record<string, unknown>;
        return AlertSchema.parse({
          ...alert,
          createdAt: JsonDate(alert.createdAt),
          resolvedAt: alert.resolvedAt === null ? null : JsonDate(alert.resolvedAt),
        });
      });
      const incidents = record.incidents.map((item) => {
        if (typeof item !== "object" || item === null || Array.isArray(item)) {
          throw new Error("Incident record must be an object.");
        }
        const incident = item as Record<string, unknown>;
        return IncidentSchema.parse({
          ...incident,
          createdAt: JsonDate(incident.createdAt),
          resolvedAt: incident.resolvedAt === null ? null : JsonDate(incident.resolvedAt),
        });
      });
      const serviceIds = new Set(services.map((service) => service.id));
      if (serviceIds.size !== services.length) {
        throw new ValidationError("JSON store contains duplicate service identifiers.");
      }
      if (new Set(alerts.map((alert) => alert.id)).size !== alerts.length) {
        throw new ValidationError("JSON store contains duplicate alert identifiers.");
      }
      if (new Set(incidents.map((incident) => incident.id)).size !== incidents.length) {
        throw new ValidationError("JSON store contains duplicate incident identifiers.");
      }
      if (alerts.some((alert) => !serviceIds.has(alert.serviceId)) ||
          incidents.some((incident) => !serviceIds.has(incident.serviceId))) {
        throw new ValidationError("JSON store contains records referencing unknown services.");
      }
      return { services, alerts, incidents };
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError("JSON store contains an invalid operational record.", { cause: error });
    }
  },
};

export interface OpsStoreDocument {
  services: Service[];
  alerts: Alert[];
  incidents: Incident[];
}

export function createSeedDocument(): OpsStoreDocument {
  return documentSchema.parse({
    services: DEFAULT_SERVICES,
    alerts: DEFAULT_ALERTS,
    incidents: [],
  });
}

export class JsonOpsStore implements OpsStore {
  private writeQueue: Promise<void> = Promise.resolve();

  constructor(
    readonly filePath = fileURLToPath(new URL("../data/ops-store.json", import.meta.url)),
  ) {}

  async listAlerts(status: AlertFilter = "firing"): Promise<AlertWithService[]> {
    const filter = AlertFilterSchema.parse(status);
    const document = await this.readDocument();
    return document.alerts
      .filter((alert) => filter === "all" || alert.status === filter)
      .map((alert) => {
        const service = document.services.find((item) => item.id === alert.serviceId);
        if (!service) {
          throw new ResourceNotFoundError(`Service ${alert.serviceId} was not found.`);
        }
        return AlertWithServiceSchema.parse({ ...alert, service });
      });
  }

  async createIncident(input: { title: string; service: string; severity: Severity }): Promise<Incident> {
    return this.mutate(async () => {
      const title = input.title.trim();
      const severity = SeveritySchema.safeParse(input.severity);
      if (!title || !severity.success) {
        throw new ValidationError("Incident title or severity is invalid.");
      }

      const document = await this.readDocument();
      const service = document.services.find((item) => item.name === input.service.trim());
      if (!service) {
        // Mesma mensagem do InMemoryOpsStore: listar os nomes válidos permite ao agente se corrigir.
        throw new ResourceNotFoundError(
          `Service ${input.service} was not found. Known services: ${document.services.map((item) => item.name).join(", ")}.`,
        );
      }

      const incident = IncidentSchema.parse({
        id: `incident-${randomUUID()}`,
        title,
        serviceId: service.id,
        severity: severity.data,
        status: "open",
        createdAt: new Date(),
        resolvedAt: null,
      });
      document.incidents.push(incident);
      await this.writeDocument(document);
      return incident;
    });
  }

  async listIncidents(status: IncidentFilter = "open"): Promise<IncidentWithService[]> {
    const filter = IncidentFilterSchema.parse(status);
    const document = await this.readDocument();
    return document.incidents
      .filter((incident) => filter === "all" || incident.status === filter)
      .map((incident) => {
        const service = document.services.find((item) => item.id === incident.serviceId);
        if (!service) {
          throw new ResourceNotFoundError(`Service ${incident.serviceId} was not found.`);
        }
        return IncidentWithServiceSchema.parse({ ...incident, service });
      });
  }

  async resolveIncident(id: string): Promise<Incident> {
    return this.mutate(async () => {
      const document = await this.readDocument();
      const incident = document.incidents.find((item) => item.id === id);
      if (!incident) {
        throw new ResourceNotFoundError(`Incident ${id} was not found.`);
      }
      if (incident.status !== "open") {
        throw new InvalidTransitionError(`Incident ${id} is already resolved.`);
      }

      const resolved = IncidentSchema.parse({
        ...incident,
        status: "resolved",
        resolvedAt: new Date(),
      });
      document.incidents = document.incidents.map((item) => item.id === id ? resolved : item);
      await this.writeDocument(document);
      return resolved;
    });
  }

  async reset(): Promise<void> {
    return this.mutate(() => this.writeDocument(createSeedDocument()));
  }

  async readSummary(): Promise<StoreSummary> {
    const document = await this.readDocument();
    return StoreSummarySchema.parse({
      services: document.services.length,
      alerts: document.alerts.length,
      firing: document.alerts.filter((item) => item.status === "firing").length,
      resolved: document.alerts.filter((item) => item.status === "resolved").length,
      incidents: document.incidents.length,
    });
  }

  async readDocument(): Promise<OpsStoreDocument> {
    const contents = await readFile(this.filePath, "utf8");
    let value: unknown;
    try {
      value = JSON.parse(contents) as unknown;
    } catch (error) {
      throw new ValidationError(`JSON store at "${this.filePath}" is not valid JSON.`, { cause: error });
    }
    return documentSchema.parse(value);
  }

  private async writeDocument(document: OpsStoreDocument): Promise<void> {
    const validated = documentSchema.parse(document);
    const folder = dirname(this.filePath);
    const temporaryPath = `${folder}/.${basename(this.filePath)}.${process.pid}.${randomUUID()}.tmp`;
    await mkdir(folder, { recursive: true });
    try {
      await writeFile(temporaryPath, `${JSON.stringify(validated, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
      await rename(temporaryPath, this.filePath);
    } catch (error) {
      try {
        await unlink(temporaryPath);
      } catch (cleanupError) {
        if (!(cleanupError instanceof Error && "code" in cleanupError && cleanupError.code === "ENOENT")) {
          throw cleanupError;
        }
      }
      throw error;
    }
  }

  private async mutate<T>(operation: () => Promise<T>): Promise<T> {
    let release!: () => void;
    const previous = this.writeQueue;
    this.writeQueue = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }
}
