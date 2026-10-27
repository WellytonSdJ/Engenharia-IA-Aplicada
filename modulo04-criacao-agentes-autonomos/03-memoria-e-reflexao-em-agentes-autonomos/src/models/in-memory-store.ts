import {
  AlertFilterSchema,
  AlertSchema,
  AlertStatusSchema,
  AlertWithServiceSchema,
  IncidentSchema,
  IncidentFilterSchema,
  IncidentWithServiceSchema,
  RunbookSchema,
  ServiceSchema,
  SeveritySchema,
  StoreSummarySchema,
  type Alert,
  type AlertFilter,
  type AlertWithService,
  type Incident,
  type IncidentFilter,
  type IncidentWithService,
  type Runbook,
  type Service,
  type Severity,
  type StoreSummary,
} from "./domain.js";
import { InvalidTransitionError, ResourceNotFoundError, ValidationError } from "../services/errors.js";

export interface OpsStore {
  listAlerts(status?: AlertFilter): Promise<AlertWithService[]>;
  listIncidents(status?: IncidentFilter): Promise<IncidentWithService[]>;
  createIncident(input: { title: string; service: string; severity: Severity }): Promise<Incident>;
  resolveIncident(id: string): Promise<Incident>;
  reset(): Promise<void>;
  readSummary(): Promise<StoreSummary>;
  findRunbook(service: string): Promise<Runbook | null>;
}

const fixedDate = (value: string): Date => new Date(value);

export const DEFAULT_SERVICES: Service[] = [
  { id: "svc-api", name: "api", team: "platform", description: "Public API" },
  { id: "svc-checkout", name: "checkout", team: "commerce", description: "Checkout service" },
  { id: "svc-payments", name: "payments", team: "finance", description: "Payment processing" },
  { id: "svc-identity", name: "identity", team: "security", description: "Identity service" },
  { id: "svc-notifications", name: "notifications", team: "messaging", description: "Notification delivery" },
  // Sem alertas: existe para o cenário C2 do bench ("abra incidentes para checkout, payment e catalog").
  { id: "svc-catalog", name: "catalog", team: "commerce", description: "Product catalog" },
];

export const DEFAULT_RUNBOOKS: Runbook[] = [
  {
    id: "runbook-checkout",
    serviceId: "svc-checkout",
    title: "Checkout latency elevated",
    content: "1. Verifique o status do gateway de pagamento. 2. Confirme a latência do banco de pedidos. 3. Acione o time de commerce se a latência persistir acima de 5 minutos.",
  },
  {
    id: "runbook-payments",
    serviceId: "svc-payments",
    title: "Falhas no processamento de pagamentos",
    content: "1. Verifique o status do provedor de pagamentos externo. 2. Confira a taxa de erro por método de pagamento. 3. Abra incidente sev1 se a taxa de falha ultrapassar 10%.",
  },
  {
    id: "runbook-identity",
    serviceId: "svc-identity",
    title: "Erros de autenticação",
    content: "1. Verifique a disponibilidade do serviço de identidade. 2. Confirme se há rotação recente de chaves/certificados. 3. Acione o time de segurança se o erro afetar login em massa.",
  },
];

export const DEFAULT_ALERTS: Alert[] = [
  { id: "alert-api-1", serviceId: "svc-api", title: "API error rate elevated", severity: "high", status: "firing", createdAt: fixedDate("2026-09-28T08:00:00.000Z"), resolvedAt: null },
  { id: "alert-checkout-1", serviceId: "svc-checkout", title: "Checkout latency elevated", severity: "medium", status: "firing", createdAt: fixedDate("2026-09-28T08:10:00.000Z"), resolvedAt: null },
  { id: "alert-payments-1", serviceId: "svc-payments", title: "Payment failures detected", severity: "critical", status: "firing", createdAt: fixedDate("2026-09-28T08:20:00.000Z"), resolvedAt: null },
  { id: "alert-api-2", serviceId: "svc-api", title: "API saturation", severity: "medium", status: "resolved", createdAt: fixedDate("2026-09-27T08:00:00.000Z"), resolvedAt: fixedDate("2026-09-27T08:30:00.000Z") },
  { id: "alert-identity-1", serviceId: "svc-identity", title: "Authentication errors", severity: "high", status: "resolved", createdAt: fixedDate("2026-09-27T09:00:00.000Z"), resolvedAt: fixedDate("2026-09-27T09:15:00.000Z") },
  { id: "alert-notifications-1", serviceId: "svc-notifications", title: "Notification queue backlog", severity: "low", status: "resolved", createdAt: fixedDate("2026-09-27T10:00:00.000Z"), resolvedAt: fixedDate("2026-09-27T10:45:00.000Z") },
];

export class InMemoryOpsStore implements OpsStore {
  private services: Service[] = [];
  private alerts: Alert[] = [];
  private incidents: Incident[] = [];
  private runbooks: Runbook[] = [];
  private incidentSequence = 0;

  constructor() {
    this.resetSync();
  }

  async listAlerts(status: AlertFilter = "firing"): Promise<AlertWithService[]> {
    const filter = AlertFilterSchema.parse(status);
    return this.alerts
      .filter((alert) => filter === "all" || alert.status === filter)
      .map((alert) => {
        const service = this.services.find((item) => item.id === alert.serviceId);
        if (!service) {
          throw new ResourceNotFoundError(`Service ${alert.serviceId} was not found.`);
        }
        return AlertWithServiceSchema.parse({ ...alert, service });
      });
  }

  async listIncidents(status: IncidentFilter = "open"): Promise<IncidentWithService[]> {
    const filter = IncidentFilterSchema.parse(status);
    return this.incidents
      .filter((incident) => filter === "all" || incident.status === filter)
      .map((incident) => {
        const service = this.services.find((item) => item.id === incident.serviceId);
        if (!service) {
          throw new ResourceNotFoundError(`Service ${incident.serviceId} was not found.`);
        }
        return IncidentWithServiceSchema.parse({ ...incident, service });
      });
  }

  async createIncident(input: { title: string; service: string; severity: Severity }): Promise<Incident> {
    const title = input.title.trim();
    if (!title) {
      throw new ValidationError("Incident title must not be empty.");
    }
    const severity = SeveritySchema.safeParse(input.severity);
    if (!severity.success) {
      throw new ValidationError("Incident severity is not supported.", { cause: severity.error });
    }
    const service = this.services.find((item) => item.name === input.service);
    if (!service) {
      // A lista de nomes válidos vira observação para o agente: o modelo que pediu "payment"
      // lê a mensagem e consegue tentar de novo com "payments".
      throw new ResourceNotFoundError(
        `Service ${input.service} was not found. Known services: ${this.services.map((item) => item.name).join(", ")}.`,
      );
    }
    const incident = IncidentSchema.parse({
      id: `incident-${++this.incidentSequence}`,
      title,
      serviceId: service.id,
      severity: severity.data,
      status: "open",
      createdAt: new Date(),
      resolvedAt: null,
      summary: null,
    });
    this.incidents.push(incident);
    return incident;
  }

  async resolveIncident(id: string): Promise<Incident> {
    const incident = this.incidents.find((item) => item.id === id);
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
    this.incidents = this.incidents.map((item) => item.id === id ? resolved : item);
    return resolved;
  }

  async findRunbook(service: string): Promise<Runbook | null> {
    const found = this.services.find((item) => item.name === service.trim());
    if (!found) {
      throw new ResourceNotFoundError(
        `Service ${service} was not found. Known services: ${this.services.map((item) => item.name).join(", ")}.`,
      );
    }
    return this.runbooks.find((item) => item.serviceId === found.id) ?? null;
  }

  async reset(): Promise<void> {
    this.resetSync();
  }

  async readSummary(): Promise<StoreSummary> {
    return StoreSummarySchema.parse({
      services: this.services.length,
      alerts: this.alerts.length,
      firing: this.alerts.filter((item) => item.status === "firing").length,
      resolved: this.alerts.filter((item) => item.status === "resolved").length,
      incidents: this.incidents.length,
    });
  }

  private resetSync(): void {
    this.services = DEFAULT_SERVICES.map((service) => ServiceSchema.parse(service));
    this.alerts = DEFAULT_ALERTS.map((alert) => AlertSchema.parse(alert));
    this.runbooks = DEFAULT_RUNBOOKS.map((runbook) => RunbookSchema.parse(runbook));
    this.incidents = [];
    this.incidentSequence = 0;
  }
}
