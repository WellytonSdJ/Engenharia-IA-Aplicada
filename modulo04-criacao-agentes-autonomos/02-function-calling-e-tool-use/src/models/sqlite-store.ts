import { DatabaseSync, type StatementSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import {
  AlertFilterSchema,
  AlertWithServiceSchema,
  IncidentSchema,
  IncidentFilterSchema,
  IncidentWithServiceSchema,
  RunbookSchema,
  ServiceSchema,
  SeveritySchema,
  StoreSummarySchema,
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
import { type OpsStore } from "./in-memory-store.js";
import { InvalidTransitionError, ResourceNotFoundError, ValidationError } from "../services/errors.js";

interface ServiceRow {
  id: string;
  name: string;
  team: string;
  description: string;
}

interface IncidentRow {
  id: string;
  title: string;
  service_id: string;
  severity: string;
  status: string;
  created_at: string;
  resolved_at: string | null;
  summary: string | null;
}

interface AlertRow {
  id: string;
  service_id: string;
  title: string;
  severity: string;
  status: string;
  created_at: string;
  resolved_at: string | null;
}

interface RunbookRow {
  id: string;
  service_id: string;
  title: string;
  content: string;
}

const toService = (row: ServiceRow): Service =>
  ServiceSchema.parse({ id: row.id, name: row.name, team: row.team, description: row.description });

const toIncident = (row: IncidentRow): Incident =>
  IncidentSchema.parse({
    id: row.id,
    title: row.title,
    serviceId: row.service_id,
    severity: row.severity,
    status: row.status,
    createdAt: new Date(row.created_at),
    resolvedAt: row.resolved_at === null ? null : new Date(row.resolved_at),
    summary: row.summary,
  });

const toAlert = (row: AlertRow, service: Service): AlertWithService =>
  AlertWithServiceSchema.parse({
    id: row.id,
    serviceId: row.service_id,
    title: row.title,
    severity: row.severity,
    status: row.status,
    createdAt: new Date(row.created_at),
    resolvedAt: row.resolved_at === null ? null : new Date(row.resolved_at),
    service,
  });

const toRunbook = (row: RunbookRow): Runbook =>
  RunbookSchema.parse({ id: row.id, serviceId: row.service_id, title: row.title, content: row.content });

export class SqliteOpsStore implements OpsStore {
  private readonly db: DatabaseSync;

  constructor(path: string = process.env.OPSPILOT_DB ?? "./data/opspilot.db") {
    if (path !== ":memory:") {
      mkdirSync(dirname(path), { recursive: true });
    }
    this.db = new DatabaseSync(path);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS services (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL UNIQUE,
        team        TEXT NOT NULL,
        description TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS alerts (
        id          TEXT PRIMARY KEY,
        service_id  TEXT NOT NULL REFERENCES services(id),
        title       TEXT NOT NULL,
        severity    TEXT NOT NULL CHECK (severity IN ('low','medium','high','critical')),
        status      TEXT NOT NULL CHECK (status IN ('firing','resolved')),
        created_at  TEXT NOT NULL,
        resolved_at TEXT
      );
      CREATE TABLE IF NOT EXISTS incidents (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        service_id  TEXT NOT NULL REFERENCES services(id),
        severity    TEXT NOT NULL CHECK (severity IN ('low','medium','high','critical')),
        status      TEXT NOT NULL CHECK (status IN ('open','resolved')),
        created_at  TEXT NOT NULL,
        resolved_at TEXT,
        summary     TEXT
      );
      CREATE TABLE IF NOT EXISTS runbooks (
        id          TEXT PRIMARY KEY,
        service_id  TEXT NOT NULL UNIQUE REFERENCES services(id),
        title       TEXT NOT NULL,
        content     TEXT NOT NULL
      );
    `);
  }

  async listAlerts(status: AlertFilter = "firing"): Promise<AlertWithService[]> {
    const filter = AlertFilterSchema.parse(status);
    const statement: StatementSync = filter === "all"
      ? this.db.prepare("SELECT * FROM alerts ORDER BY created_at ASC")
      : this.db.prepare("SELECT * FROM alerts WHERE status = ? ORDER BY created_at ASC");
    const rows = (filter === "all" ? statement.all() : statement.all(filter)) as unknown as AlertRow[];
    return rows.map((row) => toAlert(row, this.getServiceById(row.service_id)));
  }

  async listIncidents(status: IncidentFilter = "open"): Promise<IncidentWithService[]> {
    const filter = IncidentFilterSchema.parse(status);
    const statement: StatementSync = filter === "all"
      ? this.db.prepare("SELECT * FROM incidents ORDER BY created_at ASC")
      : this.db.prepare("SELECT * FROM incidents WHERE status = ? ORDER BY created_at ASC");
    const rows = (filter === "all" ? statement.all() : statement.all(filter)) as unknown as IncidentRow[];
    return rows.map((row) => IncidentWithServiceSchema.parse({ ...toIncident(row), service: this.getServiceById(row.service_id) }));
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
    const service = this.findServiceByName(input.service);
    if (!service) {
      throw new ResourceNotFoundError(
        `Service ${input.service} was not found. Known services: ${this.listServiceNames().join(", ")}.`,
      );
    }
    const id = `incident-${randomUUID()}`;
    const createdAt = new Date().toISOString();
    this.db.prepare(
      "INSERT INTO incidents (id, title, service_id, severity, status, created_at, resolved_at, summary) VALUES (?, ?, ?, ?, 'open', ?, NULL, NULL)",
    ).run(id, title, service.id, severity.data, createdAt);
    return IncidentSchema.parse({
      id,
      title,
      serviceId: service.id,
      severity: severity.data,
      status: "open",
      createdAt: new Date(createdAt),
      resolvedAt: null,
      summary: null,
    });
  }

  async resolveIncident(id: string): Promise<Incident> {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const row = this.db.prepare("SELECT * FROM incidents WHERE id = ?").get(id) as unknown as IncidentRow | undefined;
      if (!row) {
        throw new ResourceNotFoundError(`Incident ${id} was not found.`);
      }
      if (row.status !== "open") {
        throw new InvalidTransitionError(`Incident ${id} is already resolved.`);
      }
      const resolvedAt = new Date().toISOString();
      this.db.prepare("UPDATE incidents SET status = 'resolved', resolved_at = ? WHERE id = ?").run(resolvedAt, id);
      this.db.exec("COMMIT");
      return toIncident({ ...row, status: "resolved", resolved_at: resolvedAt });
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }

  async findRunbook(service: string): Promise<Runbook | null> {
    const found = this.findServiceByName(service);
    if (!found) {
      throw new ResourceNotFoundError(
        `Service ${service} was not found. Known services: ${this.listServiceNames().join(", ")}.`,
      );
    }
    const row = this.db.prepare("SELECT * FROM runbooks WHERE service_id = ?").get(found.id) as unknown as RunbookRow | undefined;
    return row ? toRunbook(row) : null;
  }

  async reset(): Promise<void> {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      this.db.exec("DELETE FROM incidents");
      this.db.exec("DELETE FROM alerts");
      this.db.exec("DELETE FROM runbooks");
      this.db.exec("DELETE FROM services");
      this.seed();
      this.db.exec("COMMIT");
    } catch (error) {
      this.db.exec("ROLLBACK");
      throw error;
    }
  }

  async readSummary(): Promise<StoreSummary> {
    const services = this.count("SELECT COUNT(*) AS count FROM services");
    const alerts = this.count("SELECT COUNT(*) AS count FROM alerts");
    const firing = this.count("SELECT COUNT(*) AS count FROM alerts WHERE status = 'firing'");
    const resolved = this.count("SELECT COUNT(*) AS count FROM alerts WHERE status = 'resolved'");
    const incidents = this.count("SELECT COUNT(*) AS count FROM incidents");
    return StoreSummarySchema.parse({ services, alerts, firing, resolved, incidents });
  }

  private count(sql: string): number {
    const row = this.db.prepare(sql).get() as unknown as { count: number };
    return row.count;
  }

  private getServiceById(id: string): Service {
    const row = this.db.prepare("SELECT * FROM services WHERE id = ?").get(id) as unknown as ServiceRow | undefined;
    if (!row) {
      throw new ResourceNotFoundError(`Service ${id} was not found.`);
    }
    return toService(row);
  }

  private findServiceByName(name: string): Service | null {
    const row = this.db.prepare("SELECT * FROM services WHERE name = ?").get(name.trim()) as unknown as ServiceRow | undefined;
    return row ? toService(row) : null;
  }

  private listServiceNames(): string[] {
    const rows = this.db.prepare("SELECT name FROM services ORDER BY name ASC").all() as unknown as { name: string }[];
    return rows.map((row) => row.name);
  }

  private seed(): void {
    const insertService = this.db.prepare("INSERT INTO services (id, name, team, description) VALUES (?, ?, ?, ?)");
    for (const service of SEED_SERVICES) {
      insertService.run(service.id, service.name, service.team, service.description);
    }
    const insertAlert = this.db.prepare(
      "INSERT INTO alerts (id, service_id, title, severity, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
    );
    for (const alert of SEED_ALERTS) {
      insertAlert.run(alert.id, alert.serviceId, alert.title, alert.severity, alert.status, alert.createdAt, alert.resolvedAt);
    }
    const insertRunbook = this.db.prepare(
      "INSERT INTO runbooks (id, service_id, title, content) VALUES (?, ?, ?, ?)",
    );
    for (const runbook of SEED_RUNBOOKS) {
      insertRunbook.run(runbook.id, runbook.serviceId, runbook.title, runbook.content);
    }
  }
}

const SEED_SERVICES = [
  { id: "svc-api", name: "api", team: "platform", description: "Public API" },
  { id: "svc-checkout", name: "checkout", team: "commerce", description: "Checkout service" },
  { id: "svc-payments", name: "payments", team: "finance", description: "Payment processing" },
  { id: "svc-identity", name: "identity", team: "security", description: "Identity service" },
  { id: "svc-notifications", name: "notifications", team: "messaging", description: "Notification delivery" },
  { id: "svc-catalog", name: "catalog", team: "commerce", description: "Product catalog" },
] as const;

const SEED_ALERTS = [
  { id: "alert-api-1", serviceId: "svc-api", title: "API error rate elevated", severity: "high", status: "firing", createdAt: "2026-09-28T08:00:00.000Z", resolvedAt: null },
  { id: "alert-checkout-1", serviceId: "svc-checkout", title: "Checkout latency elevated", severity: "medium", status: "firing", createdAt: "2026-09-28T08:10:00.000Z", resolvedAt: null },
  { id: "alert-payments-1", serviceId: "svc-payments", title: "Payment failures detected", severity: "critical", status: "firing", createdAt: "2026-09-28T08:20:00.000Z", resolvedAt: null },
  { id: "alert-api-2", serviceId: "svc-api", title: "API saturation", severity: "medium", status: "resolved", createdAt: "2026-09-27T08:00:00.000Z", resolvedAt: "2026-09-27T08:30:00.000Z" },
  { id: "alert-identity-1", serviceId: "svc-identity", title: "Authentication errors", severity: "high", status: "resolved", createdAt: "2026-09-27T09:00:00.000Z", resolvedAt: "2026-09-27T09:15:00.000Z" },
  { id: "alert-notifications-1", serviceId: "svc-notifications", title: "Notification queue backlog", severity: "low", status: "resolved", createdAt: "2026-09-27T10:00:00.000Z", resolvedAt: "2026-09-27T10:45:00.000Z" },
] as const;

const SEED_RUNBOOKS = [
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
] as const;
