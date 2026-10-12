import {
  DataTypes,
  Model,
  Sequelize,
  type Transaction,
} from "sequelize";
import {
  AlertSchema,
  AlertFilterSchema,
  AlertWithServiceSchema,
  IncidentFilterSchema,
  IncidentSchema,
  IncidentWithServiceSchema,
  ServiceSchema,
  SeveritySchema,
  StoreSummarySchema,
  type AlertFilter,
  type Alert,
  type Incident,
  type IncidentFilter,
  type IncidentWithService,
  type Severity,
  type StoreSummary,
} from "./domain.js";
import { DEFAULT_ALERTS, DEFAULT_SERVICES, type OpsStore } from "./in-memory-store.js";
import { InvalidTransitionError, ResourceNotFoundError, ValidationError } from "../services/errors.js";

interface ServiceRecord extends Model {
  id: string;
  name: string;
  team: string;
  description: string;
}

interface AlertRecord extends Model {
  id: string;
  serviceId: string;
  title: string;
  severity: Severity;
  status: Alert["status"];
  createdAt: Date;
  resolvedAt: Date | null;
}

interface IncidentRecord extends Model {
  id: string;
  title: string;
  serviceId: string;
  severity: Severity;
  status: Incident["status"];
  createdAt: Date;
  resolvedAt: Date | null;
}

export class SequelizeOpsStore implements OpsStore {
  private readonly services;
  private readonly alerts;
  private readonly incidents;

  constructor(private readonly sequelize: Sequelize) {
    this.services = sequelize.define<ServiceRecord>("Service", {
      id: { type: DataTypes.STRING(64), primaryKey: true },
      name: { type: DataTypes.STRING(128), unique: true, allowNull: false },
      team: { type: DataTypes.STRING(128), allowNull: false },
      description: { type: DataTypes.TEXT, allowNull: false },
    }, { tableName: "services", timestamps: false });
    this.alerts = sequelize.define<AlertRecord>("Alert", {
      id: { type: DataTypes.STRING(64), primaryKey: true },
      serviceId: { type: DataTypes.STRING(64), allowNull: false, references: { model: "services", key: "id" } },
      title: { type: DataTypes.STRING(255), allowNull: false },
      severity: { type: DataTypes.ENUM("low", "medium", "high", "critical"), allowNull: false },
      status: { type: DataTypes.ENUM("firing", "resolved"), allowNull: false },
      createdAt: { type: DataTypes.DATE, allowNull: false },
      resolvedAt: { type: DataTypes.DATE, allowNull: true },
    }, { tableName: "alerts", timestamps: false });
    this.incidents = sequelize.define<IncidentRecord>("Incident", {
      id: { type: DataTypes.STRING(64), primaryKey: true },
      title: { type: DataTypes.STRING(255), allowNull: false },
      serviceId: { type: DataTypes.STRING(64), allowNull: false, references: { model: "services", key: "id" } },
      severity: { type: DataTypes.ENUM("low", "medium", "high", "critical"), allowNull: false },
      status: { type: DataTypes.ENUM("open", "resolved"), allowNull: false },
      createdAt: { type: DataTypes.DATE, allowNull: false },
      resolvedAt: { type: DataTypes.DATE, allowNull: true },
    }, { tableName: "incidents", timestamps: false });
  }

  async listAlerts(status: AlertFilter = "firing") {
    const validatedStatus = AlertFilterSchema.parse(status);
    const rows = await this.alerts.findAll(validatedStatus === "all" ? {} : { where: { status: validatedStatus } });
    const result = await Promise.all(rows.map(async (row) => {
      const service = await this.services.findByPk(row.serviceId);
      if (!service) {
        throw new ResourceNotFoundError(`Service ${row.serviceId} was not found.`);
      }
      return AlertWithServiceSchema.parse({
        id: row.id,
        serviceId: row.serviceId,
        title: row.title,
        severity: row.severity,
        status: row.status,
        createdAt: row.createdAt,
        resolvedAt: row.resolvedAt,
        service: { id: service.id, name: service.name, team: service.team, description: service.description },
      });
    }));
    return result;
  }

  async createIncident(input: { title: string; service: string; severity: Severity }) {
    const title = input.title.trim();
    const severity = SeveritySchema.safeParse(input.severity);
    if (!title || !severity.success) {
      throw new ValidationError("Incident title or severity is invalid.");
    }
    const service = await this.services.findOne({ where: { name: input.service.trim() } });
    if (!service) {
      throw new ResourceNotFoundError(`Service ${input.service} was not found.`);
    }
    const incident = await this.incidents.create({
      id: `incident-${crypto.randomUUID()}`,
      title,
      serviceId: service.id,
      severity: severity.data,
      status: "open",
      createdAt: new Date(),
      resolvedAt: null,
    });
    return IncidentSchema.parse(incident.get({ plain: true }));
  }

  async listIncidents(status: IncidentFilter = "open"): Promise<IncidentWithService[]> {
    const validatedStatus = IncidentFilterSchema.parse(status);
    const rows = await this.incidents.findAll(
      validatedStatus === "all" ? {} : { where: { status: validatedStatus } },
    );
    return Promise.all(rows.map(async (row) => {
      const service = await this.services.findByPk(row.serviceId);
      if (!service) {
        throw new ResourceNotFoundError(`Service ${row.serviceId} was not found.`);
      }
      return IncidentWithServiceSchema.parse({
        id: row.id,
        title: row.title,
        serviceId: row.serviceId,
        severity: row.severity,
        status: row.status,
        createdAt: row.createdAt,
        resolvedAt: row.resolvedAt,
        service: { id: service.id, name: service.name, team: service.team, description: service.description },
      });
    }));
  }

  async resolveIncident(id: string) {
    return this.sequelize.transaction(async (transaction) => {
      const incident = await this.incidents.findByPk(id, {
        transaction,
        lock: transaction.LOCK.UPDATE,
      });
      if (!incident) {
        throw new ResourceNotFoundError(`Incident ${id} was not found.`);
      }
      if (incident.status !== "open") {
        throw new InvalidTransitionError(`Incident ${id} is already resolved.`);
      }
      await incident.update({ status: "resolved", resolvedAt: new Date() }, { transaction });
      return IncidentSchema.parse(incident.get({ plain: true }));
    });
  }

  async reset(): Promise<void> {
    await this.sequelize.transaction(async (transaction: Transaction) => {
      await this.incidents.destroy({ where: {}, transaction });
      await this.alerts.destroy({ where: {}, transaction });
      await this.services.destroy({ where: {}, transaction });
      await this.services.bulkCreate(DEFAULT_SERVICES.map((service) => ServiceSchema.parse(service)), { transaction });
      await this.alerts.bulkCreate(DEFAULT_ALERTS.map((alert) => AlertSchema.parse(alert)), { transaction });
    });
  }

  async readSummary(): Promise<StoreSummary> {
    const [services, alerts, firing, resolved, incidents] = await Promise.all([
      this.services.count(),
      this.alerts.count(),
      this.alerts.count({ where: { status: "firing" } }),
      this.alerts.count({ where: { status: "resolved" } }),
      this.incidents.count(),
    ]);
    return StoreSummarySchema.parse({ services, alerts, firing, resolved, incidents });
  }
}
