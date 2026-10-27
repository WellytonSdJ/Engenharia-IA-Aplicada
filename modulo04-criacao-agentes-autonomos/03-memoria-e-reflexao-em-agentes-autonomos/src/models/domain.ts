import { z } from "zod";

export const AlertStatusSchema = z.enum(["firing", "resolved"]);
export const AlertFilterSchema = z.enum(["firing", "resolved", "all"]);
export const IncidentStatusSchema = z.enum(["open", "resolved"]);
export const IncidentFilterSchema = z.enum(["open", "resolved", "all"]);
export const SeveritySchema = z.enum(["low", "medium", "high", "critical"]);

export const ServiceSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  team: z.string().min(1),
  description: z.string(),
});

export const AlertSchema = z.object({
  id: z.string().min(1),
  serviceId: z.string().min(1),
  title: z.string().min(1),
  severity: SeveritySchema,
  status: AlertStatusSchema,
  createdAt: z.date(),
  resolvedAt: z.date().nullable(),
});

export const IncidentSchema = z.object({
  id: z.string().min(1),
  title: z.string().min(1),
  serviceId: z.string().min(1),
  severity: SeveritySchema,
  status: IncidentStatusSchema,
  createdAt: z.date(),
  resolvedAt: z.date().nullable(),
  summary: z.string().nullable(),
});

export const RunbookSchema = z.object({
  id: z.string().min(1),
  serviceId: z.string().min(1),
  title: z.string().min(1),
  content: z.string().min(1),
});

export const AlertWithServiceSchema = AlertSchema.extend({
  service: ServiceSchema,
});
export const IncidentWithServiceSchema = IncidentSchema.extend({
  service: ServiceSchema,
});

export const StoreSummarySchema = z.object({
  services: z.number().int().nonnegative(),
  alerts: z.number().int().nonnegative(),
  firing: z.number().int().nonnegative(),
  resolved: z.number().int().nonnegative(),
  incidents: z.number().int().nonnegative(),
});

export const MessageRoleSchema = z.enum(["user", "assistant"]);

export const ConversationSchema = z.object({
  id: z.string().min(1),
  createdAt: z.string(),
});

export const MessageSchema = z.object({
  id: z.string().min(1),
  conversationId: z.string().min(1),
  role: MessageRoleSchema,
  content: z.string(),
  createdAt: z.string(),
});

export type AlertStatus = z.infer<typeof AlertStatusSchema>;
export type AlertFilter = z.infer<typeof AlertFilterSchema>;
export type IncidentStatus = z.infer<typeof IncidentStatusSchema>;
export type IncidentFilter = z.infer<typeof IncidentFilterSchema>;
export type Severity = z.infer<typeof SeveritySchema>;
export type Service = z.infer<typeof ServiceSchema>;
export type Alert = z.infer<typeof AlertSchema>;
export type AlertWithService = z.infer<typeof AlertWithServiceSchema>;
export type Incident = z.infer<typeof IncidentSchema>;
export type IncidentWithService = z.infer<typeof IncidentWithServiceSchema>;
export type StoreSummary = z.infer<typeof StoreSummarySchema>;
export type Runbook = z.infer<typeof RunbookSchema>;
export type MessageRole = z.infer<typeof MessageRoleSchema>;
export type Conversation = z.infer<typeof ConversationSchema>;
export type Message = z.infer<typeof MessageSchema>;
