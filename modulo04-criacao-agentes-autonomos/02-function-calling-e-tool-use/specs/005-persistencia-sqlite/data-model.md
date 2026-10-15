# Data Model: Persistência Real de Operações

Reflete os tipos já existentes em `src/models/domain.ts`, mais o conceito novo (`Runbook`) e o campo novo em `Incident` (`summary`), e mapeia cada um para a tabela SQLite correspondente.

## Entidades

### Service (sem mudança de schema)

| Campo | Tipo (Zod) | Regra |
| --- | --- | --- |
| `id` | `string().min(1)` | chave primária |
| `name` | `string().min(1)` | único |
| `team` | `string().min(1)` | — |
| `description` | `string()` | — |

### Alert (sem mudança de schema)

| Campo | Tipo (Zod) | Regra |
| --- | --- | --- |
| `id` | `string().min(1)` | chave primária |
| `serviceId` | `string().min(1)` | referencia `Service.id` |
| `title` | `string().min(1)` | — |
| `severity` | `SeveritySchema` (`low\|medium\|high\|critical`) | domínio fechado |
| `status` | `AlertStatusSchema` (`firing\|resolved`) | domínio fechado |
| `createdAt` | `date()` | — |
| `resolvedAt` | `date().nullable()` | nulo enquanto `firing` |

### Incident (+ `summary` opcional)

| Campo | Tipo (Zod) | Regra |
| --- | --- | --- |
| `id` | `string().min(1)` | chave primária |
| `title` | `string().min(1)` | — |
| `serviceId` | `string().min(1)` | referencia `Service.id` |
| `severity` | `SeveritySchema` | domínio fechado |
| `status` | `IncidentStatusSchema` (`open\|resolved`) | domínio fechado; transição única `open -> resolved` |
| `createdAt` | `date()` | — |
| `resolvedAt` | `date().nullable()` | nulo enquanto `open` |
| `summary` | `string().nullable()` **novo** | opcional; sem capacidade hoje que o preencha (ver Assumptions da spec) |

### Runbook (novo)

| Campo | Tipo (Zod) | Regra |
| --- | --- | --- |
| `id` | `string().min(1)` | chave primária |
| `serviceId` | `string().min(1)` | referencia `Service.id`; **único** (no máximo 1 runbook por serviço) |
| `title` | `string().min(1)` | — |
| `content` | `string().min(1)` | passos recomendados, texto livre (markdown simples) |

Relação: `Service 1 --- 0..1 Runbook`. Buscar runbook por nome de serviço precisa diferenciar dois casos de ausência:
- serviço não existe → `ResourceNotFoundError`.
- serviço existe mas não tem runbook → retorno "não encontrado" tratado como observação de domínio, não como erro de sistema (mesmo padrão hoje usado por `resolveIncident`/`createIncident`).

## Esquema SQLite (DDL conceitual)

```sql
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
```

Datas são gravadas como texto ISO-8601 (`Date#toISOString()`) e reconstruídas como `Date` na leitura, para casar com `z.date()` em `domain.ts`.

## Seed (cenário "Mercadinho")

Mesmo conjunto hoje hardcoded em `DEFAULT_SERVICES`/`DEFAULT_ALERTS` (`src/models/in-memory-store.ts`), mais os runbooks:

- **Services (6)**: `api`, `checkout`, `payments`, `identity`, `notifications`, `catalog` (ids/times/descrições iguais aos já existentes).
- **Alerts (6)**: os mesmos 6 já definidos (3 `firing`: api/high, checkout/medium, payments/critical; 3 `resolved`: api/medium, identity/high, notifications/low).
- **Runbooks (3)**: um para `checkout`, um para `payments`, um para `identity` — conteúdo placeholder com passos de triagem coerentes com a descrição do serviço. `api`, `notifications` e `catalog` ficam sem runbook (caso esperado, não erro).

O seed é idempotente: roda `reset()` (limpa e recarrega) ou, quando chamado sem intenção de reset, verifica se as tabelas já têm os dados esperados antes de inserir — a store garante isso via `INSERT OR IGNORE`/verificação de contagem, nunca duplicando linhas.

## Transições de estado

```
Incident: open --(resolveIncident)--> resolved   (única transição; resolver de novo = InvalidTransitionError)
Alert:    firing / resolved definidos só pelo seed nesta feature (sem tool de transição de alerta)
```
