# SqliteOpsStore Contract

`src/models/sqlite-store.ts` exporta `SqliteOpsStore`, que implementa a interface `OpsStore` já definida em `src/models/in-memory-store.ts`:

```ts
interface OpsStore {
  listAlerts(status?: AlertFilter): Promise<AlertWithService[]>;
  listIncidents(status?: IncidentFilter): Promise<IncidentWithService[]>;
  createIncident(input: { title: string; service: string; severity: Severity }): Promise<Incident>;
  resolveIncident(id: string): Promise<Incident>;
  reset(): Promise<void>;
  readSummary(): Promise<StoreSummary>;
  // + novo nesta feature:
  findRunbook(service: string): Promise<Runbook | null>;
}
```

## Construção

```ts
new SqliteOpsStore(path?: string)
```

- `path` default: valor de `process.env.OPSPILOT_DB`, ou `./data/opspilot.db` se a env var não estiver definida.
- `path === ":memory:"` cria um banco em memória (usado por testes e por bench/arena quando este store for usado em cenários reprodutíveis).
- O construtor garante, antes de qualquer query:
  1. o diretório do arquivo existe (cria se faltar, exceto para `:memory:`);
  2. as 4 tabelas existem (`CREATE TABLE IF NOT EXISTS`, ver `data-model.md`);
  3. nenhuma linha é apagada ou duplicada só por reabrir o mesmo arquivo.

## Métodos

### `listAlerts(status = "firing")`

- Valida `status` com `AlertFilterSchema`.
- `SELECT` com prepared statement, filtrando por `status` exceto quando `"all"`.
- Faz `JOIN` com `services` para montar `AlertWithService`; se o `service_id` referenciado não existir (não deveria, dada a FK), lança `ResourceNotFoundError`.
- Retorno validado por `AlertWithServiceSchema` (array).

### `listIncidents(status = "open")`

- Mesma forma de `listAlerts`, sobre `incidents`/`IncidentFilterSchema`/`IncidentWithServiceSchema`.
- Inclui `summary` no retorno (pode ser `null`).

### `createIncident({ title, service, severity })`

- Valida `title` não vazio e `severity` com `SeveritySchema`; falha → `ValidationError`, nenhuma escrita ocorre.
- Busca `service` por nome (`SELECT ... WHERE name = ?`); não encontrado → `ResourceNotFoundError` listando os nomes de serviço conhecidos (mesma mensagem de hoje, para o agente se corrigir).
- `INSERT` do incidente com `status = 'open'`, `resolved_at = NULL`, `summary = NULL`, `id` gerado (`incident-${crypto.randomUUID()}`).
- Retorna o incidente criado, validado por `IncidentSchema`.

### `resolveIncident(id)`

- Busca o incidente por `id`. Não encontrado → `ResourceNotFoundError`.
- Se `status !== 'open'` → `InvalidTransitionError`, nenhuma escrita ocorre.
- `UPDATE` para `status = 'resolved'`, `resolved_at = <agora>`.
- A busca + verificação + update ocorrem dentro de uma única transação SQLite, para que duas chamadas concorrentes a `resolveIncident(mesmoId)` não resolvam o mesmo incidente duas vezes de forma inconsistente (FR-008).

### `findRunbook(service)` — novo

- Busca o serviço por nome. Não encontrado → `ResourceNotFoundError` (mesma mensagem padrão dos serviços conhecidos).
- Serviço encontrado, sem runbook associado → retorna `null` (não é erro).
- Serviço com runbook → retorna `Runbook` validado por `RunbookSchema`.

### `reset()`

- Idempotente: apaga `incidents`, `alerts`, `services`, `runbooks` e recarrega o seed do cenário "Mercadinho" (`data-model.md`), dentro de uma transação.
- Chamado por `seed()` (`src/services/seed.ts`) e pelos testes que precisam de um estado inicial conhecido.

### `readSummary()`

- `COUNT(*)` por tabela/filtro, igual ao contrato já existente (`services`, `alerts`, `firing`, `resolved`, `incidents`).

## Erros

Mesmas classes já existentes em `src/services/errors.ts` (`ValidationError`, `ResourceNotFoundError`, `InvalidTransitionError`) — o store não introduz nenhuma classe de erro nova. Nenhuma operação deixa mutação parcial em caso de erro.

## Garantias não funcionais

- Toda query usa prepared statement (`db.prepare(sql)`); nenhuma interpolação de valor de domínio em string SQL.
- Toda coluna que espelha um enum do domínio (`severity`, `status`) tem `CHECK` na definição da tabela.
- DDL de criação de tabela é idempotente (`IF NOT EXISTS`); chamar o construtor duas vezes sobre o mesmo arquivo não apaga dados.
