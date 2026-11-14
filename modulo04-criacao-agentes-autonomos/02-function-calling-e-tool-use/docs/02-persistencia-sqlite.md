# Persistência com SQLite nativo — memória que sobrevive ao reinício

> Este doc cobre a spec 005 ([`005-persistencia-sqlite`](../specs/005-persistencia-sqlite/spec.md)): o `SqliteOpsStore` ([`src/models/sqlite-store.ts`](../src/models/sqlite-store.ts)). O lado das tools dessa spec está em [01-function-calling-e-design-de-tools.md](./01-function-calling-e-design-de-tools.md). O contrato `OpsStore` e o store em memória vêm do [01 / 03-tools-e-store-deterministico.md](../../01-padroes-de-raciocinio-e-execucao/docs/03-tools-e-store-deterministico.md).
>
> **Em 30 segundos**
>
> - O OpsPilot troca o store em memória (e o JSON/MySQL de antes) por **SQLite embutido no próprio Node** (`node:sqlite`, `DatabaseSync`). Não precisa de servidor nem de pacote extra.
> - 4 tabelas: `services`, `alerts`, `incidents` e `runbooks`. O DDL é **idempotente** e todo campo de domínio fechado tem `CHECK`.
> - **Só prepared statements**: nenhuma query monta SQL concatenando texto.
> - O arquivo vem de `OPSPILOT_DB` (default `./data/opspilot.db`). Os testes usam `:memory:`. O **bench continua no store em memória**, para os cenários serem reproduzíveis.

---

## O que é: por que um agente precisa de banco

No 01, o incidente que o agente abria sumia quando o processo terminava. Era um agente com **mãos de fantasma**: ele "agia", mas nada ficava.

Com SQLite, a ação **fica registrada**: abrir um incidente via `/chat`, reiniciar o servidor e listar de novo mostra o incidente lá. É isso que transforma uma tool de demonstração numa ação real.

**Analogia:** o `InMemoryOpsStore` é um quadro branco, ótimo para treino (o bench apaga e redesenha à vontade). O `SqliteOpsStore` é o livro de ocorrências do plantão: o que entra fica.

| | `InMemoryOpsStore` | `SqliteOpsStore` |
| --- | --- | --- |
| Vive em | memória do processo | arquivo `OPSPILOT_DB` (ou `:memory:`) |
| Sobrevive ao reinício? | não | sim |
| Quem usa | bench e testes de reprodução | servidor HTTP, MCP, `seed`, tools |
| Contrato | `OpsStore` | `OpsStore` (o mesmo) |

> ✅ **Checkpoint**: os dois stores implementam a **mesma interface**. O agente não sabe (nem quer saber) em qual está escrevendo.

### Por que SQLite e não um banco "de verdade"?

- **Zero infraestrutura:** `node:sqlite` vem no Node 22+. Nada de Docker, nada de `npm install`.
- **Síncrono** (`DatabaseSync`): o código fica direto, sem `await` em cada linha de SQL.
- **`:memory:`** dá um banco descartável por teste, com o mesmo SQL de produção.

O preço é que SQLite não é feito para muitos processos escrevendo ao mesmo tempo. Para um copiloto de plantão local, isso não pesa.

---

## Como está no projeto

### 1. DDL idempotente + `CHECK`

```typescript
// src/models/sqlite-store.ts (trecho do construtor)
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
```

- `CREATE TABLE IF NOT EXISTS`: rodar o construtor 10 vezes dá o mesmo resultado que rodar 1 vez (isso é **idempotência**). Nenhum dado é apagado.
- `CHECK (...)`: o banco recusa `severity = 'urgent'` mesmo que o código deixe passar. É a **segunda linha de defesa**, depois do Zod.
- O construtor cria a pasta do arquivo (`mkdirSync(dirname(path), { recursive: true })`) quando ela não existe.

### 2. Prepared statements (e por que nunca concatenar)

```typescript
// src/models/sqlite-store.ts (createIncident)
    this.db.prepare(
      "INSERT INTO incidents (id, title, service_id, severity, status, created_at, resolved_at, summary) VALUES (?, ?, ?, ?, 'open', ?, NULL, NULL)",
    ).run(id, title, service.id, severity.data, createdAt);
```

Os `?` são buracos que o SQLite preenche **como dados**, nunca como código. Aqui isso importa em dobro: quem escreve o `title` é um **modelo de linguagem**, que pode repassar qualquer texto que o usuário digitou.

```text
❌ `INSERT ... VALUES ('${title}')`   → title = "x'); DELETE FROM incidents; --"  💥
✅ prepare("... VALUES (?)").run(title) → o texto inteiro vira só o título
```

> ✅ **Checkpoint**: argumento de tool é **entrada não confiável**. Prepared statement sempre.

### 3. Transação na transição de estado

```typescript
// src/models/sqlite-store.ts
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
```

"Ler o status e depois atualizar" são dois passos. `BEGIN IMMEDIATE` trava a escrita entre eles: ninguém (o servidor HTTP **e** o servidor MCP, que abrem o mesmo arquivo) resolve o mesmo incidente no meio do caminho. Se algo falhar, `ROLLBACK` desfaz tudo.

### 4. Linha do banco → objeto de domínio validado

```typescript
// src/models/sqlite-store.ts
const toIncident = (row: IncidentRow): Incident =>
  IncidentSchema.parse({
    id: row.id,
    title: row.title,
    serviceId: row.service_id,
    // ...
    createdAt: new Date(row.created_at),
    resolvedAt: row.resolved_at === null ? null : new Date(row.resolved_at),
    summary: row.summary,
  });
```

O banco fala `snake_case` e datas em texto ISO. O domínio fala `camelCase` e `Date`. Os mappers `toX` fazem essa tradução **e** validam com Zod: um dado corrompido no arquivo estoura aqui, longe do modelo.

### 5. Seed: o cenário Mercadinho num comando

```typescript
// src/services/seed.ts
export async function seed(): Promise<void> {
  const store = new SqliteOpsStore();
  await store.reset();
  // ...
}
```

`reset()` apaga as 4 tabelas e reinsere 6 serviços, 6 alertas (3 `firing`, 3 `resolved`) e 3 runbooks (checkout, payments, identity), tudo **numa transação**. Rodar duas vezes dá o mesmo estado, sem duplicar nada.

---

## Testes com `:memory:`

```typescript
// padrão usado em src/models/sqlite-store.test.ts e src/services/tools.test.ts
const store = new SqliteOpsStore(":memory:");
await store.reset();
```

Cada teste ganha um banco novo, que some quando o teste acaba. Sem arquivo, sem limpeza e com o mesmo SQL de produção. Os testes cobrem seed, abrir/listar/resolver, filtros de status e a recusa por `CHECK`.

---

## ⚠️ Pegadinhas

- **`npm run seed` apaga os incidentes.** Ele chama `reset()`. Rodou o seed? Os incidentes que você abriu pelo `/chat` se foram.
- **`OPSPILOT_DB` vem do `.env`.** Rodar `tsx src/...` direto ignora o `.env` e cai no default `./data/opspilot.db`. Use os scripts `npm run ...` ([01 / 09-variaveis-de-ambiente-com-env-file.md](../../01-padroes-de-raciocinio-e-execucao/docs/09-variaveis-de-ambiente-com-env-file.md)).
- **`data/` fora do git.** O banco é estado local, não código (está no `.gitignore`).
- **Bench em SQLite?** Não. O bench precisa partir sempre do mesmo estado para comparar estratégias. Por isso continua no `InMemoryOpsStore`.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Store SQLite | [`src/models/sqlite-store.ts`](../src/models/sqlite-store.ts) | DDL, `CHECK`, `prepare`, `BEGIN IMMEDIATE`, mappers `toX`, seed |
| Contrato comum | [`src/models/in-memory-store.ts`](../src/models/in-memory-store.ts) | `interface OpsStore` (+ `listIncidents`, `findRunbook`) |
| Seed | [`src/services/seed.ts`](../src/services/seed.ts) | `npm run seed` → `reset()` |
| Composição | [`src/services/tools.ts`](../src/services/tools.ts) | `defaultOpsStore = new SqliteOpsStore()` |
| Bench em memória | [`src/controllers/bench-controller.ts`](../src/controllers/bench-controller.ts) | `createStore = () => new InMemoryOpsStore()` |
| Testes | [`src/models/sqlite-store.test.ts`](../src/models/sqlite-store.test.ts) | `:memory:`, filtros, `CHECK` |
| Decisões | [`specs/005-persistencia-sqlite/research.md`](../specs/005-persistencia-sqlite/research.md) | Por que `node:sqlite`, por que sair do MySQL/JSON |

---

⬅️ Anterior: [01 · Function calling e design de tools](./01-function-calling-e-design-de-tools.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [03 · Tool externa resiliente](./03-tool-externa-resiliente.md)**
