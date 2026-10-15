# Tools e Store Determinístico

## O que é

Um agente sem ferramentas só conversa. As tools são as **mãos** do OpsPilot: consultar alertas, abrir incidente, resolver incidente. E as mãos precisam mexer em alguma coisa — o **store**, que guarda alertas e incidentes.

Neste projeto há uma escolha deliberada: o store é **em memória** e começa sempre do mesmo seed. Parece pouco ambicioso, mas é o que torna possível a pergunta mais importante da unidade: *"a estratégia acertou?"*. Se o estado inicial é sempre o mesmo, dá para saber exatamente como ele **deveria** ficar depois de um pedido — e conferir.

Pense num simulador de voo. Ninguém treina pouso de emergência num avião de verdade: usa-se um cenário controlado, que começa igual toda vez, e mede-se o resultado. O `InMemoryOpsStore` + seed é o simulador do OpsPilot.

## Como está sendo usado

### O seed: 6 serviços, 6 alertas, 3 firing + 3 resolved

O seed não vem de um JSON externo — é um par de constantes TypeScript, validadas pelos próprios schemas Zod do domínio ao entrar no store:

```typescript
// src/models/in-memory-store.ts
export const DEFAULT_SERVICES: Service[] = [
  { id: "svc-api", name: "api", team: "platform", description: "Public API" },
  { id: "svc-checkout", name: "checkout", team: "commerce", description: "Checkout service" },
  { id: "svc-payments", name: "payments", team: "finance", description: "Payment processing" },
  { id: "svc-identity", name: "identity", team: "security", description: "Identity service" },
  { id: "svc-notifications", name: "notifications", team: "messaging", description: "Notification delivery" },
  // Sem alertas: existe para o cenário C2 do bench ("abra incidentes para checkout, payment e catalog").
  { id: "svc-catalog", name: "catalog", team: "commerce", description: "Product catalog" },
];

export const DEFAULT_ALERTS: Alert[] = [
  { id: "alert-api-1", serviceId: "svc-api", title: "API error rate elevated", severity: "high", status: "firing", createdAt: fixedDate("2026-09-28T08:00:00.000Z"), resolvedAt: null },
  { id: "alert-checkout-1", serviceId: "svc-checkout", title: "Checkout latency elevated", severity: "medium", status: "firing", createdAt: fixedDate("2026-09-28T08:10:00.000Z"), resolvedAt: null },
  { id: "alert-payments-1", serviceId: "svc-payments", title: "Payment failures detected", severity: "critical", status: "firing", createdAt: fixedDate("2026-09-28T08:20:00.000Z"), resolvedAt: null },
  // ... mais 3 alertas resolved (api, identity, notifications)
];
```

Guarde estes números: **3 firing, 1 deles critical** (`svc-payments`) — o bench usa exatamente esse gabarito no cenário C1. Ao reconstruir o estado (`resetSync`), cada item passa pelo schema Zod correspondente:

```typescript
// src/models/in-memory-store.ts
private resetSync(): void {
  this.services = DEFAULT_SERVICES.map((service) => ServiceSchema.parse(service));
  this.alerts = DEFAULT_ALERTS.map((alert) => AlertSchema.parse(alert));
  this.incidents = [];
  this.incidentSequence = 0;
}
```

É o mesmo princípio de [`zod-schemas-e-tipos.md`](../../../modulo03-mcp-na-pratica/06-your-legacy-api-as-mcp/customers-mcp/docs/zod-schemas-e-tipos.md) aplicado ao seed: se alguém editar `DEFAULT_ALERTS` com uma severidade inválida, o erro aparece na primeira instância do store (inclusive em qualquer teste), não no meio de uma execução do agente. O `src/services/seed.ts` é outra coisa — um script que zera o `JsonOpsStore` persistido em `src/data/ops-store.json` e imprime um resumo (`npm run seed`); quem a arena e o bench usam para isolar execuções é sempre um `InMemoryOpsStore` novo.

### O store: métodos assíncronos, erros de domínio nas transições inválidas

```typescript
// src/models/in-memory-store.ts
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
```

Todo método da interface `OpsStore` é **assíncrono** (`Promise<...>`), mesmo o `InMemoryOpsStore` não tendo I/O real — isso deixa a mesma interface servir também o `JsonOpsStore` (persistido em arquivo) e o `SequelizeOpsStore` (MySQL), sem o código que consome o store (tools, estratégias, bench) precisar saber qual implementação está por trás. `resolveIncident` nunca sobrescreve um incidente já resolvido: a segunda chamada lança `InvalidTransitionError`, uma transição de estado inválida — não um "resolve de novo" silencioso.

### As tools: `tool()` + Zod + store injetado

Você já embrulhou uma função como tool do LangChain em [`google-trends-tool.md`](../../../modulo03-mcp-na-pratica/02-google-trends-agent/docs/google-trends-tool.md). A novidade aqui é a **injeção do store** por uma factory, e a validação Zod tanto na entrada quanto na saída:

```typescript
// src/services/tools.ts
export const OpenIncidentInputSchema = z.object({
  title: z.string().trim().min(1),
  service: z.string().trim().min(1),
  severity: SeveritySchema,
});
export const OpenIncidentOutputSchema = IncidentSchema;

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
      // A descrição é o que o modelo lê para preencher os argumentos. Pedidos de plantão falam
      // "sev2", mas o domínio só conhece low/medium/high/critical — o mapeamento precisa estar aqui.
      description: "Open an incident for an existing service (use the exact service name) with a title and severity. Severity must be critical, high, medium or low; map sev1=critical, sev2=high, sev3=medium, sev4=low.",
      schema: OpenIncidentInputSchema,
    },
  );
  // ... resolveIncident e listIncidents seguem o mesmo padrão
  return [listAlerts, openIncident, resolveIncident, listIncidents] as const;
}
```

Três detalhes que valem estudo:

1. **`createOpsTools(store)`** — as tools não importam um store global; recebem um. Assim a arena e o bench criam **um store novo por execução** (`options?.store ?? defaultOpsStore`) e as estratégias não contaminam umas às outras.
2. **A `description` é prompt.** O modelo nunca lê o código da tool — só nome, descrição e schema. A frase `sev1=critical, sev2=high...` está ali porque o usuário fala "sev2" e o schema só aceita `high`. Sem esse mapeamento na descrição, o modelo chuta.
3. **A saída também passa por Zod.** `executeTool` serializa o resultado com `JSON.stringify(schema.parse(await operation()))` — se o store devolvesse um formato inesperado, o erro apareceria aqui, na fronteira da tool, não espalhado pela conversa com o LLM.

### Erro de domínio vira erro de fronteira para o modelo

```typescript
// src/services/tools.ts
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
```

O store **lança** um erro de domínio (`ResourceNotFoundError`, `InvalidTransitionError`, ...); a tool **captura e reembrulha** como `ToolBoundaryError`, preservando o `code` original e a causa. Diferente de simplesmente devolver uma string de erro para o LLM continuar a conversa, aqui o erro continua sendo um `throw` — é a estratégia (ReAct via LangGraph, ou o executor do Plan-and-Execute) quem decide o que fazer com ele. O teste `tools.test.ts` confirma que um serviço inexistente rejeita com `ToolBoundaryError` e **sem mutação parcial**: `(await store.readSummary()).incidents` continua `0` depois da falha.

### Mensagem de erro que ensina o nome certo

```typescript
// src/models/in-memory-store.ts
const service = this.services.find((item) => item.name === input.service);
if (!service) {
  // A lista de nomes válidos vira observação para o agente: o modelo que pediu "payment"
  // lê a mensagem e consegue tentar de novo com "payments".
  throw new ResourceNotFoundError(
    `Service ${input.service} was not found. Known services: ${this.services.map((item) => item.name).join(", ")}.`,
  );
}
```

É a mesma ideia de "traduzir o erro na borda" que você viu em [`erros-de-dominio-e-tratamento.md`](../../../modulo03-mcp-na-pratica/07-api-security-auth-rate-limiting/customers-mcp/docs/erros-de-dominio-e-tratamento.md) — só que aqui a borda é a conversa com o LLM, não uma resposta HTTP: a mensagem de erro lista os nomes válidos para que o próprio agente consiga se corrigir numa tentativa seguinte.

### Default tolerante no schema de entrada

```typescript
// src/services/tools.ts
export const ListAlertsInputSchema = z.object({
  status: AlertFilterSchema.default("firing"),
});
```

Se o modelo chamar `list_alerts({})` sem `status`, o `.default("firing")` do Zod preenche o valor **antes** da validação do enum (`AlertFilterSchema = z.enum(["firing", "resolved", "all"])`).

## Testes sem rede

Como o store é puro TypeScript, ele se testa sem chave, sem internet e sem mock de LLM:

```typescript
// src/models/in-memory-store.test.ts
test("default seed has six services (catalog without alerts) and a 3/3 alert status split", async () => {
  const store = new InMemoryOpsStore();
  assert.deepEqual(await store.readSummary(), {
    services: 6,
    alerts: 6,
    firing: 3,
    resolved: 3,
    incidents: 0,
  });
  assert.equal((await store.listAlerts("firing")).length, 3);
});

test("incident lifecycle is atomic, validates service, and prevents repeated resolution", async () => {
  const store = new InMemoryOpsStore();
  await assert.rejects(
    store.createIncident({ title: "Test", service: "payment", severity: "high" }),
    /Service payment was not found\. Known services: api, checkout, payments, identity, notifications, catalog\./,
  );
  const incident = await store.createIncident({ title: "Payment errors", service: "payments", severity: "critical" });
  assert.equal((await store.resolveIncident(incident.id)).status, "resolved");
  await assert.rejects(store.resolveIncident(incident.id), /already resolved/);
  await assert.rejects(store.resolveIncident("missing"), /was not found/);
});
```

Isso é a User Story 4 da spec 001 ("Validar o núcleo sem rede") em sua forma mais simples: nenhum teste acima chama rede, chave de API ou modelo — o `InMemoryOpsStore` é só TypeScript, e `assert.rejects` confere tanto o tipo do erro quanto a mensagem exata que o agente leria como observação.

## Decisões de domínio e de erro

Algumas decisões de design valem registro porque não são óbvias de antemão:

| Ponto | Decisão adotada |
| --- | --- |
| Identidade do serviço | `Alert`/`Incident` guardam `serviceId` (não o nome); as tools recebem o **nome** do usuário e o store resolve nome → id, rejeitando nomes desconhecidos com a lista de nomes válidos na mensagem |
| Categorias de erro | `OpsPilotError` com `code` em `VALIDATION`, `NOT_FOUND`, `INVALID_TRANSITION`, `CONFIGURATION`, `ITERATION_LIMIT` (`src/services/errors.ts`) |
| Resolver duas vezes | Erro de domínio (`InvalidTransitionError`): a transição só é `open -> resolved`, nunca `resolved -> resolved` |
| Validação das tools | Zod na entrada **e na saída** (`executeTool`); erro de domínio captado na fronteira vira `ToolBoundaryError`, preservando `code` e causa |
| Persistência | Três implementações da mesma interface `OpsStore`: `InMemoryOpsStore` (arena, bench, testes), `JsonOpsStore` (arquivo, usado pela CLI/tools por padrão) e `SequelizeOpsStore` (MySQL) |

A resolução de nome para `serviceId` é o ponto mais sutil: o prompt do usuário fala em **nomes** ("checkout", "payment"), e é assim que `createIncident({ service: "payments", ... })` é chamado pela tool — por dentro, o store procura `this.services.find((item) => item.name === input.service)` e, se não achar, lança `ResourceNotFoundError` listando os nomes conhecidos. Isso é o que permite ao agente se recuperar de "payment" → "payments" numa tentativa seguinte (veja o cenário C2 do bench em [arena-e-bench.md](./arena-e-bench.md)).

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Contrato das tools (nosso) | [`specs/.../contracts/tools.md`](../specs/001-nucleo-raciocinio-opspilot/contracts/tools.md) | Schemas de entrada/saída e erros |
| Modelo de dados (nosso) | [`specs/.../data-model.md`](../specs/001-nucleo-raciocinio-opspilot/data-model.md) | Transições `open -> resolved` e regras de validação |
| Tarefas relacionadas | [`specs/.../tasks.md`](../specs/001-nucleo-raciocinio-opspilot/tasks.md) | T004–T008, T016–T020 |
| Seed (nosso) | [`src/services/seed.ts`](../src/services/seed.ts) | Script que zera o `JsonOpsStore` e imprime o resumo (`npm run seed`) |
| Store (nosso) | [`src/models/in-memory-store.ts`](../src/models/in-memory-store.ts) | `DEFAULT_SERVICES`/`DEFAULT_ALERTS`, resolução de nome → `serviceId`, erro de domínio |
| Tools (nosso) | [`src/services/tools.ts`](../src/services/tools.ts) | Factory com store injetado, `description` como prompt, `executeTool` e `ToolBoundaryError` |
