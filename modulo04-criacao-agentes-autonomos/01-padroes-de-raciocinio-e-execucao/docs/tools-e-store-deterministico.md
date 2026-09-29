# Tools e Store Determinístico

## O que é

Um agente sem ferramentas só conversa. As tools são as **mãos** do OpsPilot: consultar alertas, abrir incidente, resolver incidente. E as mãos precisam mexer em alguma coisa — o **store**, que guarda alertas e incidentes.

Neste projeto há uma escolha deliberada: o store é **em memória** e começa sempre do mesmo seed. Parece pouco ambicioso, mas é o que torna possível a pergunta mais importante da unidade: *"a estratégia acertou?"*. Se o estado inicial é sempre o mesmo, dá para saber exatamente como ele **deveria** ficar depois de um pedido — e conferir.

Pense num simulador de voo. Ninguém treina pouso de emergência num avião de verdade: usa-se um cenário controlado, que começa igual toda vez, e mede-se o resultado. O `InMemoryStore` + seed é o simulador do OpsPilot.

## Como está sendo usado (código de referência)

### O seed: 5 serviços, 6 alertas, 3 + 3

```json
// src/store/seed-data.json (referência) — trecho
{ "id": "alert-001", "service": "payment-api", "description": "High error rate on /checkout",
  "severity": "critical", "status": "firing" },
{ "id": "alert-002", "service": "auth-service", "description": "JWT validation failures spike",
  "severity": "high", "status": "firing" },
{ "id": "alert-003", "service": "order-service", "description": "DB connection pool exhausted",
  "severity": "critical", "status": "firing" },
```

Os outros três (`inventory-api`, `notification-worker`, `payment-api` de novo) estão `resolved`. Guarde estes números: **3 firing, 2 deles critical**. O bench vai usá-los como gabarito.

O JSON não entra no store "no escuro" — passa por Zod primeiro:

```typescript
// src/store/seed.ts (referência)
const seedSchema = z.object({
  services: z.array(serviceSchema).min(1),
  alerts: z.array(alertSchema).min(1),
});

const parsedSeed = seedSchema.parse(seedData);
```

É o mesmo princípio de [`zod-schemas-e-tipos.md`](../../../modulo03-mcp-na-pratica/06-your-legacy-api-as-mcp/customers-mcp/docs/zod-schemas-e-tipos.md) aplicado a um arquivo de dados: se alguém editar o JSON com uma severidade inválida, o erro aparece na carga, não no meio de uma execução do agente.

### O store: cópias para fora, nunca referências

```typescript
// src/store/in-memory-store.ts (referência)
getAlerts(status?: AlertStatus): Alert[] {
  const source = status ? this.alerts.filter((alert) => alert.status === status) : this.alerts;
  return source.map((alert) => ({ ...alert }));
}

resolveIncident(id: string): Incident {
  const incident = this.incidents.find((item) => item.id === id);
  if (!incident) {
    throw new IncidentNotFoundError(id);
  }

  incident.status = "resolved";
  incident.resolvedAt = Date.now();
  return { ...incident };
}
```

Todo método devolve `{ ...objeto }` — uma cópia rasa. Se quem recebeu o alerta alterar o objeto, o store não é afetado. Só o próprio store muda o próprio estado. É um jeito barato de ter imutabilidade na fronteira, alinhado com o "prefira funções puras" da nossa constituição.

### As tools: `tool()` + Zod + store injetado

Você já embrulhou uma função como tool do LangChain em [`google-trends-tool.md`](../../../modulo03-mcp-na-pratica/02-google-trends-agent/docs/google-trends-tool.md). A novidade aqui é a **injeção do store** por uma factory:

```typescript
// src/agents/tools.ts (referência)
const openIncidentSchema = z.object({
  title: z.string().min(1),
  service: z.string().min(1),
  severity: z.enum(["critical", "high", "medium", "low"]),
});

export function createOpenIncidentTool(store: IStore): DynamicStructuredTool<typeof openIncidentSchema> {
  return tool(
    async ({ title, service, severity }) => {
      const incident = store.createIncident({ title, service, severity });
      return [
        `Incident created successfully. ID: ${incident.id}`,
        `Title: ${incident.title}`,
        `Service: ${incident.service}`,
        `Severity: ${incident.severity}`,
        `Status: ${incident.status}`,
      ].join("\n");
    },
    {
      name: "open_incident",
      description:
        "Open a new incident for a service. severity must be critical|high|medium|low (sev1=critical, sev2=high, sev3=medium, sev4=low). Use exact service names from the user request.",
      schema: openIncidentSchema,
    },
  );
}
```

Três detalhes que valem estudo:

1. **`createOpenIncidentTool(store)`** — a tool não importa um store global; recebe um. Assim a arena e o bench criam **um store novo por execução** e as estratégias não contaminam umas às outras.
2. **A `description` é prompt.** O modelo nunca lê o código da tool — só nome, descrição e schema. A frase `sev1=critical, sev2=high...` está ali porque o usuário fala "sev2" e o schema só aceita `high`. Sem esse mapeamento na descrição, o modelo chuta.
3. **A saída é texto com o ID.** O `ID: inc-...` na resposta é o que permite, num passo seguinte, o modelo chamar `resolve_incident` com o id certo.

### Erro de domínio vira texto para o modelo

```typescript
// src/agents/tools.ts (referência) — resolve_incident
try {
  const incident = store.resolveIncident(id);
  // ...
} catch (error) {
  if (error instanceof IncidentNotFoundError) {
    return `Error: ${error.message}`;
  }
  throw error;
}
```

O store **lança** `IncidentNotFoundError`; a tool **captura e devolve como texto**. Por quê? Porque para o agente um id errado não é uma falha do sistema — é uma **observação** ("esse id não existe") da qual ele pode se recuperar, por exemplo listando de novo. Já um erro desconhecido continua sendo lançado. É a mesma ideia de "traduzir o erro na borda" que você viu em [`erros-de-dominio-e-tratamento.md`](../../../modulo03-mcp-na-pratica/07-api-security-auth-rate-limiting/customers-mcp/docs/erros-de-dominio-e-tratamento.md) — só que aqui a borda é a conversa com o LLM, não uma resposta HTTP.

### `z.preprocess` para default tolerante

```typescript
const listAlertsSchema = z.object({
  status: z.preprocess((value) => value ?? "firing", z.enum(["firing", "resolved", "all"])),
});
```

Se o modelo chamar `list_alerts({})` sem status, o `preprocess` troca `undefined` por `"firing"` **antes** da validação do enum. É mais robusto que `.default()` diante de modelos que mandam `null` em vez de omitir o campo.

## Testes sem rede

Como o store é puro TypeScript, ele se testa sem chave, sem internet e sem mock de LLM:

```typescript
// src/store/in-memory-store.test.ts (referência)
test("getAlerts('firing') returns exactly 3", () => {
  const store = new InMemoryStore();
  seedStore(store);
  assert.equal(store.getAlerts("firing").length, 3);
});

test("resolveIncident throws IncidentNotFoundError for unknown id", () => {
  const store = new InMemoryStore();
  seedStore(store);
  assert.throws(() => store.resolveIncident("inc-unknown"), IncidentNotFoundError);
});
```

Isso é a User Story 4 da nossa spec ("Validar o núcleo sem rede") em sua forma mais simples.

## Onde o nosso plano diverge da referência

| Ponto | Referência | Nosso plano |
| --- | --- | --- |
| Nome da classe | `InMemoryStore` | `InMemoryOpsStore` (`src/models/in-memory-store.ts`) |
| Seed | JSON + `seed.ts` que também roda como script | `src/services/seed.ts` **idempotente**, com resumo impresso e script `npm run seed` (T019, T020) |
| Alerta | `service` (nome) + `description` | `serviceId` + `title` + `createdAt`/`resolvedAt`; `Service` com `team` e `description` |
| Validação das tools | Zod só na entrada | Zod na entrada **e na saída**; serviço precisa existir; sem mutação parcial em falha (T016, T018) |
| Erros | Só `IncidentNotFoundError` | Categorias: validação, recurso ausente, transição inválida, configuração, limite de iterações (T005) |
| Resolver duas vezes | Permitido (sobrescreve `resolvedAt`) | Erro de domínio (`open -> resolved` apenas) |
| Persistência | Só memória | + adapter Sequelize/MySQL (T008) |

A mudança de `service` para `serviceId` merece atenção: o prompt do usuário fala em **nomes** ("checkout", "payment-api"), e o bench da referência compara `incident.service` com o nome. Se o nosso modelo usar `serviceId`, a tool vai precisar resolver nome → id (e decidir o que fazer com um serviço que não existe no seed — "checkout" não está nos 5 serviços!). Vale decidir isso antes do `implement`; está listado em [roteiro-de-implementacao.md](./roteiro-de-implementacao.md).

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Contrato das tools (nosso) | [`specs/.../contracts/tools.md`](../specs/001-nucleo-raciocinio-opspilot/contracts/tools.md) | Schemas de entrada/saída e erros |
| Modelo de dados (nosso) | [`specs/.../data-model.md`](../specs/001-nucleo-raciocinio-opspilot/data-model.md) | Transições `open -> resolved` e regras de validação |
| Tarefas relacionadas | [`specs/.../tasks.md`](../specs/001-nucleo-raciocinio-opspilot/tasks.md) | T004–T008, T016–T020 |
| Seed (referência) | `src/store/seed-data.json` e `src/store/seed.ts` do snapshot | Números 5/6/3/3 e validação Zod do JSON |
| Store (referência) | `src/store/in-memory-store.ts` do snapshot | Cópias rasas e erro de domínio |
| Tools (referência) | `src/agents/tools.ts` do snapshot | Factory com store injetado, `description` como prompt, erro → texto |
