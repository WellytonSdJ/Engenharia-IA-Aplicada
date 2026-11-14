# Function calling e design de tools — o schema é o prompt

> Este doc cobre a parte de **design de tools** da spec 005 ([`005-persistencia-sqlite`](../specs/005-persistencia-sqlite/spec.md)) e as tools novas `list_incidents` e `consultar_runbook`. Antes, leia [01 / 03-tools-e-store-deterministico.md](../../01-padroes-de-raciocinio-e-execucao/docs/03-tools-e-store-deterministico.md): o básico de `tool()` + Zod + store injetado está lá e não é repetido aqui.
>
> **Em 30 segundos**
>
> - **Function calling**: o modelo não executa nada. Ele lê o *cardápio* de tools (nome + descrição + schema) e devolve "quero chamar X com estes argumentos". Quem executa é o nosso código.
> - Por isso **a descrição e o schema são o prompt** da tool: é só isso que o modelo vê para decidir *se*, *quando* e *como* chamar.
> - O projeto adota **6 regras** para escrever toda descrição e todo schema.
> - Uma única lista de definições (`createOpsToolDefinitions`) alimenta o agente (LangChain) **e** o servidor MCP: uma fonte de verdade só.

---

## O que é function calling

Pense num garçom (o modelo). Ele não cozinha: lê o cardápio, entende o pedido do cliente e entrega um papelzinho para a cozinha: *"mesa 4, um risoto, sem queijo"*. A cozinha (nosso código) prepara e devolve o prato. O garçom leva ao cliente e continua a conversa.

```text
 usuário ──"tem algo pegando fogo?"──▶ modelo
                                         │  lê o cardápio de tools
                                         ▼
                        tool_call: list_alerts({ status: "firing" })
                                         │
                       nosso código ◀────┘  executa de verdade (store, API...)
                                         │
                        observação: [ {...alerta checkout...}, ... ]
                                         ▼
                                       modelo ──▶ "Sim: 3 alertas ativos, o pior é o checkout"
```

| Quem | Faz o quê |
| --- | --- |
| **Modelo** | Escolhe a tool e preenche os argumentos (JSON) a partir do schema |
| **Nosso código** | Valida os argumentos, executa e devolve o resultado como texto |
| **Loop do agente** ([ReAct](../../01-padroes-de-raciocinio-e-execucao/docs/04-react.md)) | Repete até o modelo responder sem pedir tool |

> ✅ **Checkpoint**: o modelo **pede**, o código **executa**. Se o pedido vier errado, a culpa costuma ser do cardápio.

---

## "O schema é o prompt"

Se o garçom só tem o cardápio para decidir, um cardápio ruim gera pedidos ruins. Com tools é igual: o modelo **nunca** vê o seu código, só isto:

```text
name:        open_incident
description: "Use quando o usuário pedir explicitamente para abrir..."
schema:      { title: string (describe...), service: string (describe...), severity: enum (describe...) }
```

Cada palavra da descrição e de cada `.describe()` está, na prática, **dentro do prompt**. Uma descrição vaga custa tool errada, argumento inventado ou tool chamada sem necessidade.

---

## As 6 regras de descrição de tools

Elas foram definidas na pesquisa da spec 005 ([`research.md`](../specs/005-persistencia-sqlite/research.md)) e valem para **todas** as tools:

| # | Regra | Por que importa |
| --- | --- | --- |
| 1 | A descrição diz **quando** usar (gatilho de intenção), não só o que faz | O modelo decide *se* chama pela intenção do usuário |
| 2 | O vocabulário ambíguo do usuário é **mapeado** para o valor aceito (ex.: `sev2` → `high`) | As pessoas falam "sev2"; o domínio fala `high` |
| 3 | Todo campo Zod tem `.describe()` | É o único "manual" que o modelo tem para preencher o argumento |
| 4 | Os enums aparecem **por extenso** na descrição | O modelo não precisa adivinhar os valores válidos |
| 5 | A descrição **não promete** o que a tool não garante | Evita o modelo contar ao usuário um efeito que não aconteceu |
| 6 | Os erros previsíveis são **antecipados** (não encontrado, já resolvido) | O modelo se recupera sem tentar de novo às cegas |

💡 Outra forma comum de montar o mesmo checklist: **nome** estável em `snake_case`; **o quê**; **quando usar**; **quando NÃO usar**; `.describe()` em todo campo; domínio fechado com `z.enum`. As duas versões chegam ao mesmo lugar: dizer ao modelo *quando* e *como*, e fechar o que puder ser fechado.

### As regras aplicadas: `open_incident`

```typescript
// src/services/tools.ts
export const OpenIncidentInputSchema = z.object({
  title: z.string().trim().min(1).describe("Título curto e descritivo do incidente, não vazio."),
  service: z.string().trim().min(1).describe(
    "Nome exato do serviço afetado (ex.: 'checkout', 'payments'); use o nome como aparece na lista de serviços conhecida, não um nome aproximado.",
  ),
  severity: SeveritySchema.describe(
    "Severidade do incidente: critical, high, medium ou low. Se o usuário falar em sevN (plantão/ITIL), mapeie sev1=critical, sev2=high, sev3=medium, sev4=low antes de preencher este campo.",
  ),
});
```

```typescript
// src/services/tools.ts (descrição da tool)
description: "Use quando o usuário pedir explicitamente para abrir/registrar um incidente para um serviço — não para apenas relatar um alerta já existente sem pedido de abertura. Severidade: critical, high, medium ou low; mapeie sev1=critical, sev2=high, sev3=medium, sev4=low. Cria um incidente no store. Serviço inexistente é rejeitado sem alterar o estado, e a mensagem de erro lista os serviços conhecidos para tentar de novo com o nome certo.",
```

Releia procurando as regras: **quando** (1), **quando não** (bônus), `sev2=high` (2), enum por extenso (4), "rejeitado sem alterar o estado" (5 e 6).

> ✅ **Checkpoint**: uma boa descrição responde *quando usar*, *com quais valores* e *o que acontece se der errado*.

---

## As tools novas desta unidade

| Tool | Quando o modelo deve usar | Altera dados? |
| --- | --- | --- |
| `list_incidents` | Ver o que já está aberto **antes** de abrir outro incidente (evita duplicar) | não |
| `consultar_runbook` | O usuário pede o "como investigo/mitigo" de um serviço | não |
| `check_provider_status` | Suspeita de que o problema é do GitHub/Cloudflare, e não nosso | não (consulta externa: veja [03-tool-externa-resiliente.md](./03-tool-externa-resiliente.md)) |

Repare que `consultar_runbook` separa **"serviço sem runbook"** (resultado normal) de **"serviço inexistente"** (erro). Para o modelo, são situações diferentes:

```typescript
// src/services/tools.ts
export const ConsultarRunbookOutputSchema = z.union([
  RunbookSchema.pick({ title: true, content: true }).extend({ service: z.string() }),
  z.object({ service: z.string(), found: z.literal(false) }),
]);
```

`found: false` diz ao modelo: "o serviço existe, só não tem runbook; não insista". O serviço inexistente volta como erro, com a lista de nomes válidos. É o mesmo truque explicado no [01](../../01-padroes-de-raciocinio-e-execucao/docs/03-tools-e-store-deterministico.md).

---

## Uma fonte de verdade para dois consumidores

Até o projeto 01, as tools nasciam direto como `tool()` do LangChain. Agora elas nascem como **definições neutras**:

```typescript
// src/services/tools.ts
export interface OpsToolDefinition<TSchema extends z.ZodType = z.ZodType> {
  name: string;
  description: string;
  inputSchema: TSchema;
  run: (input: z.infer<TSchema>) => Promise<string>;
}
```

```text
                createOpsToolDefinitions(store)
                 (nome + descrição + schema + run)
                    │                       │
       toLangChainTool(...)          server.registerTool(...)
                    ▼                       ▼
         agente ReAct / Plan       servidor MCP (Copilot, IDEs)
```

Mudou a descrição de `list_alerts`? Ela muda para o agente **e** para o MCP ao mesmo tempo, e nenhum dos dois fica com um cardápio desatualizado. O lado MCP está em [05-servidor-mcp.md](./05-servidor-mcp.md).

> ✅ **Checkpoint**: definir a tool uma vez e **adaptá-la** para cada consumidor evita dois cardápios divergentes.

---

## ⚠️ Pegadinhas

- **Descrever o "o quê" e esquecer o "quando".** "Lista incidentes" não ajuda o modelo a decidir. "Use para ver o que já está aberto antes de abrir um novo" ajuda.
- **`z.string()` onde cabe `z.enum`.** Um campo aberto convida o modelo a inventar valores (`"urgent"`, `"P1"`). Domínio fechado pede enum.
- **Overclaim.** Se a descrição diz "notifica o time" e a tool não notifica, o modelo vai dizer ao usuário que notificou.
- **Descrição gigante.** Ela entra em **toda** chamada ao modelo. Seja completo, mas sem escrever um romance.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| As 6 regras | [`specs/005-persistencia-sqlite/research.md`](../specs/005-persistencia-sqlite/research.md) | A decisão e o porquê de cada regra |
| Descrições e schemas | [`src/services/tools.ts`](../src/services/tools.ts) | `.describe()` em todo campo, mapeamento `sevN` |
| Fonte única | [`src/services/tools.ts`](../src/services/tools.ts) | `OpsToolDefinition`, `createOpsToolDefinitions`, `toLangChainTool` |
| Tools novas | [`src/services/tools.ts`](../src/services/tools.ts) | `list_incidents`, `consultar_runbook` (`found: false`) |
| Testes | [`src/services/tools.test.ts`](../src/services/tools.test.ts) | Tools rodando sobre SQLite `:memory:` |

---

⬅️ Anterior: [00 · Por onde começar](./00-START-HERE.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [02 · Persistência com SQLite nativo](./02-persistencia-sqlite.md)**
