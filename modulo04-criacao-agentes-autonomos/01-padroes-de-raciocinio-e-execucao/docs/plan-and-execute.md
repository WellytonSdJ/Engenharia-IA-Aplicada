# Plan-and-Execute

## O que é

Se o ReAct é dirigir olhando placa por placa, o Plan-and-Execute é **traçar a rota no GPS antes de sair** — e deixar o GPS recalcular quando você erra uma saída.

O padrão separa três papéis:

| Papel | O que faz | Chama tools? |
| --- | --- | --- |
| **Planner** | Lê o pedido e escreve uma lista ordenada de passos | Não |
| **Executor** | Executa **um** passo por vez, com as tools | Sim |
| **Replanner** | Depois de cada passo, olha o progresso e decide: continuar, ajustar o plano ou terminar | Não |

```
START → planner → executor → replanner ─┬─ finish (answer preenchida) → END
                     ▲                  ├─ limite de passos atingido  → END
                     └──── continue ────┴─ adjust (plano revisado)
```

A vantagem: tarefas com estrutura ("abra três, nessa ordem, e resolva o primeiro") ficam explícitas num plano que dá para **ler no trace** antes de qualquer ação. A desvantagem: mais chamadas de LLM (uma para planejar, uma por passo do executor, uma por revisão do replanner) — e o bench vai mostrar esse custo.

## Como está sendo usado (código de referência)

Este é o maior arquivo da unidade (`src/strategies/plan-execute.ts` no snapshot). Vamos por partes.

### O estado do grafo

Você conhece `Annotation.Root` e reducers de [`langgraph.md`](../../../modulo02-integracao-apis-llms/04-song-highlights/docs/langgraph.md). O estado aqui tem dois tipos de reducer, e a escolha de cada um é o coração do padrão:

```typescript
const PlanExecuteState = Annotation.Root({
  input: Annotation<string>(),
  plan: Annotation<string[]>({
    reducer: (_state, update) => update,        // SUBSTITUI: o plano restante é sempre o mais novo
    default: () => [],
  }),
  done: Annotation<[string, string][]>({
    reducer: (state, update) => state.concat(update),   // ACUMULA: histórico de [passo, resultado]
    default: () => [],
  }),
  answer: Annotation<string>({ reducer: (_state, update) => update, default: () => "" }),
  trace: Annotation<TraceEvent[]>({
    reducer: (state, update) => state.concat(update),   // ACUMULA: cada nó só devolve os eventos novos
    default: () => [],
  }),
  iterations: Annotation<number>({ reducer: (_state, update) => update, default: () => 0 }),
  llmCalls: Annotation<number>({ reducer: (_state, update) => update, default: () => 0 }),
});
```

- `plan` **substitui**: o executor tira o primeiro passo e devolve o resto; o replanner pode trocar o resto inteiro.
- `done` e `trace` **acumulam**: cada nó devolve só o que é novo, e o reducer concatena.

(Os comentários "SUBSTITUI"/"ACUMULA" foram adicionados aqui para estudo, e alguns campos foram compactados em uma linha; a lógica é idêntica à do original. Os demais trechos deste doc também podem ter quebras de linha ajustadas ou partes omitidas com `// ...`.)

### Planner: structured output com teto

```typescript
const MAX_STEPS = 8;

const planSchema = z.object({
  steps: z
    .array(z.string().min(1))
    .min(1)
    .max(MAX_STEPS)
    .describe("passos curtos, ordenados, executaveis com as ferramentas disponiveis"),
});
```

```typescript
const planner = async (state: typeof PlanExecuteState.State) => {
  let steps: string[];
  try {
    const plan = await plannerModel.invoke([
      ["system", [
        "Você é o planner operacional do OpsPilot.",
        `Produza no máximo ${stepLimit} passos curtos, ordenados e executáveis com as tools disponíveis.`,
        "Não invente ferramentas fora de list_alerts, open_incident, resolve_incident.",
        // ...
      ].join(" ")],
      ["user", state.input],
    ]);
    const parsed = planSchema.parse(plan);
    steps = parsed.steps.slice(0, stepLimit);
  } catch {
    steps = [`Execute o pedido com as tools disponíveis: ${state.input}`];
  }

  return {
    plan: steps,
    trace: [{ type: "plan" as const, content: formatPlan(steps) }],
    llmCalls: state.llmCalls + 1,
  };
};
```

O limite de 8 passos aparece **três vezes**: no schema (`.max(MAX_STEPS)`), no prompt ("no máximo N passos") e no código (`.slice(0, stepLimit)`). Isso não é redundância boba — é defesa em camadas. A nossa `research.md` diz o mesmo: *"o limite de 8 passos e o limite de iterações serão guards do estado, não apenas instruções de prompt"*. O prompt pede, o schema valida, o código garante.

E se o planner falhar (modelo devolve algo fora do schema)? O `catch` cria um plano de **um passo** que é basicamente "faça o pedido" — o executor vira um ReAct comum. Degrada, mas não quebra.

`plannerModel` é `this.modelFactory().withStructuredOutput(planSchema)` — o structured output que você estudou em [`structured-output.md`](../../../modulo02-integracao-apis-llms/03-medical-appointment/docs/structured-output.md), agora usado para produzir um **plano**.

### Executor: um mini-ReAct por passo

```typescript
const executor = async (state: typeof PlanExecuteState.State) => {
  if (state.plan.length === 0 || state.iterations >= stepLimit) {
    return {};
  }

  const [currentStep, ...remainingPlan] = state.plan;
  const agent = createReactAgent({ llm: this.modelFactory(), tools: this.tools });
  const result = await agent.invoke(
    { messages: [{ role: "user", content: buildExecutorUserMessage(state.input, state.done, currentStep) }] },
    { recursionLimit: Math.max(3, stepLimit * 3) },
  );
  // ...
  return {
    plan: remainingPlan,
    done: [[currentStep, stepResult]] as [string, string][],
    trace: stepTrace,
    iterations: state.iterations + 1,
    llmCalls: state.llmCalls + aiMessages,
  };
};
```

O executor é um **ReAct com escopo reduzido**: recebe um passo só. A mensagem que ele recebe é montada assim:

```typescript
function buildExecutorUserMessage(originalInput: string, done: [string, string][], currentStep: string): string {
  const sections = [
    `Pedido original (contexto — cumpra as restrições literais):\n${originalInput}`,
    "Regras: preserve nomes de serviço exatamente como no pedido; sev1=critical, sev2=high, sev3=medium, sev4=low.",
  ];
  if (done.length > 0) {
    sections.push(`Progresso anterior (use IDs/dados já obtidos):\n${doneAsText(done)}`);
  }
  sections.push(`Passo atual a executar agora:\n${currentStep}`);
  return sections.join("\n\n");
}
```

Três blocos: **pedido original** (para não perder restrições literais como "nessa mesma ordem"), **progresso anterior** (é aqui que o id do incidente criado no passo 1 chega ao passo 4, que vai resolvê-lo) e **passo atual**. Cada execução do executor começa com um histórico de mensagens **zerado** — a memória entre passos é só o `done`, em texto. É um exemplo pequeno de gerenciamento de contexto, tema de uma unidade inteira mais adiante no curso.

### Replanner: schema plano de propósito

```typescript
/** Flat schema — OpenRouter/models often break on zod discriminatedUnion. */
const replanSchema = z.object({
  decision: z.enum(["adjust", "continue", "finish"]),
  steps: z.array(z.string().min(1)).max(MAX_STEPS).optional().describe("obrigatório quando decision=adjust"),
  answer: z.string().min(1).optional().describe("obrigatório quando decision=finish"),
});
```

O "certo" em TypeScript seria uma união discriminada (`{decision:"adjust", steps} | {decision:"finish", answer} | ...`). Mas o comentário conta a lição prática: muitos modelos via OpenRouter **quebram** com `z.discriminatedUnion` no structured output. Solução: um objeto plano com campos opcionais e a regra ("obrigatório quando...") descrita no `.describe()` — ou seja, em prompt. O código depois trata os casos em que o modelo não obedeceu (por exemplo, `adjust` com lista vazia vira `finish`).

As três decisões e o que cada uma produz no trace:

| Decisão | Efeito no estado | Evento no trace |
| --- | --- | --- |
| `finish` | `answer` preenchida → grafo termina | `answer` |
| `adjust` | `plan` substituído pelos novos passos | `critique` com o plano revisado |
| `continue` | nada muda (ou termina se o plano acabou) | `critique` "Plano atual segue válido" |

### O modo sem replanner

A estratégia aceita `enableReplanner: false`. Aí o grafo vira um laço simples — o executor roda até o plano acabar e um nó `finish` monta a resposta a partir do `done`:

```typescript
new StateGraph(PlanExecuteState)
  .addNode("planner", planner)
  .addNode("executor", executor)
  .addNode("finish", finishWithoutReplanner)
  .addEdge(START, "planner")
  .addEdge("planner", "executor")
  .addConditionalEdges("executor", (state) => {
    if (state.plan.length === 0 || state.iterations >= stepLimit) {
      return "finish";
    }
    return "executor";
  })
  .addEdge("finish", END)
  .compile();
```

Isso existe para um **experimento**: o bench tem a flag `--no-replanner`, que permite medir quanto o replanner custa (chamadas) e quanto ele ajuda (acerto). Ter a opção dentro da própria estratégia é mais barato que manter uma segunda estratégia.

### Dois limites, dois níveis

```typescript
const stepLimit = Math.min(MAX_STEPS, this.maxIterations);
// ...
{ recursionLimit: Math.max(10, stepLimit * 5) }   // no graph.invoke externo
```

`stepLimit` limita **passos do plano** (lógica de negócio, checada nos nós e nas arestas). `recursionLimit` limita **super-steps do grafo** (proteção do LangGraph contra laço infinito). O primeiro é o freio que deveria atuar; o segundo é o airbag.

## ReAct × Plan-and-Execute

| | ReAct | Plan-and-Execute |
| --- | --- | --- |
| Decide o próximo passo | A cada volta, olhando o histórico | No plano; o replanner revisa |
| Chamadas de LLM | Menos (1 por volta) | Mais (planner + executor por passo + replanner por passo) |
| Tarefas curtas | Melhor | Overkill |
| Tarefas com ordem/estrutura | Pode se perder | Mais confiável |
| Trace | `thought/action/observation/answer` | `plan/action/observation/critique/answer` |
| Implementação | Prebuilt | `StateGraph` próprio (que usa o prebuilt dentro do executor) |

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Decisão do `StateGraph` | [`specs/.../research.md`](../specs/001-nucleo-raciocinio-opspilot/research.md) | Decision 3: limites como guards de estado |
| Requisitos | [`specs/.../spec.md`](../specs/001-nucleo-raciocinio-opspilot/spec.md) | FR-011, FR-012, SC-004 e os edge cases do planner/replanner |
| Tarefas relacionadas | [`specs/.../tasks.md`](../specs/001-nucleo-raciocinio-opspilot/tasks.md) | T022, T023, T025 |
| Estratégia (referência) | `src/strategies/plan-execute.ts` do snapshot | Reducers, schemas, os três nós e as duas topologias |
| Trace específico (referência) | `src/trace/builder.ts` do snapshot | `buildPlanExecuteTrace` e os testes de replanning com `critique` |
