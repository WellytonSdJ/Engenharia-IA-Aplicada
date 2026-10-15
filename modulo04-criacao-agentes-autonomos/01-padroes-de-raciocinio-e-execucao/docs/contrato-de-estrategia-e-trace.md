# Contrato de Estratégia, Trace e Métricas

## O que é

Se você vai comparar três jeitos de raciocinar, eles precisam ter **a mesma tomada**. É como um carregador universal: não importa se o aparelho é ReAct, Plan-and-Execute ou uma versão "refletida" de qualquer um deles — todos entram pelo mesmo plugue (`run(input)`) e devolvem a mesma forma de resultado (`{ answer, trace, metrics }`).

Esse contrato tem três peças:

| Peça | Pergunta que responde |
| --- | --- |
| `answer` | O que o agente respondeu? |
| `trace` | **Como** ele chegou lá — passo a passo, com tipos |
| `metrics` | Quanto custou — chamadas de LLM e tempo |

No módulo 03 ([`observabilidade-agent-loop.md`](../../../modulo03-mcp-na-pratica/01-multiple-mcp-tools/docs/observabilidade-agent-loop.md)) você enxergava o loop do agente com **callbacks** que imprimiam no console enquanto ele rodava — útil para depurar, mas o que ia para a tela se perdia. Aqui o trace é **dado**: uma lista tipada que volta junto com a resposta, que dá para testar, comparar e imprimir do jeito que quiser.

## Como está sendo usado

### A interface

```typescript
// src/agents/trace.ts
export const TraceEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("thought"), content: z.string() }),
  z.object({
    type: z.literal("action"),
    content: z.string(),
    tool: z.string().min(1),
    args: z.unknown().refine((value) => value !== undefined, "Action arguments are required."),
  }),
  z.object({ type: z.literal("observation"), content: z.string() }),
  z.object({ type: z.literal("plan"), content: z.string(), steps: z.array(z.string()).max(8) }),
  z.object({ type: z.literal("critique"), content: z.string() }),
  z.object({ type: z.literal("answer"), content: z.string() }),
]);

export const ReasoningMetricsSchema = z.object({
  llmCalls: z.number().nonnegative(),
  latencyMs: z.number().nonnegative(),
  iterations: z.number().int().nonnegative().optional(),
});

export const ReasoningResultSchema = z.object({
  answer: z.string(),
  trace: z.array(TraceEventSchema),
  metrics: ReasoningMetricsSchema,
});

export type TraceEvent = z.infer<typeof TraceEventSchema>;
export type ReasoningMetrics = z.infer<typeof ReasoningMetricsSchema>;
export type ReasoningResult = z.infer<typeof ReasoningResultSchema>;
```

```typescript
// src/agents/strategy.ts
export const ReasoningOptionsSchema = z.object({
  maxIterations: z.number().int().positive().default(8),
});

export type ReasoningOptions = z.input<typeof ReasoningOptionsSchema> & {
  model?: BaseChatModel;
  store?: OpsStore;
};

export interface ReasoningStrategy {
  readonly name: string;
  run(input: string, options?: ReasoningOptions): Promise<ReasoningResult>;
}
```

Repare como a interface é **mínima**: um nome e um método. Tudo que é específico de cada estratégia (modelo, tools, limite de iterações) entra por `ReasoningOptions`, um parâmetro opcional do próprio `run` — não por um construtor fixo. Isso é o que permite a arena e o bench tratarem todas as estratégias iguais e a Reflection embrulhar qualquer uma. `TraceEvent` é uma **união discriminada**: o TypeScript e o Zod proíbem, por exemplo, um evento `action` sem `tool` e `args`, ou um `plan` com mais de oito passos — algo que uma interface única com campos opcionais só garantiria por disciplina.

### Os seis tipos de evento

Cada tipo representa um momento diferente do raciocínio, e cada estratégia usa um subconjunto:

| Tipo | Significado | Quem emite |
| --- | --- | --- |
| `thought` | Texto que o modelo escreveu **antes** de chamar uma tool | ReAct |
| `action` | Chamada de tool, com `tool` e `args` | ReAct, Plan-and-Execute |
| `observation` | Resultado devolvido pela tool | ReAct, Plan-and-Execute |
| `plan` | O plano numerado produzido pelo planner | Plan-and-Execute |
| `critique` | Revisão: replanner ajustando o plano, ou o crítico da Reflection | Plan-and-Execute, Reflection |
| `answer` | Resposta final | Todas |

### Do histórico de mensagens para o trace

O ReAct do LangGraph não devolve um trace — devolve o **histórico de mensagens** (as mesmas `AIMessage`/`ToolMessage` que você viu em [`langchain-messages.md`](../../../modulo02-integracao-apis-llms/02-langchain-intro/docs/langchain-messages.md)). A função `traceFromMessages` traduz uma coisa na outra:

```typescript
// src/agents/react.ts
function traceFromMessages(messages: readonly BaseMessage[]): TraceEvent[] {
  const trace: TraceEvent[] = [];
  for (const message of messages) {
    if (message instanceof AIMessage) {
      const toolCalls = message.tool_calls ?? [];
      if (toolCalls.length > 0) {
        if (message.content) {
          trace.push({ type: "thought", content: messageText(message) });
        }
        for (const call of toolCalls) {
          trace.push({
            type: "action",
            content: `Calling ${call.name}`,
            tool: call.name,
            args: call.args,
          });
        }
      } else if (message.content) {
        trace.push({ type: "answer", content: messageText(message) });
      }
    } else if (message instanceof ToolMessage) {
      trace.push({ type: "observation", content: messageText(message) });
    }
  }
  return trace;
}
```

A regra de tradução, em uma tabela:

```
AIMessage COM tool_calls   → (texto? → thought) + uma action por tool_call
AIMessage SEM tool_calls   → answer
ToolMessage                → observation
HumanMessage / System      → ignorada
```

Diferente do que se poderia imaginar, essa função **não** garante por si só que o trace termine com `answer` — quem garante o invariante é a estratégia: `createReactStrategy` procura o último evento `answer` no trace produzido e, se não encontrar nenhum, lança `AgentConfigurationError("ReAct agent completed without a final answer.")` em vez de devolver um resultado incompleto. É a mesma ideia (todo resultado tem uma resposta), só que expressa como uma falha explícita em vez de um texto de fallback.

`messageText` existe porque `message.content` nem sempre é string — alguns provedores devolvem um array de partes. Ela serializa para texto (`JSON.stringify` quando não é string).

### Métricas: contar, não estimar

```typescript
// src/agents/react.ts — dentro de run()
const trace = traceFromMessages(messages);
const answer = [...trace].reverse().find((event) => event.type === "answer")?.content;
if (!answer) {
  throw new AgentConfigurationError("ReAct agent completed without a final answer.");
}
context.trace.push(...trace);
context.llmCalls = messages.filter((message) => message instanceof AIMessage).length;
context.iterations = Math.max(0, context.llmCalls - (answer ? 1 : 0));
if (context.iterations > maxIterations) {
  throw new IterationLimitError(maxIterations);
}
return context.result(answer);
```

`context` vem de `createExecutionContext` (`src/agents/strategy.ts`), que também mede `latencyMs` com `performance.now()` e empacota tudo em `context.result(answer)`, validado pelo `ReasoningResultSchema`. No ReAct, cada `AIMessage` no histórico é uma resposta do modelo, logo uma chamada. No Plan-and-Execute, cada nó do grafo (`planner`, `executor`, `replanner`, `finalizer`) chama `options.recordLlmCall()` explicitamente antes de invocar o modelo; na Reflection, as chamadas da estratégia base são **somadas** a uma por rodada de crítica. Em todos os casos o número vem do que aconteceu, não de uma estimativa a partir do texto.

## Por que o trace é tipado e não um log

Um log é texto para humano ler. Um trace tipado é **dado para código verificar**. Olhe o que um teste consegue afirmar:

```typescript
// src/agents/react.test.ts
const trace = traceFromMessages([
  new HumanMessage("List active alerts."),
  new AIMessage({
    content: "I should inspect the alert store.",
    tool_calls: [{ name: "list_alerts", args: { status: "firing" }, id: "call-1", type: "tool_call" }],
  }),
  new ToolMessage({ content: '[{"status":"firing"}]', tool_call_id: "call-1", name: "list_alerts" }),
  new AIMessage({ content: "One alert is firing." }),
]);
assert.deepEqual(trace.map(({ type }) => type), ["thought", "action", "observation", "answer"]);
assert.deepEqual(trace[1], {
  type: "action",
  content: "Calling list_alerts",
  tool: "list_alerts",
  args: { status: "firing" },
});
```

E as mensagens desse teste são **fabricadas à mão** (`new AIMessage({ content: ..., tool_calls: [...] })`) — nenhum LLM é chamado. O trace vira uma fronteira testável entre "o que o modelo fez" e "o que a gente mostra/mede". O próprio `src/agents/trace.test.ts` testa o schema isoladamente, com eventos também fabricados à mão (sem passar por mensagem alguma):

```typescript
// src/agents/trace.test.ts
test("trace parser rejects missing action fields, unknown types, and plans over eight steps", () => {
  assert.throws(() => parseTrace([{ type: "action", content: "missing fields" }]));
  assert.throws(() => parseTrace([{ type: "action", content: "missing args", tool: "list_alerts" }]));
  assert.throws(() => parseTrace([{ type: "hidden", content: "unknown" }]));
  assert.throws(() => parseTrace([{ type: "plan", content: "too long", steps: Array(9).fill("step") }]));
});
```

Em uma fase futura deste projeto esse mesmo trace deve ser persistido e exibido numa interface web — por isso vale caprichar no contrato agora.

## Decisões do contrato de estratégia e trace

O `contracts/strategy.md` e o `data-model.md` da spec 001 fixaram um conjunto de decisões deliberadamente rígidas, confirmadas pela implementação:

| Ponto | Decisão adotada |
| --- | --- |
| Tipo de `TraceEvent` | União discriminada (`z.discriminatedUnion`): `plan` exige `steps` (até 8), `action` exige `tool` e `args` — em vez de uma interface única com campos opcionais |
| Nome dos argumentos | `args` (não `toolArgs`) |
| Validação | Schemas Zod que **rejeitam** tipos desconhecidos, não só tipagem TypeScript (T009, FR-002) |
| `run` | `run(input, options?)` com `ReasoningOptions` (`maxIterations`, mais `model` e `store` injetáveis) |
| Métricas | `llmCalls`, `latencyMs` e um `iterations?` opcional |
| Arquivos | `src/agents/strategy.ts` + `src/agents/trace.ts` |

A união discriminada é uma melhoria real: com ela, o TypeScript e o Zod **proíbem** um `action` sem `tool`, algo que de outra forma só seria garantido por disciplina. O evento `critique` do schema (`src/agents/trace.ts`) só exige `type` e `content` — a camada de Reflection (veja [reflection.md](./reflection.md)) serializa seu veredito (`{ aproved, feedback }`) como JSON dentro desse `content`, em vez de adicionar campos extras ao evento.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Contrato (nosso) | [`specs/.../contracts/strategy.md`](../specs/001-nucleo-raciocinio-opspilot/contracts/strategy.md) | União discriminada de `TraceEvent` e `ReasoningOptions` |
| Modelo de dados (nosso) | [`specs/.../data-model.md`](../specs/001-nucleo-raciocinio-opspilot/data-model.md) | `TraceEvent`, `ReasoningMetrics`, `ReasoningResult` |
| Tarefas relacionadas | [`specs/.../tasks.md`](../specs/001-nucleo-raciocinio-opspilot/tasks.md) | T009, T010, T012, T013, T014 |
| Contrato de estratégia (nosso) | [`src/agents/strategy.ts`](../src/agents/strategy.ts) | `ReasoningStrategy` e `ReasoningOptions` |
| Builder do trace (nosso) | [`src/agents/trace.ts`](../src/agents/trace.ts) | Regra mensagem → evento e o `answer` garantido |
| Testes do trace (nosso) | [`src/agents/trace.test.ts`](../src/agents/trace.test.ts) | Mensagens fabricadas à mão, zero rede |
