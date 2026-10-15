# Reflection — a camada de autocrítica

> Este doc cobre a **spec 002** (`002-camada-reflection`) deste projeto. A spec [`002-camada-reflection`](../specs/002-camada-reflection/spec.md) está implementada em [`src/agents/reflection.ts`](../src/agents/reflection.ts) (`withReflection(strategy, opts)`, crítica `{ aproved, feedback }`, até 2 regenerações por padrão), e a arena ganhou `reflect:react` e `reflect:plan-and-execute`. O bench ainda compara só as duas estratégias base.

## O que é

Reflection é o padrão em que **uma segunda passada de LLM avalia a resposta da primeira** e, se não gostar, manda refazer com feedback. É o revisor de texto: você escreve, o revisor devolve com anotações ("faltou dizer quantos sobraram"), você reescreve.

A decisão de design mais interessante da unidade é **onde** a Reflection mora. Ela não é uma terceira estratégia ao lado de ReAct e Plan-and-Execute — é um **decorator**: uma função que recebe *qualquer* `ReasoningStrategy` e devolve *outra* `ReasoningStrategy`, com a crítica embutida.

```
withReflection(reactStrategy)            → name: "reflect:react"
withReflection(planAndExecuteStrategy)   → name: "reflect:plan-and-execute"
```

Isso só é possível porque o contrato comum (veja [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md)) é mínimo: um `name` e um `run(input, options?)`. O decorator implementa a mesma interface e, por dentro, chama a estratégia original quantas vezes precisar. Quem usa não percebe a diferença — a arena trata `reflect:react` exatamente como trata `react`.

```
run(input)
  └─ base.run(input) ──────────────────────────► resposta 1
       └─ crítico(resposta 1, trace, input) → { aproved: false, feedback: "faltou X" }
            └─ base.run("input\n\nCritic feedback for revising the previous answer:\nfaltou X") ► resposta 2
                 └─ crítico(resposta 2, ...) → { aproved: true }  → devolve resposta 2
```

## Como está sendo usado

### O crítico: saída estruturada e prompt ancorado no trace

```typescript
// src/agents/reflection.ts
export const ReflectionCritiqueSchema = z.object({
  aproved: z.boolean(),
  feedback: z.string().describe(
    "If rejected, provide specific and actionable corrections. If approved, briefly explain why the answer is supported.",
  ),
});

const CRITIC_PROMPT = [
  "You are a strict answer critic.",
  "Evaluate the answer ONLY against the original request and the observations supplied in the trace.",
  "Treat the observations as the only factual evidence; do not rely on external knowledge or other trace events.",
  "Approve only when the answer addresses the request and is supported by the observations.",
  "If rejected, provide specific, actionable corrections in feedback. If approved, briefly state why it is supported.",
].join(" ");
```

Note o campo `aproved` (sem o segundo "p") — é assim mesmo no schema; vale notar para não estranhar nos testes e no trace.

O crítico não vê o trace inteiro — só as **observações** (o que as tools devolveram de verdade):

```typescript
// src/agents/reflection.ts
function observationsOf(trace: readonly TraceEvent[]): string {
  const observations = trace
    .filter((event) => event.type === "observation")
    .map((event, index) => `${index + 1}. ${event.content}`);
  return observations.length > 0 ? observations.join("\n") : "(none)";
}
```

É uma escolha esperta: as observações são a **verdade de campo**. Se a resposta diz "4 alertas críticos" e a observação diz "Found 3 firing alert(s)", o crítico tem como pegar a contradição. Os pensamentos (`thought`) e ações (`action`) do agente ficam de fora — o `reflection.test.ts` confere isso diretamente, fabricando um trace com um `thought` e uma `action` "desnecessários" e checando que o texto enviado ao modelo-crítico (`model.inputs`) não contém esse conteúdo (`assert.doesNotMatch(criticInputs, /Private reasoning not available/)`).

Esse é o mesmo mecanismo de "LLM avaliando LLM" que você viu nos guardrails de [`guardrails.md`](../../../modulo02-integracao-apis-llms/05-safeguard-prompt-injection/docs/guardrails.md) — só que lá o avaliador bloqueava entradas maliciosas; aqui ele avalia a **qualidade da saída** e pode pedir uma nova tentativa.

```typescript
// src/agents/reflection.ts
async function critique(
  request: string,
  result: ReasoningResult,
  model: BaseChatModel,
): Promise<{ decision: z.infer<typeof ReflectionCritiqueSchema>; latencyMs: number }> {
  const startedAt = performance.now();
  const output = await model.withStructuredOutput(ReflectionCritiqueSchema).invoke([
    new SystemMessage(CRITIC_PROMPT),
    new HumanMessage(
      `Request: ${request}\nObservations:\n${observationsOf(result.trace)}\nAnswer: ${result.answer}`,
    ),
  ]);
  const decision = ReflectionCritiqueSchema.parse(output);
  return { decision, latencyMs: performance.now() - startedAt };
}
```

### Sem fail-safe: um crítico quebrado propaga o erro

Diferente do que se poderia supor para uma camada "de melhoria", o nosso decorator **não** trata uma falha do crítico como aprovação automática. Se `model.withStructuredOutput(...).invoke(...)` lançar (modelo fora do ar, JSON que não valida contra `ReflectionCritiqueSchema`), o erro sobe e `strategy.run(...)` rejeita — não existe um `try/catch` ao redor da chamada ao crítico em `withReflection`. O `reflection.test.ts` confirma esse comportamento lançando um erro customizado a partir de um modelo-crítico controlado e checando que `withReflection(...).run(...)` rejeita com exatamente esse mesmo erro:

```typescript
// src/agents/reflection.test.ts
const criticError = new Error("Critic failed.");
const model = new ControlledCriticModel([criticError]);
await assert.rejects(
  withReflection(controlledStrategy(["Answer."], [], seenModels)).run("Original request.", { model }),
  (error: unknown) => error === criticError,
);
```

### O feedback volta como prefixo do input

```typescript
// src/agents/reflection.ts
function revisionInput(request: string, feedback: string): string {
  return `${request}\n\nCritic feedback for revising the previous answer:\n${feedback}`;
}
```

Como o contrato só aceita `run(input: string, options?)`, a única forma de passar o feedback para a estratégia base é **dentro do texto do input**. Não é elegante, mas mantém o decorator 100% genérico — ele não precisa saber nada de ReAct nem de Plan-and-Execute.

### O loop do decorator

```typescript
// src/agents/reflection.ts
export function withReflection(
  strategy: ReasoningStrategy,
  opts: z.input<typeof ReflectionOptionsSchema> = {},
): ReasoningStrategy {
  const { maxReflections } = ReflectionOptionsSchema.parse(opts);
  return {
    name: `reflect:${strategy.name}`,
    async run(input: string, options?: ReasoningOptions): Promise<ReasoningResult> {
      if (maxReflections === 0) {
        return parseAttempt(await strategy.run(input, options));
      }

      const model = resolveModel(options?.model);
      const strategyOptions: ReasoningOptions = { ...(options ?? {}), model };
      const trace: TraceEvent[] = [];
      let llmCalls = 0;
      let latencyMs = 0;
      let result = parseAttempt(await strategy.run(input, strategyOptions));
      let reflectionCount = 0;

      while (true) {
        trace.push(...result.trace);
        llmCalls += result.metrics.llmCalls;
        latencyMs += result.metrics.latencyMs;

        const { decision, latencyMs: critiqueLatencyMs } = await critique(input, result, model);
        llmCalls += 1;
        latencyMs += critiqueLatencyMs;
        trace.push({
          type: "critique",
          content: JSON.stringify({ aproved: decision.aproved, feedback: decision.feedback }),
        });

        if (decision.aproved || reflectionCount >= maxReflections) {
          break;
        }

        reflectionCount += 1;
        result = parseAttempt(await strategy.run(revisionInput(input, decision.feedback), strategyOptions));
      }

      return ReasoningResultSchema.parse({
        answer: result.answer,
        trace,
        metrics: { llmCalls, latencyMs },
      });
    },
  };
}
```

Pontos para estudar:

- **Teto de rodadas** (`maxReflections`, padrão 2): sem ele, um crítico exigente e um agente teimoso ficariam num loop caro para sempre. Com 2 e um crítico que sempre reprova, a conta é fixa: 3 execuções da base + 3 críticas (uma crítica acontece mesmo na última rodada, antes de desistir).
- **Trace acumulado**: todas as tentativas ficam no trace, seguidas por um evento `critique` cujo `content` é o JSON `{ aproved, feedback }` serializado — o schema de `TraceEvent` (`src/agents/trace.ts`) não tem campos dedicados de `round`/`approved`/`timestampMs` para esse tipo de evento; a Reflection carrega essa informação dentro do próprio texto.
- **Métricas somadas**: `llmCalls` = chamadas de todas as execuções da base + uma por rodada de crítica. Isso é o que torna honesta a comparação `react` × `reflect:react` na arena — a reflexão custa, e o número mostra.
- **`maxReflections: 0` vira passthrough**: nesse caso a função nem resolve um modelo e devolve o resultado da estratégia base sem tocar no crítico — não existe um ponto de injeção `opts.critic` separado do modelo; quem quiser testar sem LLM injeta `options.model` com um modelo falso (como faz `reflection.test.ts`).

## Testes sem LLM nenhum

A injeção acontece por `options.model`, não por um `opts.critic` separado: o teste monta um `ControlledCriticModel` (um `SimpleChatModel` cujo `withStructuredOutput(...)` devolve respostas de uma fila pré-programada) e injeta uma estratégia de base também falsa:

```typescript
// src/agents/reflection.test.ts
test("reflection defaults to at most two regenerations", async () => {
  const model = new ControlledCriticModel([
    rejection("First correction."),
    rejection("Second correction."),
    rejection("Final correction."),
  ]);
  const seenInputs: string[] = [];
  const strategy = withReflection(controlledStrategy(["Answer one.", "Answer two.", "Answer three."], seenInputs, []));

  const result = await strategy.run("Original request.", { model });

  assert.equal(result.answer, "Answer three.");
  assert.equal(seenInputs.length, 3);
  const critiques = result.trace.filter((event) => event.type === "critique");
  assert.equal(critiques.length, 3);
  assert.ok(critiques.every((event) => JSON.parse(event.content).aproved === false));
  assert.equal(result.metrics.llmCalls, 9);
});
```

`controlledStrategy(...)` é uma `ReasoningStrategy` fabricada à mão que registra os inputs recebidos e devolve, a cada chamada, a próxima resposta de uma lista fixa, com `metrics.llmCalls: 2`. Com um crítico que sempre rejeita, a conta de `llmCalls` é `3 execuções × 2 + 3 críticas × 1 = 9` — e é exatamente isso que o teste confere. Nenhuma rede, resultado idêntico em toda execução.

## O que muda na arena

A arena ganha dois nomes novos, e a criação é só compor o decorator em torno das estratégias já existentes:

```typescript
// src/arena.ts
export const arenaStrategies: Readonly<Record<string, ReasoningStrategy>> = {
  react: reactStrategy,
  "plan-and-execute": planAndExecuteStrategy,
  "reflect:react": withReflection(reactStrategy),
  "reflect:plan-and-execute": withReflection(planAndExecuteStrategy),
};
```

Não há `modelFactory` nem `tools` para passar na composição: `withReflection` não precisa de nenhum argumento além da estratégia base — ela resolve seu próprio modelo via `resolveModel(options?.model)` no momento do `run`, igual às outras estratégias.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Spec (nosso) | [`specs/002-camada-reflection/spec.md`](../specs/002-camada-reflection/spec.md) | US1–US3 e os requisitos funcionais |
| Contrato do decorator | [`specs/002-camada-reflection/contracts/`](../specs/002-camada-reflection/) | Assinatura de `withReflection` e das opções |
| Decorator (nosso) | [`src/agents/reflection.ts`](../src/agents/reflection.ts) | `withReflection`, crítica `{ aproved, feedback }`, sem fail-safe no erro do crítico |
| Testes (nosso) | [`src/agents/reflection.test.ts`](../src/agents/reflection.test.ts) | `ControlledCriticModel`, `controlledStrategy`, contas de `llmCalls` |
| Composição na arena (nosso) | [`src/arena.ts`](../src/arena.ts) | `reflect:react` e `reflect:plan-and-execute` no registro `arenaStrategies` |
