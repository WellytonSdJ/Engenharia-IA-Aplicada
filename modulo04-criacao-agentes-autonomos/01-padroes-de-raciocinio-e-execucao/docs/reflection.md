# Reflection — a camada de autocrítica

> Este doc cobre a **spec 002 (`002-reflection-layer`)** do curso. No nosso projeto ela ainda não foi especificada: depois de terminar a `001`, o caminho é rodar `/speckit.specify` para a reflection. O que está aqui serve de base para escrever essa spec.

## O que é

Reflection é o padrão em que **uma segunda passada de LLM avalia a resposta da primeira** e, se não gostar, manda refazer com feedback. É o revisor de texto: você escreve, o revisor devolve com anotações ("faltou dizer quantos sobraram"), você reescreve.

A decisão de design mais interessante da unidade é **onde** a Reflection mora. Ela não é uma terceira estratégia ao lado de ReAct e Plan-and-Execute — é um **decorator**: uma função que recebe *qualquer* `ReasoningStrategy` e devolve *outra* `ReasoningStrategy`, com a crítica embutida.

```
withReflection(new ReactStrategy(...))        → name: "reflect:react"
withReflection(new PlanExecuteStrategy(...))  → name: "reflect:plan-and-execute"
```

Isso só é possível porque o contrato comum (veja [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md)) é mínimo: um `name` e um `run(input)`. O decorator implementa a mesma interface e, por dentro, chama a estratégia original quantas vezes precisar. Quem usa não percebe a diferença — a arena trata `reflect:react` exatamente como trata `react`.

```
run(input)
  └─ base.run(input) ──────────────────────────► resposta 1
       └─ crítico(resposta 1, trace, input) → { approved: false, feedback: "faltou X" }
            └─ base.run("[Critique - Round 1]: faltou X ... Original request: input") ► resposta 2
                 └─ crítico(resposta 2, ...) → { approved: true }  → devolve resposta 2
```

## Como está sendo usado (código de referência)

### O crítico: saída estruturada e prompt ancorado no trace

```typescript
// src/strategies/reflect.ts (referência)
export const critiqueSchema = z.object({
  approved: z.boolean(),
  feedback: z
    .string()
    .describe("se não aprovado: o que corrigir, em específico e acionável"),
});

const CRITIC_PROMPT = [
  "Você é um crítico rigoroso de respostas de um agente de operações.",
  "Avalie APENAS contra as observações do trace e o pedido original.",
  "Não invente fatos que não apareçam nas observações.",
  "Se a resposta estiver completa e fiel às observações, approved=true.",
  "Se faltar informação exigida pelo pedido ou houver inconsistência com as observações, approved=false e descreva o que corrigir de forma específica e acionável.",
].join(" ");
```

O crítico não vê o trace inteiro — só as **observações** (o que as tools devolveram de verdade):

```typescript
function observationsOf(trace: TraceEvent[]): string {
  const observations = trace
    .filter((event) => event.type === "observation")
    .map((event) => event.content);
  return observations.length > 0 ? observations.join("\n") : "(nenhuma)";
}
```

É uma escolha esperta: as observações são a **verdade de campo**. Se a resposta diz "4 alertas críticos" e a observação diz "Found 3 firing alert(s)", o crítico tem como pegar a contradição. Os pensamentos (`thought`) do agente ficam de fora porque podem estar errados — e o crítico não deve ser convencido por eles.

Esse é o mesmo mecanismo de "LLM avaliando LLM" que você viu nos guardrails de [`guardrails.md`](../../../modulo02-integracao-apis-llms/05-safeguard-prompt-injection/docs/guardrails.md) — só que lá o avaliador bloqueava entradas maliciosas; aqui ele avalia a **qualidade da saída** e pode pedir uma nova tentativa.

### Fail-safe: crítico quebrado aprova

```typescript
export function createLLMCritic(modelFactory: () => ChatOpenAI): CriticFn {
  return async (answer, trace, originalInput) => {
    try {
      const raw = await modelFactory()
        .withStructuredOutput(critiqueSchema)
        .invoke([
          ["system", CRITIC_PROMPT],
          ["user", `Pedido: ${originalInput}\nObservações: ${observationsOf(trace)}\nResposta: ${answer}`],
        ]);
      return critiqueSchema.parse(raw);
    } catch {
      // FR-012: fail-safe — treat invalid critic output as approval
      return { approved: true, feedback: "" };
    }
  };
}
```

Se o crítico falhar (modelo fora do ar, JSON inválido), a resposta original é **aprovada**. A lógica: a Reflection é uma camada de melhoria; um defeito nela não deve derrubar uma resposta que a estratégia base já produziu. É uma decisão consciente, registrada como requisito (FR-012 da spec 002) — e poderia ser a oposta num domínio onde uma resposta errada é pior que nenhuma.

### O feedback volta como prefixo do input

```typescript
export function enrichInputWithFeedback(originalInput: string, round: number, feedback: string): string {
  const feedbackBody = feedback.trim() === "" ? "(sem feedback adicional)" : feedback;
  return `[Critique - Round ${round}]:\n${feedbackBody}\n\nOriginal request:\n${originalInput}`;
}
```

Como o contrato só aceita `run(input: string)`, a única forma de passar o feedback para a estratégia base é **dentro do texto do input**. Não é elegante, mas mantém o decorator 100% genérico — ele não precisa saber nada de ReAct nem de Plan-and-Execute.

### O loop do decorator

```typescript
export function withReflection(strategy: ReasoningStrategy, opts: ReflectionOpts = {}): ReasoningStrategy {
  const maxReflections = opts.maxReflections ?? 2;
  const criticFn = resolveCritic(opts, maxReflections);
  const effectiveMax = criticFn === undefined ? 0 : maxReflections;

  return {
    name: `reflect:${strategy.name}`,
    async run(input: string): Promise<StrategyResult> {
      // ...
      let currentResult = await strategy.run(input);
      totalLlmCalls += currentResult.metrics.llmCalls;
      accumulatedTrace.push(...currentResult.trace);
      // ...
      for (let round = 1; round <= effectiveMax; round += 1) {
        const critiqueResult = await criticFn(currentResult.answer, currentResult.trace, input);
        criticCallCount += 1;

        accumulatedTrace.push({
          type: "critique",
          content: critiqueEventContent(critiqueResult.feedback),
          round,
          approved: critiqueResult.approved,
          timestampMs: Date.now(),
        });

        if (critiqueResult.approved) {
          break;
        }

        const enrichedInput = enrichInputWithFeedback(input, round, critiqueResult.feedback);
        currentResult = await strategy.run(enrichedInput);
        totalLlmCalls += currentResult.metrics.llmCalls;
        accumulatedTrace.push(...currentResult.trace);
      }

      return {
        answer: currentResult.answer,
        trace: accumulatedTrace,
        metrics: {
          llmCalls: totalLlmCalls + criticCallCount,
          latencyMs: Date.now() - startedAt,
        },
      };
    },
  };
}
```

Pontos para estudar:

- **Teto de rodadas** (`maxReflections`, padrão 2): sem ele, um crítico exigente e um agente teimoso ficariam num loop caro para sempre. Com 2 e um crítico que sempre reprova, a conta é fixa: 3 execuções da base + 2 críticas.
- **Trace acumulado**: todas as tentativas ficam no trace, separadas por eventos `critique` com `round` e `approved`. Dá para ver a resposta ruim, a crítica e a resposta corrigida.
- **Métricas somadas**: `llmCalls` = chamadas de todas as execuções da base + uma por crítica. Isso é o que torna honesta a comparação `react` × `reflect:react` na arena — a reflexão custa, e o número mostra.
- **O crítico é injetável**: `opts.critic` permite passar uma função qualquer no lugar do LLM. Sem `critic` nem `modelFactory`, o decorator vira passthrough.

## Testes sem LLM nenhum

A injeção do crítico permite testar todo o comportamento com funções falsas:

```typescript
// src/strategies/reflect.test.ts (referência)
test("US2: maxReflections 2 always-reject — llmCalls === 5", async () => {
  const base = mockBase();
  const critic = alwaysRejectCritic();
  const strategy = withReflection(base, { critic, maxReflections: 2 });

  const result = await strategy.run("q");
  assert.equal(result.metrics.llmCalls, 5);
});
```

`mockBase()` é uma `ReasoningStrategy` que registra os inputs recebidos e devolve uma resposta fixa com `llmCalls: 1`; `alwaysRejectCritic()` sempre devolve `{ approved: false }`. Nenhuma rede, resultado idêntico em toda execução. Esse arquivo de teste é um ótimo modelo para os testes com "model/store doubles" que o nosso `tasks.md` pede para ReAct e Plan-and-Execute (T024, T025).

## O que muda na arena

A arena ganha dois nomes novos, e a criação é só compor:

```typescript
// src/arena.ts (referência)
if (name === "reflect:react") {
  const base = new ReactStrategy({ modelFactory: createModel, tools, maxIterations });
  return withReflection(base, { modelFactory: createModel });
}
```

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Spec de referência | `specs/002-reflection-layer/spec.md` do snapshot | US1–US3 e FR-001..FR-012 — base para o nosso futuro `/speckit.specify` |
| Contrato do decorator | `specs/002-reflection-layer/contracts/reflect-decorator.md` do snapshot | Assinatura de `withReflection` e das opções |
| Decorator (referência) | `src/strategies/reflect.ts` do snapshot | `withReflection`, `createLLMCritic`, `enrichInputWithFeedback` |
| Testes (referência) | `src/strategies/reflect.test.ts` do snapshot | `mockBase`, `sequenceCritic`, contas de `llmCalls` |
| Tipo com metadados de crítica | `src/domain/types.ts` do snapshot | `round`, `approved`, `timestampMs` em `TraceEvent` |
