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

## Como está sendo usado (código de referência)

### A interface

```typescript
// src/domain/types.ts (referência)
export type TraceEventType = "thought" | "action" | "observation" | "plan" | "critique" | "answer";

export interface TraceEvent {
  type: TraceEventType;
  content: string;
  tool?: string;
  toolArgs?: Record<string, unknown>;
  /** Reflection-layer critique round (1-based). */
  round?: number;
  /** Reflection-layer critic verdict. */
  approved?: boolean;
  /** Unix timestamp (ms) when the critique event was recorded. */
  timestampMs?: number;
}

export interface ExecutionMetrics {
  llmCalls: number;
  latencyMs: number;
}

export interface StrategyResult {
  answer: string;
  trace: TraceEvent[];
  metrics: ExecutionMetrics;
}

export interface ReasoningStrategy {
  readonly name: string;
  run(input: string): Promise<StrategyResult>;
}
```

Repare como a interface é **mínima**: um nome e um método. Tudo que é específico de cada estratégia (modelo, tools, limite de iterações) entra pelo **construtor**, não pelo `run`. Isso é o que permite a arena tratar todas iguais e a Reflection embrulhar qualquer uma.

### Os seis tipos de evento

Cada tipo representa um momento diferente do raciocínio, e cada estratégia usa um subconjunto:

| Tipo | Significado | Quem emite |
| --- | --- | --- |
| `thought` | Texto que o modelo escreveu **antes** de chamar uma tool | ReAct |
| `action` | Chamada de tool, com `tool` e `toolArgs` | ReAct, Plan-and-Execute |
| `observation` | Resultado devolvido pela tool | ReAct, Plan-and-Execute |
| `plan` | O plano numerado produzido pelo planner | Plan-and-Execute |
| `critique` | Revisão: replanner ajustando o plano, ou o crítico da Reflection | Plan-and-Execute, Reflection |
| `answer` | Resposta final | Todas |

### Do histórico de mensagens para o trace

O ReAct do LangGraph não devolve um trace — devolve o **histórico de mensagens** (as mesmas `AIMessage`/`ToolMessage` que você viu em [`langchain-messages.md`](../../../modulo02-integracao-apis-llms/02-langchain-intro/docs/langchain-messages.md)). O `buildTraceFromMessages` traduz uma coisa na outra:

```typescript
// src/trace/builder.ts (referência)
export function buildTraceFromMessages(messages: BaseMessage[]): TraceEvent[] {
  const trace: TraceEvent[] = [];

  for (const message of messages) {
    if (message instanceof AIMessage) {
      const content = toText(message.content).trim();
      if (message.tool_calls && message.tool_calls.length > 0) {
        if (content.length > 0) {
          trace.push({ type: "thought", content });
        }

        for (const call of message.tool_calls) {
          trace.push({
            type: "action",
            content: `${call.name}(${JSON.stringify(call.args ?? {})})`,
            tool: call.name,
            toolArgs: (call.args as Record<string, unknown>) ?? {},
          });
        }
      } else if (content.length > 0) {
        trace.push({ type: "answer", content });
      }
      continue;
    }

    if (message instanceof ToolMessage) {
      const content = toText(message.content).trim();
      trace.push({ type: "observation", content });
    }
  }

  const hasAnswer = trace.some((event) => event.type === "answer");
  if (!hasAnswer) {
    trace.push({ type: "answer", content: "No answer generated." });
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
nenhum answer no final     → answer "No answer generated."   ← garante o invariante
```

O último passo é importante: **todo trace termina com `answer`**. É um invariante do contrato, e o builder garante isso mesmo quando o modelo não produz texto final.

A função `toText` existe porque `message.content` nem sempre é string — alguns provedores devolvem um array de partes (`[{ type: "text", text: "..." }]`). Ela achata tudo em texto.

### Métricas: contar, não estimar

```typescript
// src/agents/react.ts (referência) — dentro de run()
const startedAt = Date.now();
// ...
const llmCalls = result.messages.filter((message) => message instanceof AIMessage).length;

return {
  answer,
  trace,
  metrics: {
    llmCalls,
    latencyMs: Date.now() - startedAt,
  },
};
```

No ReAct, cada `AIMessage` no histórico é uma resposta do modelo, logo uma chamada. No Plan-and-Execute o estado do grafo carrega um contador `llmCalls` que cada nó incrementa; na Reflection, as chamadas da estratégia base são **somadas** às do crítico. Em todos os casos o número vem do que aconteceu, não de uma estimativa a partir do texto — a nossa `research.md` registrou exatamente essa decisão ("Inferir chamadas e latência a partir de texto final foi rejeitado por ser impreciso").

## Por que o trace é tipado e não um log

Um log é texto para humano ler. Um trace tipado é **dado para código verificar**. Olhe o que um teste consegue afirmar:

```typescript
// src/trace/builder.test.ts (referência)
const trace = buildTraceFromMessages(messages);
assert.deepEqual(trace.map((event) => event.type), ["thought", "action", "observation", "answer"]);
assert.equal(trace[1]?.tool, "list_alerts");
assert.deepEqual(trace[1]?.toolArgs, { status: "firing" });
assert.equal(trace.at(-1)?.type, "answer");
```

E as mensagens desse teste são **fabricadas à mão** (`new AIMessage({ content: ..., tool_calls: [...] })`) — nenhum LLM é chamado. O trace vira uma fronteira testável entre "o que o modelo fez" e "o que a gente mostra/mede".

Nas próximas unidades do curso esse mesmo trace é persistido (spec `015-persistent-trace-logs`) e exibido numa interface web — por isso vale caprichar no contrato agora.

## Onde o nosso plano diverge da referência

O nosso `contracts/strategy.md` e o `data-model.md` são **mais rígidos** que o snapshot do curso:

| Ponto | Referência | Nosso plano |
| --- | --- | --- |
| Tipo de `TraceEvent` | Interface única com campos opcionais | União discriminada: `plan` exige `steps`, `action` exige `tool` e `args` |
| Nome dos argumentos | `toolArgs` | `args` |
| Validação | Só tipos TypeScript | Schemas Zod que **rejeitam** tipos desconhecidos (T009, FR-002) |
| `run` | `run(input)` | `run(input, options?)` com `ReasoningOptions` (`maxIterations` e dependências injetáveis) |
| Métricas | `llmCalls`, `latencyMs` | + `iterations?` opcional |
| Arquivos | `src/domain/types.ts` + `src/trace/builder.ts` | `src/agents/strategy.ts` + `src/agents/trace.ts` |

A união discriminada é uma melhoria real: com ela, o TypeScript **proíbe** um `action` sem `tool`, enquanto na referência isso só é garantido por disciplina. Só tome cuidado para que a Reflection (spec 002) ainda caiba — ela adiciona `round`, `approved` e `timestampMs` aos eventos `critique`, e o nosso contrato já diz que "a implementação pode adicionar metadados não obrigatórios".

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Contrato (nosso) | [`specs/.../contracts/strategy.md`](../specs/001-nucleo-raciocinio-opspilot/contracts/strategy.md) | União discriminada de `TraceEvent` e `ReasoningOptions` |
| Modelo de dados (nosso) | [`specs/.../data-model.md`](../specs/001-nucleo-raciocinio-opspilot/data-model.md) | `TraceEvent`, `ReasoningMetrics`, `ReasoningResult` |
| Tarefas relacionadas | [`specs/.../tasks.md`](../specs/001-nucleo-raciocinio-opspilot/tasks.md) | T009, T010, T012, T013, T014 |
| Interface (referência) | `src/domain/types.ts` do snapshot do curso | `ReasoningStrategy` com um único método |
| Builder (referência) | `src/trace/builder.ts` do snapshot do curso | Regra mensagem → evento e o `answer` garantido |
| Testes (referência) | `src/trace/builder.test.ts` do snapshot do curso | Mensagens fabricadas à mão, zero rede |
