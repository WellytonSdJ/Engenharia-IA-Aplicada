# ReAct — Reasoning + Acting

## O que é

ReAct (de *Reasoning + Acting*, paper de Yao et al., 2022) é o padrão de agente mais simples e mais usado: o modelo **alterna** entre pensar e agir, e cada ação traz uma observação que alimenta o próximo pensamento.

```
Thought:      "preciso saber quais alertas estão disparando"
Action:       list_alerts({ status: "firing" })
Observation:  "Found 3 firing alert(s): payment-api, auth-service, order-service"
Thought:      "já sei o suficiente"
Answer:       "Três serviços têm alertas ativos: ..."
```

Ele não faz plano. Decide **um passo de cada vez**, olhando só para o que já aconteceu. É como dirigir numa cidade desconhecida sem GPS: você olha a placa, vira, olha a próxima placa, vira de novo. Funciona muito bem para trajetos curtos; em trajetos longos, você pode ficar dando voltas.

Você já usou ReAct sem esse nome: o `createAgent` do [`01-multiple-mcp-tools`](../../../modulo03-mcp-na-pratica/01-multiple-mcp-tools/docs/agente-autonomo-vs-orquestracao-explicita.md) roda exatamente esse loop ("tool? → executa → observa"). A diferença é que aqui ele é **uma estratégia entre outras**, atrás do contrato comum, com trace e métricas extraídos para comparação.

## Como está sendo usado (código de referência)

```typescript
// src/agents/react.ts (referência)
export class ReactStrategy implements ReasoningStrategy {
  readonly name = "react";
  private readonly modelFactory: () => ChatOpenAI;
  private readonly tools: DynamicStructuredTool[];
  private readonly maxIterations: number;

  constructor(options: ReactStrategyOptions) {
    this.modelFactory = options.modelFactory;
    this.tools = options.tools;
    this.maxIterations = options.maxIterations;
  }

  async run(input: string): Promise<StrategyResult> {
    const startedAt = Date.now();
    const model = this.modelFactory();
    const agent = createReactAgent({ llm: model, tools: this.tools });

    try {
      const result = await agent.invoke(
        { messages: [{ role: "user", content: input }] },
        { recursionLimit: Math.max(3, this.maxIterations * 3) },
      );
      const trace = buildTraceFromMessages(result.messages);
      const answer = trace.filter((item) => item.type === "answer").at(-1)?.content ?? "No answer generated.";
      const llmCalls = result.messages.filter((message) => message instanceof AIMessage).length;
      // ...
```

### `createReactAgent`: o loop pronto do LangGraph

`createReactAgent` (de `@langchain/langgraph/prebuilt`) monta por você um `StateGraph` de dois nós que você poderia ter escrito à mão depois do módulo 02:

```
START → agent (LLM com tools "bindadas")
          │
          ├── resposta tem tool_calls? → tools (executa) → volta para agent
          └── não tem?                 → END
```

O estado desse grafo é só a lista de `messages`. Por isso o resultado é `result.messages` — e por isso precisamos do `buildTraceFromMessages` (veja [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md)) para transformar esse histórico em trace tipado.

Nossa `research.md` registrou a escolha: *"Implementar manualmente o loop ReAct foi rejeitado porque duplicaria lógica do LangGraph"*.

### `modelFactory` em vez de `model`

A estratégia recebe uma **função que cria** o modelo, não o modelo pronto. Dois motivos:

1. **Isolamento**: `createReactAgent` "binda" as tools no modelo. Se a mesma instância fosse compartilhada com outra estratégia (ou com o planner do Plan-and-Execute, que usa `withStructuredOutput`), uma configuração vazaria na outra. Cada `run` pede uma instância nova.
2. **Testabilidade**: num teste, basta passar `() => { throw new Error("unused") }` ou um modelo falso — e nada vai para a rede. O teste da Reflection faz exatamente isso para construir uma `ReactStrategy` sem LLM.

A fábrica real é uma só para o projeto inteiro:

```typescript
// src/agents/model.ts (referência)
export function createModel(): ChatOpenAI {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY environment variable is required");
  }

  return new ChatOpenAI({
    apiKey,
    model: process.env.OPENROUTER_MODEL ?? "openai/gpt-4o-mini",
    temperature: 0,
    configuration: {
      baseURL: OPENROUTER_BASE_URL,
    },
  });
}
```

`ChatOpenAI` apontado para o OpenRouter é o mesmo truque que você viu em [`openrouter-sdk.md`](../../../modulo02-integracao-apis-llms/01-smart-model-router-gateway/docs/openrouter-sdk.md): o OpenRouter fala o protocolo da OpenAI, então o cliente da OpenAI serve. `temperature: 0` reduz a variação entre execuções — importante quando o objetivo é **comparar** estratégias.

## O limite de iterações

Um agente ReAct pode entrar em loop: chama a mesma tool de novo, e de novo. O freio é o `recursionLimit` do LangGraph — o número máximo de **passos do grafo** (super-steps) numa execução:

```typescript
{ recursionLimit: Math.max(3, this.maxIterations * 3) }
```

Por que `* 3`? Porque uma "iteração" ReAct (pensar + agir) atravessa mais de um nó do grafo (`agent` → `tools` → `agent`). O `maxIterations` é a unidade que faz sentido para humanos; o `recursionLimit` é a unidade do LangGraph; o fator 3 é uma folga de conversão. `Math.max(3, ...)` garante um mínimo para o grafo conseguir ao menos rodar.

Quando o limite estoura, o LangGraph lança `GraphRecursionError`, e a estratégia transforma isso numa resposta, em vez de derrubar a arena:

```typescript
} catch (error) {
  if (error instanceof GraphRecursionError) {
    const answer = `[Iteration limit reached after ${this.maxIterations} steps. Partial result unavailable.]`;
    return {
      answer,
      trace: [{ type: "answer", content: answer }],
      metrics: {
        llmCalls: 0,
        latencyMs: Date.now() - startedAt,
      },
    };
  }
  throw error;
}
```

Repare numa limitação honesta: nesse caminho o trace parcial e o `llmCalls` real **se perdem** (`llmCalls: 0`, trace só com o answer), porque o `invoke` não devolveu as mensagens. O nosso `tasks.md` pede mais: T023 exige "uma `answer` final ou uma falha de limite **estruturada**", e T013 exige contar **toda** invocação do modelo. Uma forma de cumprir isso é usar `agent.stream(...)` em vez de `invoke` e ir acumulando mensagens — assim, quando o limite estourar, você ainda tem o que já aconteceu.

## Quando o ReAct vai bem e quando tropeça

| Pedido | Como o ReAct se sai |
| --- | --- |
| "quantos alertas críticos estão disparando?" (C1) | Bem: 1 tool, 1 resposta. Barato e rápido |
| "abra três incidentes sev2 para checkout, payment e catalog, nessa ordem, e resolva o primeiro" (C2) | Arriscado: muitas ações encadeadas, precisa lembrar da ordem e do id do primeiro. Pode abrir em paralelo e perder a ordem |
| "dos alertas disparando, abra um incidente para o mais antigo e diga quantos sobraram" (C3) | Depende do modelo: precisa interpretar "mais antigo" a partir da observação e fazer uma conta no final |

Essa é justamente a motivação do próximo padrão: quando a tarefa tem **estrutura**, vale a pena escrever o plano antes. Veja [plan-and-execute.md](./plan-and-execute.md).

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Decisão de usar o prebuilt | [`specs/.../research.md`](../specs/001-nucleo-raciocinio-opspilot/research.md) | Decision 2 |
| Requisito | [`specs/.../spec.md`](../specs/001-nucleo-raciocinio-opspilot/spec.md) | FR-010 e FR-013 |
| Tarefas relacionadas | [`specs/.../tasks.md`](../specs/001-nucleo-raciocinio-opspilot/tasks.md) | T011, T021, T023, T024 |
| Estratégia (referência) | `src/agents/react.ts` do snapshot | `createReactAgent`, `recursionLimit`, `GraphRecursionError` |
| Fábrica (referência) | `src/agents/model.ts` do snapshot | OpenRouter via `ChatOpenAI`, `temperature: 0` |
