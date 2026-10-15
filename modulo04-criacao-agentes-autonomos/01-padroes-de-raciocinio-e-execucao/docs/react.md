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

## Como está sendo usado

```typescript
// src/agents/react.ts
export function createReactStrategy(agentFactory: ReactAgentFactory = langGraphAgentFactory): ReasoningStrategy {
  return {
    name: "react",
    async run(input: string, options?: ReasoningOptions) {
      const { maxIterations } = ReasoningOptionsSchema.parse({ maxIterations: options?.maxIterations });
      if (!input.trim()) {
        throw new ValidationError("Input must not be empty.");
      }
      const model = resolveModel(options?.model);
      const store = options?.store ?? defaultOpsStore;
      const context = createExecutionContext(maxIterations);
      const agent = agentFactory(model, [...createOpsTools(store)], maxIterations * 2 + 1);
      let messages: BaseMessage[];
      try {
        const result = await agent.invoke(
          { messages: [new HumanMessage(input)] },
          { recursionLimit: maxIterations * 2 + 1 },
        );
        messages = result.messages;
      } catch (error) {
        if (error instanceof Error && error.name === "GraphRecursionError") {
          throw new IterationLimitError(maxIterations);
        }
        throw error;
      }
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
    },
  };
}

export const reactStrategy = createReactStrategy();
```

A estratégia não é uma classe — é uma função fábrica que devolve um objeto `{ name, run }`. `createReactStrategy` aceita um `agentFactory` opcional (que por padrão monta o `createReactAgent` do LangGraph): é esse parâmetro que permite ao `react.test.ts` injetar um agente falso e testar o comportamento sem rede.

### `createReactAgent`: o loop pronto do LangGraph

`createReactAgent` (de `@langchain/langgraph/prebuilt`) monta por você um `StateGraph` de dois nós que você poderia ter escrito à mão depois do módulo 02:

```
START → agent (LLM com tools "bindadas")
          │
          ├── resposta tem tool_calls? → tools (executa) → volta para agent
          └── não tem?                 → END
```

O estado desse grafo é só a lista de `messages`. Por isso o resultado é `result.messages` — e por isso precisamos de `traceFromMessages` (veja [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md)) para transformar esse histórico em trace tipado.

Nossa `research.md` registrou a escolha: *"Implementar manualmente o loop ReAct foi rejeitado porque duplicaria lógica do LangGraph"*.

### `options.model` em vez de um modelo fixo no construtor

A estratégia recebe o modelo por `ReasoningOptions.model`, um campo **opcional**, nunca por construtor. Quando não é passado, `resolveModel` cria um por conta própria:

```typescript
// src/agents/model.ts
export function resolveModel(
  injectedModel?: BaseChatModel,
  factory: () => BaseChatModel = createModel,
): BaseChatModel {
  return injectedModel ?? factory();
}
```

Dois motivos para esse desenho:

1. **Isolamento**: `createReactAgent` "binda" as tools no modelo. Se a mesma instância fosse compartilhada com outra estratégia (ou com o planner do Plan-and-Execute, que usa `withStructuredOutput`), uma configuração vazaria na outra. Cada chamada de `run` sem `options.model` pede uma instância nova via `createModel()`.
2. **Testabilidade**: num teste, basta injetar `options.model` com um modelo falso (`SimpleChatModel` que lança se for chamado) — e nada vai para a rede. O `react.test.ts` e o `reflection.test.ts` fazem exatamente isso.

A fábrica real é uma só para o projeto inteiro:

```typescript
// src/agents/model.ts
export function createModel(): ChatOpenAI {
  const apiKey = process.env.OPENROUTER_API_KEY;
  const model = process.env.OPENROUTER_MODEL;
  // `!valor` (e não `valor === undefined`) também barra string vazia — o caso de copiar
  // o .env.example sem preencher, que faria o erro aparecer só no provedor.
  if (!apiKey || !model) {
    throw new AgentConfigurationError(
      "OPENROUTER_API_KEY and OPENROUTER_MODEL are required to run an LLM strategy.",
    );
  }
  return new ChatOpenAI({
    model,
    apiKey,
    configuration: { baseURL: "https://openrouter.ai/api/v1" },
    temperature: 0,
  });
}
```

`ChatOpenAI` apontado para o OpenRouter é o mesmo truque que você viu em [`openrouter-sdk.md`](../../../modulo02-integracao-apis-llms/01-smart-model-router-gateway/docs/openrouter-sdk.md): o OpenRouter fala o protocolo da OpenAI, então o cliente da OpenAI serve. `temperature: 0` reduz a variação entre execuções — importante quando o objetivo é **comparar** estratégias. Diferente de um valor padrão hardcoded, aqui a ausência de `OPENROUTER_MODEL` é tratada como erro de configuração (`AgentConfigurationError`), não como fallback silencioso para outro modelo.

## O limite de iterações

Um agente ReAct pode entrar em loop: chama a mesma tool de novo, e de novo. O freio é o `recursionLimit` do LangGraph — o número máximo de **passos do grafo** (super-steps) numa execução:

```typescript
// src/agents/react.ts
const agent = agentFactory(model, [...createOpsTools(store)], maxIterations * 2 + 1);
// ...
const result = await agent.invoke(
  { messages: [new HumanMessage(input)] },
  { recursionLimit: maxIterations * 2 + 1 },
);
```

Por que `* 2 + 1`? Porque uma "iteração" ReAct (pensar + agir) atravessa mais de um nó do grafo (`agent` → `tools` → `agent`). O `maxIterations` é a unidade que faz sentido para humanos; o `recursionLimit` é a unidade do LangGraph; `* 2 + 1` converte um número de iterações num teto de passos do grafo com folga para a resposta final.

Quando o limite estoura, o LangGraph lança um erro chamado `GraphRecursionError`, e a estratégia o traduz para o próprio vocabulário de erro do domínio em vez de deixá-lo escapar como está:

```typescript
// src/agents/react.ts
try {
  const result = await agent.invoke(
    { messages: [new HumanMessage(input)] },
    { recursionLimit: maxIterations * 2 + 1 },
  );
  messages = result.messages;
} catch (error) {
  if (error instanceof Error && error.name === "GraphRecursionError") {
    throw new IterationLimitError(maxIterations);
  }
  throw error;
}
```

A estratégia checa `error.name === "GraphRecursionError"` (em vez de `instanceof`) para não precisar importar a classe interna do LangGraph só para esse `catch`. Note que, nesse caminho, a execução **lança** `IterationLimitError` — ela não devolve um `ReasoningResult` parcial com uma resposta tipo "[Iteration limit reached]". Quem chama a estratégia (arena, bench) decide o que fazer com esse erro: o bench, por exemplo, captura a exceção na própria célula e registra `llmCalls: null` na linha correspondente (veja [arena-e-bench.md](./arena-e-bench.md)).

Há um segundo limite, verificado depois do `invoke` retornar com sucesso: a estratégia conta quantas respostas do modelo (`AIMessage`) vieram no histórico e, se isso exceder `maxIterations`, também lança `IterationLimitError` — mesmo que o grafo não tenha estourado o `recursionLimit` por conta da folga do `+1`.

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
| Estratégia (nosso) | [`src/agents/react.ts`](../src/agents/react.ts) | `createReactAgent`, `recursionLimit`, `GraphRecursionError` |
| Fábrica (nosso) | [`src/agents/model.ts`](../src/agents/model.ts) | OpenRouter via `ChatOpenAI`, `temperature: 0` |
