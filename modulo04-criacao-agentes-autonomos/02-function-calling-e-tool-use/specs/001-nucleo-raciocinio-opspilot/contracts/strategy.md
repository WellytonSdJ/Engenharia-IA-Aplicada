# Strategy Contract

## Common Interface

```ts
interface ReasoningStrategy {
  readonly name: string;
  run(input: string, options?: ReasoningOptions): Promise<ReasoningResult>;
}
```

`ReasoningOptions` inclui pelo menos `maxIterations` e dependências injetáveis necessárias para testes. A estratégia não deve ler flags diretamente.

## Result

```ts
interface ReasoningResult {
  answer: string;
  trace: TraceEvent[];
  metrics: {
    llmCalls: number;
    latencyMs: number;
    iterations?: number;
  };
}
```

## Trace Events

```ts
type TraceEvent =
  | { type: "thought" | "observation" | "critique" | "answer"; content: string }
  | { type: "plan"; content: string; steps: string[] }
  | { type: "action"; content: string; tool: string; args: unknown };
```

A implementação pode adicionar metadados não obrigatórios, mas não pode remover os campos exigidos nem emitir tipos desconhecidos.

## Reflection Wrapper

```ts
withReflection(strategy: ReasoningStrategy, opts?: { maxReflections?: number }): ReasoningStrategy
```

- O wrapper preserva a estratégia recebida e a nomeia `reflect:<strategy.name>`.
- `maxReflections` é um inteiro não negativo que conta regenerações; o padrão é `2`. Com zero, apenas a estratégia base é executada, sem chamada ao crítico.
- Para valores positivos, a estratégia base e o crítico compartilham a mesma instância de modelo resolvida ou injetada.
- O crítico usa saída estruturada `{ aproved: boolean; feedback: string }`, validada por Zod.
- O crítico recebe o pedido original, a resposta candidata e somente os conteúdos dos eventos `observation`; outros eventos e conhecimento externo não são evidência permitida.
- Uma reprovação com ciclos restantes chama novamente a estratégia com o pedido original e o feedback anterior no contexto. Cada tentativa é avaliada usando o pedido original.
- Cada avaliação acrescenta um evento `critique` cujo campo `content` contém JSON com `aproved` e `feedback`. Se o limite se esgotar após reprovação, o resultado contém a resposta mais recente e a crítica final reprovadora.
- `llmCalls` soma as métricas de todas as tentativas e uma chamada por crítica. `latencyMs` acumula as latências das tentativas e o tempo decorrido de cada crítica.
- Falhas do crítico, da validação estruturada ou de uma regeneração são propagadas; não produzem aprovação nem resposta substituta.
