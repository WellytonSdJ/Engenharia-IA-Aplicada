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
