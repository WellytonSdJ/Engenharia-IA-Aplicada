# Reflection Contract

## Strategy Wrapper

```ts
withReflection(strategy: ReasoningStrategy, opts?: { maxReflections?: number }): ReasoningStrategy
```

- O nome da estratégia decorada é `reflect:<strategy.name>`.
- `maxReflections` é inteiro não negativo e tem default `2`; conta regenerações, não críticas. Valor zero executa a estratégia base sem avaliação.
- O wrapper reusa o mesmo `BaseChatModel` resolvido/injetado para as execuções base e o crítico.
- Estratégias sem prefixo permanecem inalteradas.

## Critic Input and Output

O crítico usa saída estruturada validada por Zod:

```ts
{
  aproved: boolean;
  feedback: string;
}
```

O prompt de sistema limita a avaliação ao pedido original e às observações do trace. A entrada do usuário contém o pedido original, apenas conteúdos de eventos `observation`, e a resposta candidata. O crítico não pode usar outros eventos do trace nem conhecimento externo como evidência.

Quando `aproved` é `false` e existe espaço de reflexão, a estratégia é chamada novamente com o pedido original e o feedback anterior em um bloco contextual separado. A crítica seguinte continua recebendo o pedido original, não a string com feedback.

## Result

- O trace preserva todos os eventos produzidos pela estratégia e acrescenta após cada crítica um evento `critique` cujo `content` identifica `aproved` e `feedback`.
- A resposta aprovada encerra o ciclo. Se o limite de regenerações se esgota com reprovação, retorna-se a resposta mais recente e o evento final registra essa reprovação.
- `llmCalls` soma as chamadas de todas as tentativas e uma chamada por avaliação do crítico.
- `latencyMs` inclui a latência das tentativas e o tempo decorrido em cada crítica.
- Falhas do modelo ou parsing da estrutura são falhas da execução e não geram resultado substituto.

## Arena CLI

```text
npm run arena -- --strategies reflect:react,reflect:plan-and-execute --max-iterations 8 "investigue os alertas firing"
```

A arena lista as duas variantes como execuções independentes. `react` e `plan-and-execute` sem prefixo continuam selecionando suas estratégias base.
