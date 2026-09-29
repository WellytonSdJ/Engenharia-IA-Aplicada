# Arena CLI Contract

## Invocation

```text
npm run arena -- --strategies react,plan-and-execute --max-iterations 8 "investigue os alertas firing"
```

- `--strategies`: lista separada por vírgulas; nomes estáveis `react` e `plan-and-execute`.
- `--max-iterations`: inteiro positivo usado por todas as estratégias selecionadas.
- Input posicional: texto operacional comum às estratégias.

## Output

A arena imprime um bloco por estratégia contendo:

- nome da estratégia;
- resposta final;
- trace em ordem, incluindo ações e observações;
- `llmCalls`, `latencyMs` e iterações quando disponíveis.

Argumentos inválidos devem produzir mensagem de erro na borda CLI e código de saída diferente de zero, sem iniciar chamadas de LLM.
