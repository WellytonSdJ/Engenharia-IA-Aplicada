# Arena CLI Contract

## Invocation

```text
npm run arena -- --strategies react,plan-and-execute --max-iterations 8 "investigue os alertas firing"
```

- `--strategies`: lista separada por vírgulas; nomes estáveis `react`, `plan-and-execute`, `reflect:react` e `reflect:plan-and-execute`. Os nomes sem prefixo executam as estratégias base sem reflection.
- `--max-iterations`: inteiro positivo usado por todas as estratégias selecionadas.
- Input posicional: texto operacional comum às estratégias.

## Output

A arena imprime um bloco por estratégia contendo:

- nome da estratégia;
- resposta final;
- trace em ordem, incluindo ações e observações;
- `llmCalls`, `latencyMs` e iterações quando disponíveis.

As variantes refletidas incluem eventos `critique` no trace. Suas métricas contam as chamadas e a latência das tentativas de regeneração e do crítico, sem misturar resultados entre estratégias.

## Reflection Variants

```text
npm run arena -- --strategies reflect:react,reflect:plan-and-execute --max-iterations 8 "investigue os alertas firing"
```

Cada identificador seleciona a estratégia base correspondente com revisão estruturada e até duas regenerações por padrão. A camada pode ser configurada por `maxReflections` quando usada diretamente como decorador; essa opção não altera a flag `--max-iterations` da arena.

Argumentos inválidos devem produzir mensagem de erro na borda CLI e código de saída diferente de zero, sem iniciar chamadas de LLM.
