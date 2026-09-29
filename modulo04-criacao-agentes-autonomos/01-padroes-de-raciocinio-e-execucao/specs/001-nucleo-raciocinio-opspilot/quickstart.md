# Quickstart: Núcleo de raciocínio do OpsPilot

## Prerequisites

- Node.js 22 LTS.
- Dependências instaladas com `npm install`.
- `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` configurados somente para execuções com LLM.
- MySQL disponível apenas para validar o adapter Sequelize; o fluxo determinístico usa memória.

## Deterministic Validation

```powershell
npm test
npm run typecheck
```

Os testes do store e do formatador de trace devem passar sem credenciais e sem rede.

## Seed

```powershell
npm run seed
```

A saída deve confirmar 5 serviços e 6 alertas, sendo 3 `firing` e 3 `resolved`. O seed deve ser seguro para repetição.

## Arena

```powershell
npm run arena -- --strategies react,plan-and-execute --max-iterations 8 "liste os alertas firing"
```

A saída deve separar as estratégias e mostrar resposta, trace e métricas. Sem credenciais válidas, use testes ou um double de modelo; não registre chaves em logs.

## Expected Checks

- Nenhuma chamada de rede nos testes determinísticos.
- Nenhum plano com mais de 8 passos.
- Nenhuma estratégia ultrapassa `--max-iterations`.
- `llmCalls` e `latencyMs` aparecem em cada resultado válido.
