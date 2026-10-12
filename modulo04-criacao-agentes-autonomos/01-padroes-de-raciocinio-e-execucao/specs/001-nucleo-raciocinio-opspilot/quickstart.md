# Quickstart: Núcleo de raciocínio do OpsPilot

## Prerequisites

- Node.js 22 LTS.
- Dependências instaladas com `npm install`.
- `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` configurados somente para execuções com LLM.
- O runtime e o seed usam `src/data/ops-store.json`; não é necessário MySQL.

## Deterministic Validation

```powershell
npm test
npm run typecheck
```

Os testes do store e do formatador de trace devem passar sem credenciais e sem rede.
O padrão de testes segue o runner `node:test`, com arquivos co-localizados em `src/**/*.test.ts`.

## Seed

```powershell
npm run seed
```

A saída deve confirmar 5 serviços e 6 alertas, sendo 3 `firing` e 3 `resolved`. O comando inicializa/restaura `src/data/ops-store.json`; ferramentas usam esse arquivo para consultar alertas e incidentes e persistir incidentes novos/resolvidos. Nos testes, injete stores ou arquivos temporários para não alterar o seed versionado.

## Focused Deterministic Checks

```powershell
node --import tsx --test "src/models/in-memory-store.test.ts" "src/agents/trace.test.ts"
```

Esses testes usam somente dados em memória e não constroem clientes nem fazem chamadas ao OpenRouter.

## Arena

```powershell
npm run arena -- --strategies react,plan-and-execute --max-iterations 8 "liste os alertas firing"
```

A saída deve separar as estratégias e mostrar resposta, trace e métricas. Sem credenciais válidas, use testes ou um double de modelo; não registre chaves em logs.

## Expected Checks

- Nenhuma chamada de rede nos testes determinísticos.
- Nenhum plano Plan-and-Execute excede 8 passos acumulados; ajustes do replanner contam os passos concluídos.
- O replanner retorna uma decisão explícita `continue`, `adjust` ou `finish`.
- Nenhuma estratégia ultrapassa `--max-iterations`.
- `llmCalls` e `latencyMs` aparecem em cada resultado válido.
