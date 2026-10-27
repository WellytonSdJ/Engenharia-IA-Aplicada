# Quickstart: Persistência Real de Operações

## Prerequisites

- Node.js 22 LTS (compatível com Node 24 no ambiente local).
- Dependências instaladas com `npm install` — nenhuma dependência nova de produção é necessária (`node:sqlite` é nativo).
- Nenhum servidor de banco de dados externo é necessário.
- `data/` (onde fica `opspilot.db`) é criado automaticamente na primeira execução e não deve ser versionado.

## Deterministic Validation

```powershell
npm run typecheck
npm test
```

Os testes do `SqliteOpsStore` usam `:memory:` e não tocam no arquivo real de dados nem exigem rede.

## Seed

```powershell
npm run seed
```

A saída deve confirmar 6 serviços, 6 alertas (3 `firing`, 3 `resolved`) e 3 runbooks (checkout, payments, identity). O comando inicializa/zera `OPSPILOT_DB` (default `./data/opspilot.db`) com o cenário "Mercadinho" (ver `data-model.md`). Rodar o seed duas vezes seguidas deve produzir exatamente o mesmo resultado (idempotência, FR-010).

## Persistência entre execuções (User Story 1)

```powershell
npm run seed
node --env-file-if-exists=.env --import tsx -e "
  const { createOpsTools } = await import('./src/services/tools.ts');
  const tools = createOpsTools((await import('./src/services/tools.ts')).defaultOpsStore);
  console.log(await tools[1].invoke({ title: 'Teste de persistência', service: 'checkout', severity: 'high' }));
"
node --env-file-if-exists=.env --import tsx -e "
  const { defaultOpsStore } = await import('./src/services/tools.ts');
  console.log(await defaultOpsStore.listIncidents('open'));
"
```

O segundo processo (novo `node`, sem estado em memória do primeiro) deve listar o incidente criado pelo primeiro — prova de que o dado sobreviveu ao reinício do processo.

## Consultar runbook (User Story 2)

```powershell
node --env-file-if-exists=.env --import tsx -e "
  const { defaultOpsStore } = await import('./src/services/tools.ts');
  console.log(await defaultOpsStore.findRunbook('payments'));   // runbook encontrado
  console.log(await defaultOpsStore.findRunbook('notifications')); // null (sem runbook, sem erro)
"
```

## Focused Deterministic Checks

```powershell
node --import tsx --test "src/models/sqlite-store.test.ts" "src/services/tools.test.ts"
```

## Bench / Arena (sem alteração de comportamento)

```powershell
npm run bench
npm run arena -- --strategies react,plan-and-execute --max-iterations 8 "liste os alertas firing"
```

Continuam usando `InMemoryOpsStore` por execução (reprodutibilidade dos cenários C1/C2/C3), sem depender do arquivo SQLite real.

## Expected Checks

- Nenhuma dependência de banco de dados externo instalada na máquina que roda os testes.
- `data/opspilot.db` nunca aparece em `git status` como arquivo rastreado.
- Rodar `npm run seed` duas vezes seguidas produz a mesma contagem de serviços/alertas/runbooks.
- Um incidente criado em um processo aparece em `list_incidents` de outro processo, após `npm run seed` ter rodado uma vez.
- `findRunbook` diferencia serviço inexistente (erro) de serviço sem runbook (retorno `null`/"não encontrado").
