# Roteiro de Implementação — do `tasks.md` ao código

Este doc é a ponte entre o que já **planejamos** (`specs/001-nucleo-raciocinio-opspilot/`) e o que o curso **implementou** (snapshot `02-padroes-de-raciocinio-e-execucao`). Use-o antes e durante o `/speckit.implement`: ele diz onde olhar na referência para cada fase, o que o nosso plano faz diferente e quais armadilhas já dá para ver só lendo os arquivos.

> Nada aqui altera os arquivos do projeto. Os pontos marcados como **decisão** são coisas para resolver na spec/plano (via `/speckit.clarify`, `/speckit.analyze` ou editando os artefatos) antes de implementar — não durante.

---

## Onde estamos

```
✅ constituição (v1.0.1)
✅ /speckit.specify  → spec.md + checklists/requirements.md
✅ /speckit.plan     → plan.md, research.md, data-model.md, contracts/, quickstart.md
✅ /speckit.tasks    → tasks.md (T001..T036)
⬜ /speckit.analyze  → recomendado agora: cruzar spec × plan × tasks
⬜ /speckit.implement
⬜ spec 002 (reflection): /speckit.specify → plan → tasks → implement
```

---

## Mapa: nossa estrutura × referência

O nosso `plan.md` organiza `src/` em camadas (constituição, princípio II: Model → Service → Controller). A referência tem outra organização (e vários arquivos que são só re-export). A correspondência:

| Nosso arquivo (plan.md) | Equivalente na referência | Doc de estudo |
| --- | --- | --- |
| `src/models/domain.ts` | `src/domain/types.ts` (+ schemas Zod de `src/store/seed.ts` e `src/agents/tools.ts`) | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/services/errors.ts` | `src/domain/errors.ts` | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/models/in-memory-store.ts` | `src/store/in-memory-store.ts` | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/models/sequelize-store.ts` | — (não existe na U2 do curso) | — |
| `src/services/seed.ts` | `src/store/seed.ts` + `src/store/seed-data.json` | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/services/tools.ts` | `src/agents/tools.ts` | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/agents/model.ts` | `src/agents/model.ts` | [react.md](./react.md) |
| `src/agents/trace.ts` | `src/trace/builder.ts` + `TraceEvent` de `src/domain/types.ts` | [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md) |
| `src/agents/strategy.ts` | `ReasoningStrategy`/`StrategyResult` de `src/domain/types.ts` | [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md) |
| `src/agents/react.ts` | `src/agents/react.ts` | [react.md](./react.md) |
| `src/agents/plan-and-execute.ts` | `src/strategies/plan-execute.ts` | [plan-and-execute.md](./plan-and-execute.md) |
| `src/controllers/arena-controller.ts` + `src/arena.ts` | `src/arena.ts` | [arena-e-bench.md](./arena-e-bench.md) |
| `src/bench.ts` | `src/bench.ts` | [arena-e-bench.md](./arena-e-bench.md) |
| (spec 002) | `src/strategies/reflect.ts` | [reflection.md](./reflection.md) |

---

## Fase a fase

| Fase do `tasks.md` | O que estudar antes | Atenção |
| --- | --- | --- |
| **1. Setup** (T001–T003) | [spec-driven-com-spec-kit.md](./spec-driven-com-spec-kit.md) | O script `seed` é novo — a referência não tem |
| **2. Foundation** (T004–T012) | Contrato/trace, tools/store | Nosso `TraceEvent` é união discriminada com Zod — mais rígido que a referência |
| **3. US1** (T013–T015) | [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md) | Contagem de `llmCalls` tem de ser real em todos os caminhos, inclusive no estouro de limite |
| **4. US2** (T016–T020) | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) | Seed idempotente; resolver duas vezes deve dar erro |
| **5. US3** (T021–T028) | [react.md](./react.md), [plan-and-execute.md](./plan-and-execute.md), [arena-e-bench.md](./arena-e-bench.md) | Testes com model double: veja o padrão `mockBase`/`sequenceCritic` em [reflection.md](./reflection.md) |
| **6. US4** (T029–T031) | Testes do store e do trace da referência | "Sem rede" precisa ser **provado** (T029), não só presumido |
| **7. Polish** (T032–T036) | `quickstart.md` | Só atualizar o quickstart com comportamento verificado |

---

## Armadilhas já visíveis (verificadas nos arquivos atuais)

Estas não são opiniões sobre o futuro código — são inconsistências presentes **hoje** nos arquivos de configuração e specs. Nenhuma delas foi corrigida (a regra é não mexer em configuração nem em arquivos que os agentes leem); ficam aqui para você decidir.

### 1. Driver do MySQL: `mysql` em vez de `mysql2`

O nosso `package.json` declara `"mysql": "^2.18.1"`. O dialeto `mysql` do Sequelize 6 usa o pacote **`mysql2`** como driver — com só o `mysql` instalado, criar uma instância `new Sequelize(..., { dialect: "mysql" })` falha pedindo para instalar o `mysql2`. A referência do curso declara `"mysql2": "^3.11.0"`. Afeta T008 (adapter Sequelize).

### 2. `@types/sequelize` é desnecessário

O Sequelize 6 já traz os próprios tipos. O pacote `@types/sequelize` é de uma era anterior (v4) e pode gerar conflito de tipos no `npm run typecheck`. A referência não o usa.

### 3. Os scripts `arena` e `bench` não carregam o `.env`

```json
"arena": "tsx src/arena.ts",
"bench": "tsx src/bench.ts",
```

O `tsx` não lê `.env` sozinho, e a constituição proíbe `dotenv` ("carregue variáveis de ambiente usando o suporte nativo do Node.js"). A referência resolve assim:

```json
"arena": "node --env-file-if-exists=.env --import tsx src/arena.ts",
"bench": "node --env-file-if-exists=.env --import tsx src/bench.ts",
```

Sem isso, `OPENROUTER_API_KEY` não chega ao processo e a fábrica de modelo lança erro na primeira execução real da arena (T035).

### 4. `OPENROUTER_MODEL` vazio não cai no default

O `.env.example` traz `OPENROUTER_MODEL=` (vazio). A referência faz `process.env.OPENROUTER_MODEL ?? "openai/gpt-4o-mini"` — mas `??` só troca `null`/`undefined`; uma **string vazia** passa direto e o modelo fica `""`. Se copiar o `.env.example` sem preencher, o erro vai aparecer no provedor, longe da causa. Na nossa fábrica (T011), valide com Zod ou use `||` / trate string vazia.

### 5. `resolveJsonModule` ausente no `tsconfig.json`

A referência importa o seed com `import seedData from "./seed-data.json" with { type: "json" };` e tem `"resolveJsonModule": true`. O nosso `tsconfig.json` não tem essa opção. Se for usar JSON para o seed, ela precisa entrar; se o seed for um módulo `.ts`, não.

### 6. **Decisão:** `service` por nome ou `serviceId`?

O `data-model.md` define `Alert.serviceId` e diz que `open_incident` "exige título, **serviço existente**". Mas:

- os pedidos do usuário usam **nomes** ("payment-api");
- o cenário C2 do bench de referência pede incidentes para `checkout`, `payment` e `catalog` — que **não estão** entre os 5 serviços do seed (`payment-api`, `auth-service`, `order-service`, `inventory-api`, `notification-worker`).

Com a regra "serviço existente", o C2 falha por construção. Opções: aceitar serviços fora do seed, trocar o C2 por serviços existentes ou deixar a rejeição como comportamento correto e ajustar o cenário. Qualquer uma é válida; precisa estar escrita na spec.

### 7. **Decisão:** `list_alerts` sem `all` e sem default

O nosso `contracts/tools.md` define `status: "firing" | "resolved"`. A referência aceita também `"all"` e usa `"firing"` como default via `z.preprocess`. Sem default, um modelo que chame `list_alerts({})` recebe erro de validação (que, se traduzido em texto, ele pode corrigir — mas custa uma chamada).

### 8. **Decisão:** o bench entra nesta spec?

`src/bench.ts` está na estrutura do `plan.md` e o script existe, mas não há tarefa para os cenários C1–C3. Ver [arena-e-bench.md](./arena-e-bench.md).

### 9. Limite de iterações sem perder o trace

A referência devolve `llmCalls: 0` e um trace vazio quando o ReAct estoura o `recursionLimit`. As nossas T013/T023 pedem contagem real e falha estruturada. Caminho sugerido: `agent.stream(...)` acumulando mensagens, para ter o trace parcial quando o `GraphRecursionError` chegar. Ver [react.md](./react.md).

---

## Depois da spec 001

1. Rodar `/speckit.specify` para a **reflection layer**, usando [reflection.md](./reflection.md) e a spec `002-reflection-layer` da referência como insumo.
2. Seguir `plan → tasks → implement` normalmente.
3. Estender a arena com `reflect:react` e `reflect:plan-and-execute`.

A partir daí o curso segue para a unidade 03 (Function Calling e Tool Use: `POST /chat`, store SQLite, tool de status de provedor e um servidor MCP de operações), que será o projeto `02` deste módulo.
