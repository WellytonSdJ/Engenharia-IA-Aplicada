# Roteiro de Implementação — do `tasks.md` ao código

Este doc registra as decisões tomadas durante a implementação da spec `001-nucleo-raciocinio-opspilot/`. Foi escrito antes do `/speckit.implement`, como roteiro de atenção fase a fase, e depois passou a funcionar como histórico: o que o plano previa, o que mudou na prática e quais armadilhas foram encontradas e corrigidas.

> Nada aqui altera os arquivos do projeto. Os pontos marcados como **decisão** são coisas para resolver na spec/plano (via `/speckit.clarify`, `/speckit.analyze` ou editando os artefatos) antes de implementar — não durante.

---

## Onde estamos

```text
✅ constituição (v1.0.1)
✅ /speckit.specify  → spec.md + checklists/requirements.md
✅ /speckit.plan     → plan.md, research.md, data-model.md, contracts/, quickstart.md
✅ /speckit.tasks    → tasks.md (todas as tarefas marcadas [X])
✅ /speckit.implement → src/ implementado; 29 testes passando na época, typecheck limpo
✅ spec 003 (fix-replanner-output): replanner pede uma correção quando a decisão vem contraditória
✅ spec 002 (reflection): withReflection + reflect:react / reflect:plan-and-execute na arena
✅ bench: src/bench.ts com C1/C2/C3, --scenario e --no-replanner (43 testes)
✅ primeira execução real da arena (depois de corrigir o carregamento do .env — armadilha 3)
```

> Este doc foi escrito **antes** da implementação. A partir daqui, o status de cada armadilha está anotado logo abaixo do título dela.

---

## Mapa da nossa estrutura

O nosso `plan.md` organiza `src/` em camadas (constituição, princípio II: Model → Service → Controller). Cada arquivo tem uma responsabilidade clara e um doc de estudo correspondente:

| Nosso arquivo (plan.md) | O que faz | Doc de estudo |
| --- | --- | --- |
| `src/models/domain.ts` | Tipos de domínio e schemas Zod | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/services/errors.ts` | Erros de domínio | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/models/in-memory-store.ts` | Store em memória, cópias rasas | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/models/sequelize-store.ts` | Adapter Sequelize (não usado nesta fase) | — |
| `src/services/seed.ts` | Carga e validação do seed | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/services/tools.ts` | Factory de tools com store injetado | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) |
| `src/agents/model.ts` | Fábrica do modelo (OpenRouter via `ChatOpenAI`) | [react.md](./react.md) |
| `src/agents/trace.ts` | Construção do trace a partir das mensagens | [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md) |
| `src/agents/strategy.ts` | Contrato `ReasoningStrategy`/`StrategyResult` | [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md) |
| `src/agents/react.ts` | Estratégia ReAct | [react.md](./react.md) |
| `src/agents/plan-and-execute.ts` | Estratégia plan-and-execute | [plan-and-execute.md](./plan-and-execute.md) |
| `src/controllers/arena-controller.ts` + `src/arena.ts` | Execução comparativa das estratégias | [arena-e-bench.md](./arena-e-bench.md) |
| `src/controllers/bench-controller.ts` + `src/bench.ts` | Cenários C1–C3 com verificação automática | [arena-e-bench.md](./arena-e-bench.md) |
| `src/agents/reflection.ts` (spec 002) | Decorator `withReflection` | [reflection.md](./reflection.md) |

---

## Fase a fase

| Fase do `tasks.md` | O que estudar antes | Atenção |
| --- | --- | --- |
| **1. Setup** (T001–T003) | [spec-driven-com-spec-kit.md](./spec-driven-com-spec-kit.md) | O script `seed` é novo |
| **2. Foundation** (T004–T012) | Contrato/trace, tools/store | Nosso `TraceEvent` é união discriminada com Zod — validação rígida de propósito |
| **3. US1** (T013–T015) | [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md) | Contagem de `llmCalls` tem de ser real em todos os caminhos, inclusive no estouro de limite |
| **4. US2** (T016–T020) | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) | Seed idempotente; resolver duas vezes deve dar erro |
| **5. US3** (T021–T028) | [react.md](./react.md), [plan-and-execute.md](./plan-and-execute.md), [arena-e-bench.md](./arena-e-bench.md) | Testes com model double: veja o padrão `mockBase`/`sequenceCritic` em [reflection.md](./reflection.md) |
| **6. US4** (T029–T031) | Testes do store e do trace | "Sem rede" precisa ser **provado** (T029), não só presumido |
| **7. Polish** (T032–T036) | `quickstart.md` | Só atualizar o quickstart com comportamento verificado |

---

## Armadilhas já visíveis (verificadas nos arquivos atuais)

Estas não eram opiniões sobre o futuro código: eram inconsistências presentes, **antes da implementação**, nos arquivos de configuração e nas specs. Na época, nenhuma foi corrigida (a regra era não mexer em configuração nem em arquivos que os agentes leem). Elas ficaram aqui para você decidir. O status atual de cada uma vem logo abaixo do título.

| # | Armadilha | Status após a implementação |
| --- | --- | --- |
| 1 | Driver `mysql` × `mysql2` | ✅ resolvida (`mysql2` adicionado) |
| 2 | `@types/sequelize` desnecessário | ⚠️ ainda está no `package.json`, mas o typecheck passa |
| 3 | Scripts não carregam o `.env` | ✅ resolvida: ver [variaveis-de-ambiente-com-env-file.md](./variaveis-de-ambiente-com-env-file.md) |
| 4 | `OPENROUTER_MODEL` vazio | ✅ resolvida: `createModel` usa `!model`, que barra string vazia |
| 5 | `resolveJsonModule` | ✅ não se aplica: o store lê o JSON com `readFile` (`src/models/json-store.ts`) |
| 6 | `service` por nome × `serviceId` | ✅ resolvida no bench: `catalog` entrou no seed, o check do C2 compara `serviceId` e o erro de serviço inexistente lista os nomes válidos. Ver [arena-e-bench.md](./arena-e-bench.md#decisões-que-o-c2-forçou) |
| 7 | `list_alerts` sem `all`/default | ✅ resolvida: `status` aceita `all` e tem default `firing` |
| 8 | Bench nesta spec? | ✅ implementado fora do `tasks.md`: `src/bench.ts` + `src/controllers/bench-controller.ts` |
| 9 | Limite de iterações sem perder o trace | ✅ tratado em `src/agents/react.ts` (captura `GraphRecursionError`) |

### 1. Driver do MySQL: `mysql` em vez de `mysql2`

O nosso `package.json` declarava `"mysql": "^2.18.1"`. O dialeto `mysql` do Sequelize 6 usa o pacote **`mysql2`** como driver — com só o `mysql` instalado, criar uma instância `new Sequelize(..., { dialect: "mysql" })` falha pedindo para instalar o `mysql2`. A correção foi adicionar `"mysql2": "^3.11.0"` às dependências. Afeta T008 (adapter Sequelize).

### 2. `@types/sequelize` é desnecessário

O Sequelize 6 já traz os próprios tipos. O pacote `@types/sequelize` é de uma era anterior (v4) e pode gerar conflito de tipos no `npm run typecheck`. Não é necessário instalá-lo.

### 3. Os scripts `arena` e `bench` não carregam o `.env`

```json
"arena": "tsx src/arena.ts",
"bench": "tsx src/bench.ts",
```

O `tsx` não lê `.env` sozinho, e a constituição proíbe `dotenv` ("carregue variáveis de ambiente usando o suporte nativo do Node.js"). A correção aplicada foi assim:

```json
"arena": "node --env-file-if-exists=.env --import tsx src/arena.ts",
"bench": "node --env-file-if-exists=.env --import tsx src/bench.ts",
```

Sem isso, `OPENROUTER_API_KEY` não chega ao processo e a fábrica de modelo lança erro na primeira execução real da arena (T035).

> **Aconteceu exatamente isso.** A implementação preservou os scripts antigos, e a primeira execução real da arena falhou com `OPENROUTER_API_KEY and OPENROUTER_MODEL are required`, mesmo com o `.env` preenchido. A correção aplicada foi a que está acima, estendida para `dev` e `seed`. O `test` ficou de fora de propósito. Explicação completa, outros usos e passo a passo do zero: [variaveis-de-ambiente-com-env-file.md](./variaveis-de-ambiente-com-env-file.md).

### 4. `OPENROUTER_MODEL` vazio não cai no default

O `.env.example` traz `OPENROUTER_MODEL=` (vazio). Um `process.env.OPENROUTER_MODEL ?? "openai/gpt-4o-mini"` ingênuo não resolve isso: `??` só troca `null`/`undefined`; uma **string vazia** passa direto e o modelo fica `""`. Se copiar o `.env.example` sem preencher, o erro vai aparecer no provedor, longe da causa. A correção foi validar com `!model` na fábrica (T011), que barra string vazia junto com `null`/`undefined`.

### 5. `resolveJsonModule` ausente no `tsconfig.json`

Se o seed fosse importado como JSON (`import seedData from "./seed-data.json" with { type: "json" };`), o `tsconfig.json` precisaria de `"resolveJsonModule": true`. Como o seed lê o arquivo em runtime com `readFile` (`src/models/json-store.ts`), essa opção não é necessária.

### 6. **Decisão:** `service` por nome ou `serviceId`?

O `data-model.md` define `Alert.serviceId` e diz que `open_incident` "exige título, **serviço existente**". Mas:

- os pedidos do usuário usam **nomes** ("payment-api");
- o cenário C2 do bench pede incidentes para `checkout`, `payment` e `catalog` — que **não estão** entre os 5 serviços do seed (`payment-api`, `auth-service`, `order-service`, `inventory-api`, `notification-worker`).

Com a regra "serviço existente", o C2 falha por construção. Opções: aceitar serviços fora do seed, trocar o C2 por serviços existentes ou deixar a rejeição como comportamento correto e ajustar o cenário. Qualquer uma é válida; precisa estar escrita na spec.

### 7. **Decisão:** `list_alerts` sem `all` e sem default

O nosso `contracts/tools.md` define `status: "firing" | "resolved"`. Sem um valor `"all"` e sem default, um modelo que chame `list_alerts({})` recebe erro de validação (que, se traduzido em texto, ele pode corrigir — mas custa uma chamada). A correção adotada foi aceitar também `"all"` e usar `"firing"` como default via `z.preprocess`.

### 8. **Decisão:** o bench entra nesta spec?

`src/bench.ts` está na estrutura do `plan.md` e o script existe, mas não há tarefa para os cenários C1–C3. Ver [arena-e-bench.md](./arena-e-bench.md).

### 9. Limite de iterações sem perder o trace

Um `agent.invoke(...)` simples devolve `llmCalls: 0` e um trace vazio quando o ReAct estoura o `recursionLimit`, porque a exceção interrompe antes de qualquer resultado. As nossas T013/T023 pedem contagem real e falha estruturada. A solução foi usar `agent.stream(...)` acumulando mensagens, para ter o trace parcial quando o `GraphRecursionError` chegar. Ver [react.md](./react.md).

---

## Depois da spec 001

1. Rodar `/speckit.specify` para a **reflection layer**, usando [reflection.md](./reflection.md) como insumo.
2. Seguir `plan → tasks → implement` normalmente.
3. Estender a arena com `reflect:react` e `reflect:plan-and-execute`.

O passo seguinte do roteiro interno é a unidade de Function Calling e Tool Use (`POST /chat`, store SQLite, tool de status de provedor e um servidor MCP de operações), que corresponde ao projeto `02` deste módulo.

> **Atualização:** o `POST /chat` foi planejado antecipadamente neste projeto como a spec [`004-http-chat-endpoint`](../specs/004-http-chat-endpoint/spec.md) (Express 5, registro de estratégias em `src/agents/index.ts`, `reflect` no corpo, timeout de 60 s) — `specify → plan → tasks` concluídos, mas a implementação (`/speckit.implement`) não chegou a rodar neste projeto. Ela acontece no projeto `02-function-calling-e-tool-use`, junto com o store SQLite, a tool de status de provedor e o servidor MCP.
