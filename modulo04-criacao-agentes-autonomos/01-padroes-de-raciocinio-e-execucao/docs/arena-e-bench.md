# Arena e Bench — comparar e medir

## O que é

Ter três estratégias só vale a pena se der para responder "qual é melhor, para quê, a que custo?". A unidade entrega duas ferramentas de linha de comando para isso, com propósitos diferentes:

| | Arena | Bench |
| --- | --- | --- |
| Pergunta | "Como cada estratégia **raciocina** sobre este pedido?" | "Cada estratégia **acerta** estes cenários?" |
| Entrada | Um pedido livre (input posicional) | Cenários fixos (C1, C2, C3) |
| Saída | Trace + métricas + resposta, por estratégia | Tabela `cenário × estratégia` com acerto sim/não |
| Uso | Explorar, entender, depurar | Comparar de forma repetível |
| Critério | Você lê e julga | O código confere o **estado do store** |

A arena é o laboratório; o bench é a prova.

## Arena

### Um registro de estratégias, um store por chamada

```typescript
// src/arena.ts
export const arenaStrategies: Readonly<Record<string, ReasoningStrategy>> = {
  react: reactStrategy,
  "plan-and-execute": planAndExecuteStrategy,
  "reflect:react": withReflection(reactStrategy),
  "reflect:plan-and-execute": withReflection(planAndExecuteStrategy),
};

export async function main(args = process.argv.slice(2)): Promise<void> {
  const runs = await runArena(args, arenaStrategies);
  console.log(formatArenaRuns(runs));
}
```

Não há uma `createStrategy(name)` com `if`/`switch` por nome: as quatro estratégias (incluindo as duas versões com Reflection) já existem como um objeto estático, montado uma vez no módulo. Quem resolve o nome pedido pela CLI é `runArena`, em `src/controllers/arena-controller.ts`:

```typescript
// src/controllers/arena-controller.ts
export async function runArena(
  args: readonly string[],
  strategyRegistry: Readonly<Record<string, ReasoningStrategy>>,
): Promise<ArenaRun[]> {
  const parsed = parseArenaArgs(args);
  const selected = parsed.strategies.map((name) => {
    const strategy = strategyRegistry[name];
    if (!strategy) {
      throw new Error(`Unknown strategy "${name}". Available strategies: ${Object.keys(strategyRegistry).join(", ")}.`);
    }
    return strategy;
  });
  const results: ArenaRun[] = [];
  const options: ReasoningOptions = { maxIterations: parsed.maxIterations };
  for (const strategy of selected) {
    const result = ReasoningResultSchema.parse(await strategy.run(parsed.input, options));
    results.push({ strategy: strategy.name, ...result });
  }
  return results;
}
```

Cada `strategy.run(parsed.input, options)` não recebe um store próprio aqui — sem `options.store`, ReAct e Plan-and-Execute caem no `defaultOpsStore` (o `JsonOpsStore` persistido em `src/data/ops-store.json`). Isso é o oposto do bench, que **sempre** passa um `InMemoryOpsStore` novo por célula (veja mais abaixo) justamente para impedir que uma estratégia contamine a outra; na arena, o objetivo é explorar o comportamento, não comparar acertos isolados.

### Parse de flags sem biblioteca

```typescript
// src/controllers/arena-controller.ts
export function parseArenaArgs(args: readonly string[]): ArenaArguments {
  let strategyList = "react,plan-and-execute";
  let maxIterations = 8;
  const positional: string[] = [];

  for (let index = 0; index < args.length; index += 1) {
    const value = args[index];
    if (value === "--strategies" || value === "--max-iterations") {
      const next = args[index + 1];
      if (!next || next.startsWith("--")) {
        throw new Error(`Missing value for ${value}.`);
      }
      // ...
    } else if (value.startsWith("--")) {
      throw new Error(`Unknown option "${value}".`);
    } else {
      positional.push(value);
    }
  }

  const parsed = ArenaInputSchema.parse({
    strategies: strategyList.split(",").map((name) => name.trim()),
    maxIterations,
    input: positional.join(" "),
  });
  return { strategies: parsed.strategies, maxIterations: parsed.maxIterations, input: parsed.input };
}
```

Validação na borda, **antes** de qualquer chamada de LLM: a validação final passa por `ArenaInputSchema` (um schema Zod com `strategies`, `maxIterations` e `input`), então uma lista de estratégias vazia ou um `input` vazio falham ali, sem gastar um token. Isso cumpre o mesmo contrato descrito em `contracts/arena-cli.md`: "argumentos inválidos devem produzir mensagem de erro na borda CLI e código de saída diferente de zero, sem iniciar chamadas de LLM".

### Falha não é isolada por estratégia

Diferente do que se poderia esperar de uma arena que compara várias estratégias lado a lado, `runArena` não executa cada estratégia dentro de um `try`/`catch` próprio — um erro em qualquer uma (nome desconhecido ou falha durante `run`) propaga e interrompe o laço inteiro. É o `main` do `src/arena.ts` que captura a falha final e sinaliza a CLI:

```typescript
// src/arena.ts
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error: unknown) => {
    console.error("Arena failed:", error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
```

`process.exitCode = 1` (em vez de `process.exit(1)`) ainda sinaliza falha para quem chamou (um script, um CI) sem matar o processo imediatamente — mas aqui ele é usado para a falha do `main()` como um todo, não para permitir que as estratégias seguintes de uma mesma chamada continuem depois de uma falhar.

## Bench: acerto medido no estado

Aqui está a ideia mais importante da unidade. Olhe o cenário C2:

```typescript
// src/controllers/bench-controller.ts
{
  id: "C2",
  label: "estruturado",
  prompt: "abra três incidentes de sev2 para checkout, payment e catalog, nessa mesma ordem e resolva o primeiro.",
  async check(store) {
    // Tudo verificado no store: quantidade, ordem de criação, severidade e status.
    const reasons: string[] = [];
    const incidents = await store.listIncidents("all");
    const serviceIds = incidents.map((incident) => incident.serviceId);
    if (incidents.length !== 3) reasons.push(`esperava 3 incidentes, veio ${incidents.length}`);
    if (C2_EXPECTED_SERVICE_IDS.some((id, index) => serviceIds[index] !== id)) {
      reasons.push(`serviços/ordem: [${serviceIds.join(", ")}] (esperado: ${C2_EXPECTED_SERVICE_IDS.join(", ")})`);
    }
    const wrongSeverity = incidents.filter((incident) => incident.severity !== SEV2);
    if (wrongSeverity.length > 0) {
      reasons.push(`severidade diferente de ${SEV2} (sev2): ${wrongSeverity.map((item) => `${item.service.name}=${item.severity}`).join(", ")}`);
    }
    if (incidents[0] && incidents[0].status !== "resolved") reasons.push(`primeiro incidente está ${incidents[0].status} (esperado resolved)`);
    const notOpen = incidents.slice(1).filter((incident) => incident.status !== "open");
    if (notOpen.length > 0) reasons.push(`deveriam continuar open: ${notOpen.map((item) => item.service.name).join(", ")}`);
    return reasons;
  },
},
```

O `check` do C2 **nem recebe a resposta em texto** (o parâmetro `answer` nem é declarado nessa função). Ele abre o store e confere: existem exatamente 3 incidentes? Na ordem checkout → payment → catalog (`C2_EXPECTED_SERVICE_IDS`)? Todos `high` (sev2)? O primeiro resolvido e os outros abertos? Note que `check` devolve uma **lista de motivos** (vazia = acerto), não um booleano — é essa lista, explicada mais abaixo, que serve de diagnóstico quando o cenário falha.

Por que isso importa tanto? Porque um LLM pode escrever *"Pronto! Abri os três incidentes e resolvi o primeiro."* sem ter feito nada disso — ou tendo aberto quatro, ou na ordem errada, ou com severidade `medium`. Avaliar agente pelo texto é avaliar o relatório do estagiário sem olhar se o trabalho foi feito. Quando o agente **age**, o que conta é o **efeito**.

Os três cenários cobrem tipos diferentes de tarefa:

| Cenário | Tipo | O que o `check` confere |
| --- | --- | --- |
| C1 `direto` | Só leitura | Nenhum incidente criado **e** a resposta cita o número certo de alertas críticos firing (calculado do store: 1) |
| C2 `estruturado` | Várias escritas com ordem | Só o estado (3 incidentes, ordem, severidade, status) |
| C3 `dinâmico` | Leitura → decisão → escrita → conta | 1 incidente aberto para o serviço do alerta firing mais antigo **e** a resposta cita quantos sobraram |

Mesmo quando o texto importa (C1, C3), o **gabarito vem do store**, não de um número fixo no código:

```typescript
// src/controllers/bench-controller.ts — check do C1
async check(store, answer) {
  const reasons: string[] = [];
  const incidents = await store.listIncidents("all");
  const expected = (await store.listAlerts("firing")).filter((alert) => alert.severity === "critical").length;
  if (incidents.length !== 0) reasons.push(`esperava 0 incidentes (pergunta só de leitura), veio ${incidents.length}`);
  if (!answerMentionsCount(answer, expected)) reasons.push(`resposta não cita a contagem esperada (${expected})`);
  return reasons;
},
```

E `answerMentionsCount` usa regex com lookbehind/lookahead (`(?<!\d)${count}(?!\d)`) para que, por exemplo, "2" não case com "12" ou "2024":

```typescript
// src/controllers/bench-controller.ts
export function answerMentionsCount(answer: string, count: number): boolean {
  const text = answer.toLowerCase();
  if (new RegExp(`(?<!\\d)${count}(?!\\d)`).test(text)) {
    return true;
  }
  return (NUMBER_WORDS[count] ?? []).some((word) => new RegExp(`(?<![\\p{L}])${word}(?![\\p{L}])`, "u").test(text));
}
```

### Diagnóstico: por que errou

Cada cenário devolve, do próprio `check`, uma **lista de motivos** (`reasons`) — vazia significa acerto. Não existe uma função `diagnose` separada: a mesma chamada que decide se a célula acertou já explica o porquê quando não acerta. Uma saída real de "miss" se parece com isto:

```text
Por que errou:
C2 / plan-and-execute:
  - erro: [ { "code": "invalid_type", "expected": "object", "received": "undefined", ... } ]
```

ou, quando o problema é o estado em si (não um erro lançado):

```text
C2 / sloppy:
  - esperava 3 incidentes, veio 2
```

Um "não" sozinho não ensina nada. Com a lista de motivos você vê **qual** das condições falhou — e `formatBenchMisses` (`src/controllers/bench-controller.ts`) é só quem formata essas listas na saída final do CLI.

### `--no-replanner` e o custo do replanner

```typescript
// src/bench.ts
export function benchStrategies(noReplanner: boolean): ReasoningStrategy[] {
  return [reactStrategy, createPlanAndExecuteStrategy({ replanner: !noReplanner })];
}
```

Rodando `npm run bench` e depois `npm run bench -- --no-replanner`, você compara Plan-and-Execute com e sem replanner nos mesmos cenários: quantas chamadas o replanner adiciona e quantos acertos ele salva. É o tipo de pergunta que só o bench responde.

## O nosso bench (implementado)

O bench saiu do papel fora do `tasks.md` da spec 001 (era a armadilha 8 do [roteiro](./roteiro-de-implementacao.md)). Ele segue a mesma divisão da arena:

```text
src/bench.ts                        → entrypoint: escolhe as estratégias e imprime
src/controllers/bench-controller.ts → cenários, check no estado, flags, execução e tabela
src/bench.test.ts                   → testes sem rede com estratégias falsas
```

### Rodando

```powershell
npm run bench                                  # C1, C2, C3 × react, plan-and-execute
npm run bench -- --scenario C2                 # um cenário (aceita c2, ou lista: c1,c3)
npm run bench -- --no-replanner                # plan-and-execute vira plan-and-execute:no-replanner
npm run bench -- --scenario c1 --max-iterations 4
```

A saída tem três blocos: uma linha por célula enquanto roda, a tabela consolidada e, no fim, **por que** cada erro aconteceu (a lista de motivos do `check`, formatada por `formatBenchMisses`):

```text
| cenário | estratégia       | acerto | llmCalls | latencyMs |
|---------|------------------|--------|----------|-----------|
| C1      | react            | sim    | 2        | 3566      |
| C1      | plan-and-execute | não    | 1        | 1767      |
| C2      | react            | sim    | 6        | 13626     |
| C2      | plan-and-execute | não    | -        | 3028      |
| C3      | react            | sim    | 3        | 14103     |
| C3      | plan-and-execute | não    | -        | 20696     |

Por que errou:
C1 / plan-and-execute:
  - resposta não cita a contagem esperada (1)
C2 / plan-and-execute:
  - erro: [ { "code": "invalid_type", "expected": "object", "received": "undefined", ... } ]
C3 / plan-and-execute:
  - erro: A continue decision cannot include replacement steps or an answer.
```

Essa é uma execução real com `OPENROUTER_MODEL=openrouter/free` (out/2026). Leia assim: `llmCalls` igual a `-` significa que a estratégia **lançou erro** e não devolveu métricas, e a latência é o tempo até o erro. `C1 / plan-and-execute` com `llmCalls: 1` mostra que o planner respondeu sem chamar nenhuma tool. O número parece certo de longe, mas o check pegou o erro.

### Como cada cenário é julgado

| Cenário | Pedido | O check confere no store | E na resposta |
| --- | --- | --- | --- |
| C1 direto | quantos alertas críticos estão disparando? | **nenhum** incidente criado (é só leitura) | cita a contagem real (1) |
| C2 estruturado | abra três incidentes de sev2 para checkout, payment e catalog, nessa ordem, e resolva o primeiro | 3 incidentes, na ordem `svc-checkout`, `svc-payments`, `svc-catalog`, todos `high`, o 1º `resolved` e os outros `open` | — |
| C3 dinâmico | dos alertas disparando, abra um incidente para o mais antigo e diga quantos sobraram | exatamente 1 incidente, `open`, no serviço do alerta firing mais antigo (`api`) | cita quantos sobraram (2) |

Em C1 e C3, o pedido é **uma pergunta**, e a resposta é parte do resultado. Mesmo assim, o número esperado vem do store, não fica fixo no código: `answerMentionsCount` aceita o dígito isolado (não casa o "1" de "10") ou o número por extenso (`duas`, `três`...). "um/one" fica de fora porque aparece em qualquer frase ("abri um incidente").

### Decisões que o C2 forçou

O C2 pede incidentes para `payment` e `catalog`, mas o store **rejeita serviço inexistente** — e o seed original não tinha um serviço `catalog`, e o serviço de pagamentos se chama `payments`, não `payment`. Isso exigiu algumas decisões deliberadas:

1. **`catalog` entrou no seed** (`DEFAULT_SERVICES`, sem alertas), que agora tem 6 serviços.
2. **O check compara `serviceId`, não o texto.** "payment" do usuário e `payments` do seed apontam para `svc-payments`.
3. **A mensagem de erro ensina o nome certo.** `Service payment was not found. Known services: api, checkout, payments, ...`. O ReAct recebe esse erro como observação e tenta de novo com `payments`. Foi assim que ele acertou o C2 acima (6 chamadas).
4. **`sev2 = high` está na descrição da tool `open_incident`** (`sev1=critical, sev2=high, sev3=medium, sev4=low`). O domínio só conhece `low/medium/high/critical`, e o bench espera `high`.

### O custo do replanner, em chamadas

Sem replanner, o grafo vira `planner → executor → executor → ... → finalizer`. O `finalizer` faz uma chamada final que redige a resposta a partir dos resultados. Num plano de N passos, isso dá `1 + N + 1` chamadas, contra `1 + 2N` com replanner (cada passo ganha uma chamada extra do nó `replanner`). O teste `without replanner the graph runs every planned step...` (`src/agents/plan-and-execute.test.ts`) fixa essa conta: 4 chamadas sem replanner contra 5 com replanner, para um plano de 2 passos.

### Por que um store novo por célula

```ts
// src/controllers/bench-controller.ts
// Store novo (seed limpo) por célula: o C2 de uma estratégia não pode contaminar a outra.
const store = createStore();
```

O padrão é `new InMemoryOpsStore()`. O bench **nunca** toca `src/data/ops-store.json`, então pode rodar quantas vezes quiser sem `npm run seed`.

### Limite da conta gratuita

Cada `npm run bench` completo faz de 20 a 40 chamadas. Com `openrouter/free`, a cota diária acaba rápido, e as células passam a falhar com `429 Rate limit exceeded: free-models-per-day`. Isso é erro de cota, não da estratégia. Use `--scenario` para rodar menos células enquanto estuda.

### Decisões de contrato da nossa arena

A nossa arena fixou duas decisões de contrato:

- o input é **posicional** (`npm run arena -- --strategies react "liste os alertas firing"`), sem uma flag `--input` equivalente;
- a separação entre `src/controllers/arena-controller.ts` (seleção de estratégia e validação das flags) e `src/arena.ts` (entrypoint), em vez de um único arquivo concentrando tudo.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Contrato da arena (nosso) | [`specs/.../contracts/arena-cli.md`](../specs/001-nucleo-raciocinio-opspilot/contracts/arena-cli.md) | Flags, input posicional e falha sem chamar LLM |
| Requisitos | [`specs/.../spec.md`](../specs/001-nucleo-raciocinio-opspilot/spec.md) | US3, FR-014, SC-005 |
| Tarefas relacionadas | [`specs/.../tasks.md`](../specs/001-nucleo-raciocinio-opspilot/tasks.md) | T026, T027, T028, T035 |
| Arena (nosso) | [`src/arena.ts`](../src/arena.ts) | Registro `arenaStrategies`, `process.exitCode` |
| Controller da arena (nosso) | [`src/controllers/arena-controller.ts`](../src/controllers/arena-controller.ts) | `parseArenaArgs`, `ArenaInputSchema`, `runArena` sem isolamento de falha por estratégia |
| Bench (nosso) | [`src/controllers/bench-controller.ts`](../src/controllers/bench-controller.ts) | `BENCH_SCENARIOS` com `check` que devolve motivos, `answerMentionsCount`, store por célula |
| Entrypoint do bench | [`src/bench.ts`](../src/bench.ts) | `benchStrategies(noReplanner)`, `check` no estado, `--scenario`, `--no-replanner` |
| Testes do bench | [`src/bench.test.ts`](../src/bench.test.ts) | Estratégia "sloppy": a resposta parece certa e o estado reprova |
| Modo sem replanner | [`src/agents/plan-and-execute.ts`](../src/agents/plan-and-execute.ts) | `createPlanAndExecuteStrategy({ replanner })`, nó `finalizer` |
