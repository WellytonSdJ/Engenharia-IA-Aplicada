# Arena e Bench — comparar e medir

## O que é

Ter três estratégias só vale a pena se der para responder "qual é melhor, para quê, a que custo?". A unidade entrega duas ferramentas de linha de comando para isso, com propósitos diferentes:

| | Arena | Bench |
| --- | --- | --- |
| Pergunta | "Como cada estratégia **raciocina** sobre este pedido?" | "Cada estratégia **acerta** estes cenários?" |
| Entrada | Um pedido livre (`--input`) | Cenários fixos (C1, C2, C3) |
| Saída | Trace + métricas + resposta, por estratégia | Tabela `cenário × estratégia` com acerto sim/não |
| Uso | Explorar, entender, depurar | Comparar de forma repetível |
| Critério | Você lê e julga | O código confere o **estado do store** |

A arena é o laboratório; o bench é a prova.

## Arena (código de referência)

### Uma estratégia, um store novo

```typescript
// src/arena.ts (referência)
export function createStrategy(name: StrategyName, maxIterations: number): ReasoningStrategy {
  const store = new InMemoryStore();
  seedStore(store);
  const tools = createTools(store);

  if (name === "react") {
    return new ReactStrategy({ modelFactory: createModel, tools, maxIterations });
  }
  if (name === "plan-and-execute") {
    return new PlanExecuteStrategy({ modelFactory: createModel, tools, maxIterations });
  }
  // reflect:react e reflect:plan-and-execute: mesma coisa, embrulhada em withReflection
  // ...
  const _exhaustive: never = name;
  throw new Error(`Unknown strategy: ${_exhaustive}`);
}
```

Cada estratégia recebe um store **recém-semeado**. Se o ReAct abrir um incidente, o Plan-and-Execute não vai vê-lo. É isso que o nosso contrato quer dizer com "resultados separados... sem misturar seus eventos" — e é possível graças às tools criadas por factory (veja [tools-e-store-deterministico.md](./tools-e-store-deterministico.md)).

O `const _exhaustive: never = name` é um truque de TypeScript: se alguém adicionar um nome novo ao tipo `StrategyName` e esquecer de tratá-lo acima, a atribuição a `never` **não compila**. O compilador vira o checklist.

### Parse de flags sem biblioteca

```typescript
export function parseArgs(argv: string[]): ArenaArgs {
  const args: Record<string, string> = {};
  const positional: string[] = [];
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      positional.push(token);
      continue;
    }
    args[token] = argv[i + 1] ?? "";
    i += 1;
  }
  // ... valida --strategies (obrigatória), --max-iterations (inteiro positivo), input
```

Validação na borda, **antes** de qualquer chamada de LLM: estratégia inválida ou `--max-iterations 0` fazem a CLI falhar sem gastar um token. Nosso contrato (`contracts/arena-cli.md`) pede o mesmo: "Argumentos inválidos devem produzir mensagem de erro na borda CLI e código de saída diferente de zero, sem iniciar chamadas de LLM". A constituição pede Zod nas fronteiras de CLI — no nosso projeto, esse `parseArgs` é um bom lugar para um schema Zod no lugar dos `if`s manuais.

### Falha de uma não derruba as outras

```typescript
for (const strategyName of args.strategies) {
  try {
    const strategy = createStrategy(strategyName, args.maxIterations);
    const result = await strategy.run(args.input);
    printResult(strategyName, result);
  } catch (error) {
    hadErrors = true;
    printHeader(strategyName);
    console.error(`ERROR: ${error instanceof Error ? error.message : String(error)}`);
    console.log();
  }
}

if (hadErrors) {
  process.exitCode = 1;
}
```

`process.exitCode = 1` (em vez de `process.exit(1)`) deixa o laço terminar e as outras estratégias imprimirem, mas ainda sinaliza falha para quem chamou (um script, um CI).

## Bench: acerto medido no estado

Aqui está a ideia mais importante da unidade. Olhe o cenário C2:

```typescript
// src/bench.ts (referência)
{
  id: "C2",
  label: "estruturado",
  prompt:
    "abra três incidentes sev2 para checkout, payment e catalog, nessa mesma ordem, e resolva o primeiro.",
  check(store) {
    const incidents = store.getIncidents().sort((a, b) => a.createdAt - b.createdAt);
    if (incidents.length !== 3) {
      return false;
    }
    const servicesMatch = C2_SERVICES.every(
      (service, index) => incidents[index]?.service.toLowerCase() === service,
    );
    const severitiesMatch = incidents.every((incident) => incident.severity === SEV2);
    const firstResolved = incidents[0]?.status === "resolved";
    const othersOpen = incidents.slice(1).every((incident) => incident.status === "open");
    return servicesMatch && severitiesMatch && firstResolved && othersOpen;
  },
```

O `check` do C2 **nem recebe a resposta em texto**. Ele abre o store e confere: existem exatamente 3 incidentes? Na ordem checkout → payment → catalog? Todos `high` (sev2)? O primeiro resolvido e os outros abertos?

Por que isso importa tanto? Porque um LLM pode escrever *"Pronto! Abri os três incidentes e resolvi o primeiro."* sem ter feito nada disso — ou tendo aberto quatro, ou na ordem errada, ou com severidade `medium`. Avaliar agente pelo texto é avaliar o relatório do estagiário sem olhar se o trabalho foi feito. Quando o agente **age**, o que conta é o **efeito**.

Os três cenários cobrem tipos diferentes de tarefa:

| Cenário | Tipo | O que o `check` confere |
| --- | --- | --- |
| C1 `direto` | Só leitura | Nenhum incidente criado **e** a resposta cita o número certo de alertas críticos firing (calculado do store: 2) |
| C2 `estruturado` | Várias escritas com ordem | Só o estado (3 incidentes, ordem, severidade, status) |
| C3 `dinâmico` | Leitura → decisão → escrita → conta | 1 incidente aberto para o serviço do alerta firing mais antigo **e** a resposta cita quantos sobraram |

Mesmo quando o texto importa (C1, C3), o **gabarito vem do store**, não de um número fixo no código:

```typescript
const expected = store
  .getAlerts("firing")
  .filter((alert) => alert.severity === "critical").length;
return store.getIncidents().length === 0 && answerMentionsCount(answer, expected);
```

E `answerMentionsCount` usa regex com lookbehind/lookahead (`(?<!\d)2(?!\d)`) para que "2" não case com "12" ou "2024".

### `diagnose`: por que errou

Cada cenário tem, além do `check`, um `diagnose` que **não muda o critério** — só explica o erro. Uma saída de "miss" fica mais ou menos assim (exemplo ilustrativo, montado a partir dos formatos de mensagem do `bench.ts`):

```
Running C2 / react... miss
  - serviços/ordem: [payment, checkout, catalog] (esperado: checkout, payment, catalog)
  - answer: Abri os três incidentes sev2 e resolvi o primeiro...
  - store: incidents=[inc-...:payment/high/resolved; ...] firing=[...]
```

Um "não" sozinho não ensina nada. Com o diagnóstico você vê **qual** das condições falhou e compara a resposta (o que o agente disse) com o snapshot do store (o que ele fez).

### `--no-replanner` e o custo do replanner

```typescript
return new PlanExecuteStrategy({
  modelFactory: createModel,
  tools,
  maxIterations,
  enableReplanner: !noReplanner,
});
```

Rodando `npm run bench` e depois `npm run bench -- --no-replanner`, você compara Plan-and-Execute com e sem replanner nos mesmos cenários: quantas chamadas o replanner adiciona e quantos acertos ele salva. É o tipo de pergunta que só o bench responde.

## O que o nosso plano pede

O nosso `tasks.md` foca na arena (T026–T028) e não detalha o bench — o `bench.ts` aparece na estrutura do `plan.md` e o script `npm run bench` já existe no `package.json`, mas não há tarefa específica para os cenários. Dá para seguir dois caminhos no `implement`: incluir o bench agora (acrescentando tarefas via `/speckit.converge` ou editando o `tasks.md`) ou deixá-lo para uma spec própria. Vale decidir antes; está anotado em [roteiro-de-implementacao.md](./roteiro-de-implementacao.md).

A nossa arena também tem duas diferenças de contrato em relação à referência:

- o input é **posicional** (`npm run arena -- --strategies react "liste os alertas firing"`), enquanto a referência aceita `--input` e usa o posicional como alternativa;
- a separação entre `src/controllers/arena-controller.ts` (seleção de estratégia e validação das flags) e `src/arena.ts` (entrypoint) — a referência junta tudo num arquivo só.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Contrato da arena (nosso) | [`specs/.../contracts/arena-cli.md`](../specs/001-nucleo-raciocinio-opspilot/contracts/arena-cli.md) | Flags, input posicional e falha sem chamar LLM |
| Requisitos | [`specs/.../spec.md`](../specs/001-nucleo-raciocinio-opspilot/spec.md) | US3, FR-014, SC-005 |
| Tarefas relacionadas | [`specs/.../tasks.md`](../specs/001-nucleo-raciocinio-opspilot/tasks.md) | T026, T027, T028, T035 |
| Arena (referência) | `src/arena.ts` do snapshot | Store por estratégia, `never` exaustivo, `process.exitCode` |
| Bench (referência) | `src/bench.ts` do snapshot | `check` no estado, `diagnose`, `--scenario`, `--no-replanner` |
