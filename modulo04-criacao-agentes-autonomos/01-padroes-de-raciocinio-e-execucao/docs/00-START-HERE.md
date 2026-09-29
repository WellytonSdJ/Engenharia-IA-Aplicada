# Por onde começar

Este é o primeiro projeto do módulo 04. O último projeto que estudamos foi o [`07-api-security-auth-rate-limiting`](../../../modulo03-mcp-na-pratica/07-api-security-auth-rate-limiting/customers-mcp/docs/00-START-HERE.md), do módulo 03 — lá o foco era **proteger** as ferramentas que um agente usa (JWT, RBAC, rate limiting). Aqui a pergunta muda de lado: em vez de "o que o agente pode acessar?", passa a ser **"como o agente pensa?"**.

---

## O que estamos construindo e por quê

> O OpsPilot é um copiloto de plantão. A pessoa de plantão escreve "quais serviços têm alertas ativos?" ou "abra três incidentes sev2 para checkout, payment e catalog e resolva o primeiro" — e o agente decide sozinho quais ferramentas chamar, em que ordem, até resolver.

No módulo 03, em [`01-multiple-mcp-tools`](../../../modulo03-mcp-na-pratica/01-multiple-mcp-tools/docs/agente-autonomo-vs-orquestracao-explicita.md), você já viu um agente autônomo com `createAgent`: um loop de "pensar → chamar tool → observar" escondido dentro de uma única chamada. Funcionava — mas era **uma caixa-preta**, e só havia **um jeito** de raciocinar.

Este projeto abre a caixa e coloca três jeitos diferentes de raciocinar lado a lado:

```
Módulo 03 (01-multiple-mcp-tools):        Este projeto:
um agente, um padrão (loop implícito)     três padrões atrás da MESMA interface
resultado = texto final                   resultado = answer + trace tipado + métricas
"funcionou?" = ler a resposta             "funcionou?" = conferir o estado do store (bench)
```

| Padrão | Ideia em uma frase |
| --- | --- |
| **ReAct** | Pensa um pouco, age, observa, repete — decide o próximo passo a cada volta |
| **Plan-and-Execute** | Primeiro escreve um plano completo, depois executa passo a passo, e um replanner revisa o plano depois de cada passo |
| **Reflection** | Um crítico avalia a resposta de *qualquer* estratégia e manda refazer com feedback se não estiver boa |

O ponto central da unidade é **comparar**: qual padrão acerta mais, em que tipo de pedido, gastando quantas chamadas de LLM e quanto tempo. Por isso tudo gira em torno de um contrato comum (`ReasoningStrategy`), um trace tipado e duas ferramentas de medição (arena e bench).

E tem uma novidade de **processo**: o projeto é conduzido com **Spec Kit** — spec, plano e tarefas escritos (e versionados) antes do código. Já fizemos essas três etapas; a próxima é `/speckit.implement`.

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [spec-driven-com-spec-kit.md](./spec-driven-com-spec-kit.md) | Entender o processo que já seguimos (e o que o `implement` vai fazer) antes de olhar código |
| 2 | [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md) | A interface que **todas** as estratégias implementam — base de tudo o que vem depois |
| 3 | [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) | As mãos do agente: tools com Zod sobre um store em memória, testáveis sem rede |
| 4 | [react.md](./react.md) | O primeiro padrão de raciocínio, o mais simples |
| 5 | [plan-and-execute.md](./plan-and-execute.md) | O segundo padrão, com um `StateGraph` próprio de três nós |
| 6 | [reflection.md](./reflection.md) | A camada que envolve as outras duas (spec 002) |
| 7 | [arena-e-bench.md](./arena-e-bench.md) | Como comparar as estratégias e medir acerto de verdade |
| 8 | [roteiro-de-implementacao.md](./roteiro-de-implementacao.md) | O mapa prático para rodar o `/speckit.implement` no nosso projeto |
| 9 | [glossario.md](./glossario.md) | Referência rápida |

---

## Mapa do código

O nosso projeto ainda não tem código além do `src/index.ts` inicial. O mapa abaixo é do **snapshot de referência do curso** (`02-padroes-de-raciocinio-e-execucao/`), em ordem lógica de leitura. O nosso `plan.md` reorganiza esses arquivos em outra estrutura de pastas — a tabela de correspondência está em [roteiro-de-implementacao.md](./roteiro-de-implementacao.md).

```
src/domain/types.ts            → tipos do domínio: Alert, Incident, TraceEvent, StrategyResult,
                                 ReasoningStrategy e IStore — o contrato de tudo
src/domain/errors.ts           → IncidentNotFoundError (erro de domínio)
src/store/seed-data.json       → 5 serviços e 6 alertas (3 firing, 3 resolved) do Mercadinho
src/store/seed.ts              → valida o JSON com Zod e popula o store; também roda como script
src/store/in-memory-store.ts   → InMemoryStore: getAlerts, createIncident, resolveIncident, getIncidents
src/agents/model.ts            → createModel(): fábrica única de ChatOpenAI → OpenRouter, temperature 0
src/llm/factory.ts             → re-export de createModel
src/agents/tools.ts            → list_alerts, open_incident, resolve_incident com tool() + Zod
src/tools/*.ts                 → re-exports de cada tool (index, list-alerts, open-incident, resolve-incident)
src/trace/builder.ts           → buildTraceFromMessages (mensagens → TraceEvent[]) e buildPlanExecuteTrace
src/agents/react.ts            → ReactStrategy (createReactAgent + recursionLimit)
src/strategies/react.ts        → re-export de ReactStrategy
src/strategies/plan-execute.ts → PlanExecuteStrategy (StateGraph planner/executor/replanner)
src/agents/plan-execute.ts     → re-export de PlanExecuteStrategy
src/strategies/reflect.ts      → withReflection (decorator), createLLMCritic, enrichInputWithFeedback
src/arena.ts                   → CLI que roda N estratégias sobre o mesmo input e imprime trace + métricas
src/bench.ts                   → CLI com cenários C1/C2/C3 e checagem de acerto no estado do store
src/index.ts                   → bootstrapOpsPilot(): monta store + tools + estratégias
src/store/in-memory-store.test.ts → testes do store (seed 6/3/3, ids únicos, transição, erro)
src/trace/builder.test.ts         → testes do trace a partir de mensagens fabricadas à mão
src/strategies/reflect.test.ts    → testes da reflection com estratégia e crítico mockados
```

> Repare nos re-exports (`src/llm/factory.ts`, `src/tools/*.ts`, `src/strategies/react.ts`...): o snapshot tem arquivos que só apontam para outros. Isso é resíduo da evolução do projeto em aula — a spec pedia uma estrutura de pastas e o código foi nascendo em outra. Não precisamos copiar isso; nosso `plan.md` já define uma estrutura única.

---

## O fluxo em uma linha

```
pedido (texto)
  → estratégia.run(input)                      ← ReAct | Plan-and-Execute | reflect:<qualquer uma>
    → LLM (via createModel) decide tool calls
      → tool (Zod valida args) → InMemoryStore muda estado
    → trace tipado + métricas
  → { answer, trace, metrics }
  → arena imprime lado a lado  |  bench confere o STORE e diz "sim/não"
```

---

## Como rodar e ver o que importa

Os comandos abaixo são os do snapshot de referência — depois do `/speckit.implement`, os nossos devem ficar equivalentes (veja as diferenças de script em [roteiro-de-implementacao.md](./roteiro-de-implementacao.md)).

```bash
# 1. Testes determinísticos — não precisam de chave nem de rede
npm test

# 2. Mesma pergunta, duas estratégias, lado a lado (precisa de OPENROUTER_API_KEY no .env)
npm run arena -- --strategies react,plan-and-execute --input "Quais serviços têm alertas ativos?"

# 3. Acerto medido no estado: 3 cenários × 2 estratégias
npm run bench
```

O comando que mais ensina é o 2: olhe o trace de cada estratégia e conte quantas vezes o LLM foi chamado. Depois rode o 3 e veja que "a resposta parece boa" e "o store ficou certo" são coisas diferentes.
