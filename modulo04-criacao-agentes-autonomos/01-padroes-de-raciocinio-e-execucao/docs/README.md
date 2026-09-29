# Documentação — 01-padroes-de-raciocinio-e-execucao

Este é o primeiro projeto do módulo 04 neste repositório e o nascimento do **OpsPilot**: um copiloto de plantão (incident commander) para o e-commerce fictício "Mercadinho". Nesta unidade nasce o **cérebro** do agente — três padrões de raciocínio (ReAct, Plan-and-Execute e Reflection) atrás de uma interface comum, com trace tipado, uma arena para comparar estratégias lado a lado e um bench que mede acerto **no estado do sistema**, não no texto da resposta.

> **Estado atual do nosso projeto:** fizemos `/speckit.specify`, `/speckit.plan` e `/speckit.tasks` para a spec `001-nucleo-raciocinio-opspilot` — o código ainda não existe (só o `src/index.ts` inicial). Esta documentação serve como **material de estudo para a implementação**: os trechos de código vêm do snapshot de referência do curso (`modulo04-criacao-de-agentes-autonomos-novo/02-padroes-de-raciocinio-e-execucao`), e cada doc aponta onde o nosso plano diverge dele. Pulamos a unidade 01 do curso (`notas-api`); por isso este projeto é o `01` aqui.

---

## Documentos

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Por onde começar: o que é o OpsPilot, trilha de leitura, mapa do código de referência e comandos |
| [spec-driven-com-spec-kit.md](./spec-driven-com-spec-kit.md) | Spec-Driven Development com o Spec Kit: constituição, `specify → plan → tasks → implement`, o que já produzimos e o que vem |
| [contrato-de-estrategia-e-trace.md](./contrato-de-estrategia-e-trace.md) | A interface `ReasoningStrategy`, o `TraceEvent` tipado (`thought/action/observation/plan/critique/answer`) e as métricas `llmCalls`/`latencyMs` |
| [tools-e-store-deterministico.md](./tools-e-store-deterministico.md) | As tools `list_alerts`/`open_incident`/`resolve_incident` com `tool()` + Zod, store injetado, seed validado e testes sem rede |
| [react.md](./react.md) | O padrão ReAct (Thought → Action → Observation) com `createReactAgent`, `recursionLimit` e `GraphRecursionError` |
| [plan-and-execute.md](./plan-and-execute.md) | Plan-and-Execute como `StateGraph`: planner → executor → replanner, schemas planos, teto de 8 passos e modo sem replanner |
| [reflection.md](./reflection.md) | A camada de Reflection (spec 002) como **decorator**: crítico com saída estruturada, feedback injetado, fail-safe e métricas acumuladas |
| [arena-e-bench.md](./arena-e-bench.md) | Arena (comparação lado a lado) e bench (acerto medido no store, cenários C1/C2/C3 e diagnóstico) |
| [roteiro-de-implementacao.md](./roteiro-de-implementacao.md) | Ponte entre o nosso `tasks.md` e o código de referência: ordem de implementação, divergências e armadilhas já identificadas |
| [glossario.md](./glossario.md) | Termos novos deste projeto — os cobertos no módulo 03 não são repetidos |

---

## Contexto do projeto

| Tecnologia | Papel no projeto |
| --- | --- |
| **Spec Kit** (`.specify/`, skills `speckit-*`) | Conduz o desenvolvimento por especificação: constituição → spec → plano → tarefas → implementação |
| **LangGraph** (`createReactAgent`, `StateGraph`) | ReAct pré-construído e o grafo planner/executor/replanner do Plan-and-Execute |
| **LangChain** (`@langchain/core`, `@langchain/openai`) | `tool()` para as ferramentas, mensagens (`AIMessage`, `ToolMessage`) para o trace, `ChatOpenAI` apontado para o OpenRouter |
| **OpenRouter** | Gateway de modelos; a fábrica única cria `ChatOpenAI` com `baseURL` do OpenRouter e `temperature: 0` |
| **Zod** | Schemas das tools, do plano, da decisão do replanner, do crítico e do seed |
| **`node:test` + `tsx`** | Testes determinísticos sem rede (store, trace, reflection com mocks) |
| **Sequelize + MySQL** | Fronteira de persistência prevista no nosso plano (o snapshot de referência ainda não usa) |
