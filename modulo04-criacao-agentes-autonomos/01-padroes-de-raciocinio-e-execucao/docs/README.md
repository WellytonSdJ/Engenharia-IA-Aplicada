# Documentação — 01-padroes-de-raciocinio-e-execucao

Este é o primeiro projeto do módulo 04 neste repositório e o nascimento do **OpsPilot**: um copiloto de plantão (incident commander) para o e-commerce fictício "Mercadinho". Nesta unidade nasce o **cérebro** do agente — três padrões de raciocínio (ReAct, Plan-and-Execute e Reflection) atrás de uma interface comum, com trace tipado, uma arena para comparar estratégias lado a lado e um bench que mede acerto **no estado do sistema**, não no texto da resposta.

> **Estado atual do nosso projeto:** a spec `001-nucleo-raciocinio-opspilot` foi implementada (`/speckit.implement`): o `src/` existe, os 43 testes passam, a arena roda com LLM real e o bench mede acerto no estado. As specs `002-camada-reflection` (Reflection) e `003-fix-replanner-output` também foram implementadas. A spec `004-http-chat-endpoint` (`POST /chat`) já passou por `specify → plan → tasks`, mas não chegou a ser implementada neste projeto — a implementação real do endpoint HTTP acontece no projeto `02-function-calling-e-tool-use`, que nasce como cópia deste. Esta documentação foi escrita **antes** da implementação, como material de estudo para ela, e cada doc registra onde a implementação final divergiu do desenho inicial.

---

## Documentos

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Por onde começar: o que é o OpsPilot, trilha de leitura, mapa do código de referência e comandos |
| [01-spec-driven-com-spec-kit.md](./01-spec-driven-com-spec-kit.md) | Spec-Driven Development com o Spec Kit: constituição, `specify → plan → tasks → implement`, o que já produzimos e o que vem |
| [02-contrato-de-estrategia-e-trace.md](./02-contrato-de-estrategia-e-trace.md) | A interface `ReasoningStrategy`, o `TraceEvent` tipado (`thought/action/observation/plan/critique/answer`) e as métricas `llmCalls`/`latencyMs` |
| [03-tools-e-store-deterministico.md](./03-tools-e-store-deterministico.md) | As tools `list_alerts`/`open_incident`/`resolve_incident` com `tool()` + Zod, store injetado, seed validado e testes sem rede |
| [04-react.md](./04-react.md) | O padrão ReAct (Thought → Action → Observation) com `createReactAgent`, `recursionLimit` e `GraphRecursionError` |
| [05-plan-and-execute.md](./05-plan-and-execute.md) | Plan-and-Execute como `StateGraph`: planner → executor → replanner, schemas planos, teto de 8 passos e modo sem replanner |
| [06-reflection.md](./06-reflection.md) | A camada de Reflection (spec 002) como **decorator**: crítico com saída estruturada, feedback injetado, sem fail-safe no erro do crítico, e métricas acumuladas |
| [07-arena-e-bench.md](./07-arena-e-bench.md) | Arena (comparação lado a lado) e bench (acerto medido no store, cenários C1/C2/C3 e diagnóstico) |
| [09-variaveis-de-ambiente-com-env-file.md](./09-variaveis-de-ambiente-com-env-file.md) | Como o `.env` chega ao `process.env` com `node --env-file-if-exists=.env --import tsx`, o bug que isso resolveu, outros usos e como montar do zero |
| [08-roteiro-de-implementacao.md](./08-roteiro-de-implementacao.md) | Ponte entre o nosso `tasks.md` e o código de referência: ordem de implementação, divergências e armadilhas já identificadas |
| [10-glossario.md](./10-glossario.md) | Termos novos deste projeto — os cobertos no módulo 03 não são repetidos |

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
| **`--env-file-if-exists` (Node ≥ 22.9)** | Carrega o `.env` nos scripts `dev`/`arena`/`bench`/`seed` sem `dotenv`, como exige a constituição |
| **Sequelize + MySQL** | Fronteira de persistência inicialmente prevista no plano deste projeto, substituída depois por SQLite nativo (projeto 02, spec 005) |

---

**➡️ Próximo passo: [00 · Por onde começar](./00-START-HERE.md)**
