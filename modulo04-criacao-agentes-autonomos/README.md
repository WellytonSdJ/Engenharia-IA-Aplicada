# Módulo 04 — Criação de Agentes Autônomos

Do "agente que chama tools" para o **agente que raciocina, lembra, se autocorrige e é observável**. O módulo inteiro constrói um único projeto que evolui unidade a unidade: o **OpsPilot**, um copiloto de plantão / incident commander do e-commerce fictício "Mercadinho", com LangChain/LangGraph sobre OpenRouter.

Diferente dos módulos anteriores, o desenvolvimento aqui é conduzido por **Spec-Driven Development com Spec Kit**: cada feature nasce como spec numerada (`specs/NNN-slug/`), passa por plano e tarefas, e só então é implementada pelo agente de código (GitHub Copilot). O módulo 03 já tinha introduzido as peças — agentes declarativos (`.agent.md`) e Agent Skills (`SKILL.md`); o Spec Kit junta essas peças num processo.

Material de referência do curso: `POS_ENGENHARIA_IA/engenharia-de-software-com-ia-aplicada/modulo04-criacao-de-agentes-autonomos-novo/` — cada pasta de lá é um snapshot cumulativo do OpsPilot ao final de uma unidade.

> A unidade 01 do curso (`01-arquitetura-de-agentes-de-codigo`, projeto `notas-api`, sobre operar agentes de código) foi **pulada** neste repositório. Por isso a numeração aqui começa na unidade 02 do curso: o projeto `01` deste repositório corresponde à pasta `02` do curso.

## Projetos

| # | Projeto | Unidade do curso | Status | Descrição |
|---|---------|------------------|--------|-----------|
| 01 | [padroes-de-raciocinio-e-execucao](./01-padroes-de-raciocinio-e-execucao/) | U2 — Padrões de Raciocínio e Execução | 🚧 em andamento (spec, plano e tarefas prontos; implementação pendente) | Kickoff do OpsPilot: estratégias ReAct e Plan-and-Execute atrás de uma interface comum, trace tipado, métricas, tools mock sobre store em memória, arena de comparação e bench com acerto medido no estado; depois, a camada de Reflection |

## Roteiro do módulo (orientação, não implementado ainda)

Os próximos projetos seguem as unidades do curso. Como o OpsPilot é um projeto único e evolutivo, cada novo projeto parte do anterior. A lista abaixo é só orientação — cada um será documentado a fundo quando for de fato estudado e trazido para este repositório.

| # | Pasta de referência no curso | Tema | Specs do OpsPilot no curso |
|---|------------------------------|------|----------------------------|
| 02 | `03-function-calling-e-tool-use` | Function Calling e Tool Use | 003-chat-api, 004-sqlite-ops-store, 005-provider-status-tool, 006-mcp-ops-server |
| 03 | `04-memoria-e-reflexao-em-agentes-autonomos` | Memória e Reflexão em Agentes Autônomos | 007-persistent-conversation, 008-semantic-memory, 009-learning-reflector |
| 04 | `05-gerenciamento-de-contextos` | Gerenciamento de Contextos | 010-context-measurement, 011-history-summarization, 012-context-builder-budget |
| 05 | `06-langgraph-e-workflows-complexos` | LangGraph e Workflows Complexos | 013-unified-production-graph, 014-model-resilience |
| 06 | `07-observabilidade-e-limites-de-autonomia` | Observabilidade e Limites de Autonomia | 015-persistent-trace-logs (+ aprovação humana) |
| 07 | `08-projeto-pratico-opspilot-publicado` | Projeto Prático: o OpsPilot de ponta a ponta, publicado | 016-war-room-web, 017-pages-web-deploy |
| 08 | `09-multi-agent-systems` | Multi-Agent Systems | 018-team-mode |

## Requisitos gerais

| Requisito | Versão mínima | Observação |
| --- | --- | --- |
| **Node.js** | 22 LTS | A constituição do projeto fixa Node 22; variáveis de ambiente via `--env-file` nativo (sem `dotenv`) |
| **npm** | 10+ | — |
| **Conta OpenRouter** | — | `OPENROUTER_API_KEY` (e `OPENROUTER_MODEL`); modelos `:free` custam zero |
| **Spec Kit** | 1.0.x | Integração `copilot`, scripts PowerShell (`.specify/`) |
| **GitHub Copilot** | — | Agente de código que executa os comandos `speckit.*` |
| **MySQL** | — | Opcional nesta unidade — só para o adapter Sequelize previsto no plano; testes e arena usam store em memória |

> Cada projeto tem seu próprio `docs/` com requisitos detalhados e instruções de execução.

## Conceitos abordados

- Spec-Driven Development com Spec Kit: constituição, `specify → plan → tasks → implement`, gates de revisão, `analyze`/`converge`
- Padrões de raciocínio de agentes: ReAct, Plan-and-Execute (planner/executor/replanner) e Reflection (crítico + regeneração)
- Interface comum de estratégia (`ReasoningStrategy`) e decorator (`withReflection`) para compor estratégias
- Trace tipado (`thought/action/observation/plan/critique/answer`) e métricas (`llmCalls`, `latencyMs`) como dado verificável
- `createReactAgent`, `recursionLimit`/`GraphRecursionError` e `StateGraph` próprio com reducers de substituição e acumulação
- Structured output para planos e decisões, com schemas planos por compatibilidade com modelos via OpenRouter
- Tools com `tool()` + Zod sobre store injetado; erros de domínio traduzidos em observações para o modelo
- Avaliação de agentes: arena (comparação lado a lado) e bench com acerto medido no estado do sistema
- Testes determinísticos sem rede com doubles de modelo, estratégia e crítico
- (Roteiro futuro) Function calling em API, MCP de operações, memória persistente e semântica, gerenciamento de contexto, grafos de produção, observabilidade, aprovação humana, publicação e multi-agentes

## Documentação de conceitos (projeto 01)

Documentação aprofundada disponível em [`01-padroes-de-raciocinio-e-execucao/docs/`](./01-padroes-de-raciocinio-e-execucao/docs/). Como a implementação ainda não foi feita, os trechos de código vêm do snapshot de referência do curso, e cada doc aponta onde o nosso plano diverge dele.

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md) | Por onde começar: o OpsPilot, trilha de leitura, mapa do código de referência e comandos |
| [spec-driven-com-spec-kit.md](./01-padroes-de-raciocinio-e-execucao/docs/spec-driven-com-spec-kit.md) | O processo Spec Kit, a constituição, o que já produzimos e os comandos de apoio |
| [contrato-de-estrategia-e-trace.md](./01-padroes-de-raciocinio-e-execucao/docs/contrato-de-estrategia-e-trace.md) | `ReasoningStrategy`, `TraceEvent` tipado, `buildTraceFromMessages` e métricas |
| [tools-e-store-deterministico.md](./01-padroes-de-raciocinio-e-execucao/docs/tools-e-store-deterministico.md) | Tools com `tool()` + Zod, store injetado, seed validado e testes sem rede |
| [react.md](./01-padroes-de-raciocinio-e-execucao/docs/react.md) | ReAct com `createReactAgent`, `modelFactory` e limite de iterações |
| [plan-and-execute.md](./01-padroes-de-raciocinio-e-execucao/docs/plan-and-execute.md) | `StateGraph` planner → executor → replanner, schemas planos e modo sem replanner |
| [reflection.md](./01-padroes-de-raciocinio-e-execucao/docs/reflection.md) | Reflection como decorator: crítico, fail-safe, feedback injetado e métricas somadas |
| [arena-e-bench.md](./01-padroes-de-raciocinio-e-execucao/docs/arena-e-bench.md) | Arena lado a lado e bench com acerto no estado (C1/C2/C3, `diagnose`) |
| [roteiro-de-implementacao.md](./01-padroes-de-raciocinio-e-execucao/docs/roteiro-de-implementacao.md) | Ponte `tasks.md` × referência, divergências e armadilhas já identificadas |
| [glossario.md](./01-padroes-de-raciocinio-e-execucao/docs/glossario.md) | Termos novos do projeto |
