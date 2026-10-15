# Módulo 04 — Criação de Agentes Autônomos

Do "agente que chama tools" para o **agente que raciocina, lembra, se autocorrige e é observável**. O módulo inteiro constrói um único projeto que evolui unidade a unidade: o **OpsPilot**, um copiloto de plantão / incident commander do e-commerce fictício "Mercadinho", com LangChain/LangGraph sobre OpenRouter.

Diferente dos módulos anteriores, o desenvolvimento aqui é conduzido por **Spec-Driven Development com Spec Kit**: cada feature nasce como spec numerada (`specs/NNN-slug/`), passa por plano e tarefas, e só então é implementada pelo agente de código (GitHub Copilot). O módulo 03 já tinha introduzido as peças — agentes declarativos (`.agent.md`) e Agent Skills (`SKILL.md`); o Spec Kit junta essas peças num processo.

> A primeira unidade deste módulo, sobre operar agentes de código (projeto de notas via CLI), foi **pulada** neste repositório. Por isso a numeração aqui começa na segunda unidade do módulo: o projeto `01` deste repositório trata de padrões de raciocínio e execução.

## Projetos

| # | Projeto | Tema | Status | Descrição |
|---|---------|------|--------|-----------|
| 01 | [padroes-de-raciocinio-e-execucao](./01-padroes-de-raciocinio-e-execucao/) | Padrões de Raciocínio e Execução | ✅ specs 001 (núcleo ReAct/Plan-and-Execute), 002 (Reflection) e 003 (fix replanner) implementadas; bench C1/C2/C3 pronto. Spec 004 (`POST /chat`) especificada (specify → plan → tasks), mas a implementação acontece no projeto 02 | Kickoff do OpsPilot: estratégias ReAct e Plan-and-Execute atrás de uma interface comum, trace tipado, métricas, tools mock sobre store em memória, arena de comparação e bench com acerto medido no estado; depois, a camada de Reflection |
| 02 | [function-calling-e-tool-use](./02-function-calling-e-tool-use/) | Function Calling e Tool Use | ✅ specs 004 a 008 implementadas (99 testes): `POST /chat`, persistência SQLite com tool de runbook, tool de status de provedores externos, servidor MCP e conversa persistente | Evolução do OpsPilot: endpoint HTTP de chat, persistência real via `node:sqlite` com tool de runbook, tool de status de provedores externos, servidor MCP de operações e conversa persistente no `/chat` |

## Roteiro do módulo (orientação, não implementado ainda)

Os próximos projetos seguem a trilha de temas do módulo. Como o OpsPilot é um projeto único e evolutivo, cada novo projeto parte do anterior. A lista abaixo é só orientação — cada um será documentado a fundo quando for de fato estudado e trazido para este repositório.

| # | Tema | Especificações previstas do OpsPilot |
|---|------|----------------------------|
| 03 | Memória e Reflexão em Agentes Autônomos | memória semântica, reflector de aprendizado de longo prazo |
| 04 | Gerenciamento de Contextos | medição de contexto, sumarização de histórico, orçamento de construção de contexto |
| 05 | LangGraph e Workflows Complexos | grafo de produção unificado, resiliência de modelo |
| 06 | Observabilidade e Limites de Autonomia | logs de trace persistentes, aprovação humana |
| 07 | Projeto Prático: o OpsPilot de ponta a ponta, publicado | interface web de "war room", publicação |
| 08 | Multi-Agent Systems | modo de equipe multi-agente |

## Requisitos gerais

| Requisito | Versão mínima | Observação |
| --- | --- | --- |
| **Node.js** | 22.9+ (22 LTS) | A constituição do projeto fixa Node 22; variáveis de ambiente via `--env-file-if-exists` nativo (sem `dotenv`), que exige 22.9+ |
| **npm** | 10+ | — |
| **Conta OpenRouter** | — | `OPENROUTER_API_KEY` (e `OPENROUTER_MODEL`); modelos `:free` custam zero |
| **Spec Kit** | 1.0.x | Integração `copilot`, scripts PowerShell (`.specify/`) |
| **GitHub Copilot** | — | Agente de código que executa os comandos `speckit.*` |
| **SQLite** | nativo (`node:sqlite`) | Usado a partir do projeto 02 (spec 005) como persistência real via `DatabaseSync`, sem servidor externo; testes e bench usam `:memory:` |

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
- Carregamento nativo de `.env` (`--env-file-if-exists`) sem `dotenv`, e por que os testes não o carregam
- Persistência real sem servidor externo (`node:sqlite`), tool de runbook, tool de status de provedores externos, servidor MCP de operações e conversa persistente no `/chat`
- (Roteiro futuro) Memória persistente e semântica, gerenciamento de contexto, grafos de produção, observabilidade, aprovação humana, publicação e multi-agentes

## Documentação de conceitos (projeto 01)

Documentação aprofundada disponível em [`01-padroes-de-raciocinio-e-execucao/docs/`](./01-padroes-de-raciocinio-e-execucao/docs/). Os docs de conceito foram escritos antes da implementação, e cada doc registra onde a implementação final divergiu do desenho inicial. O roteiro registra o que mudou depois da implementação.

O projeto 02 nasce como cópia do 01 e **não repete** essa documentação — seu [`docs/README.md`](./02-function-calling-e-tool-use/docs/README.md) linka de volta para cada doc abaixo e documenta só o que é novo (persistência SQLite, tool `consultar_runbook`, status de provedores externos, servidor MCP, conversa persistente).

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md) | Por onde começar: o OpsPilot, trilha de leitura, mapa do código de referência e comandos |
| [spec-driven-com-spec-kit.md](./01-padroes-de-raciocinio-e-execucao/docs/spec-driven-com-spec-kit.md) | O processo Spec Kit, a constituição, o que já produzimos e os comandos de apoio |
| [contrato-de-estrategia-e-trace.md](./01-padroes-de-raciocinio-e-execucao/docs/contrato-de-estrategia-e-trace.md) | `ReasoningStrategy`, `TraceEvent` tipado, `buildTraceFromMessages` e métricas |
| [tools-e-store-deterministico.md](./01-padroes-de-raciocinio-e-execucao/docs/tools-e-store-deterministico.md) | Tools com `tool()` + Zod, store injetado, seed validado e testes sem rede |
| [react.md](./01-padroes-de-raciocinio-e-execucao/docs/react.md) | ReAct com `createReactAgent`, `modelFactory` e limite de iterações |
| [plan-and-execute.md](./01-padroes-de-raciocinio-e-execucao/docs/plan-and-execute.md) | `StateGraph` planner → executor → replanner, schemas planos e modo sem replanner |
| [reflection.md](./01-padroes-de-raciocinio-e-execucao/docs/reflection.md) | Reflection como decorator: crítico, feedback injetado e métricas somadas |
| [arena-e-bench.md](./01-padroes-de-raciocinio-e-execucao/docs/arena-e-bench.md) | Arena lado a lado e bench com acerto no estado (C1/C2/C3), com lista de motivos quando o cenário falha |
| [variaveis-de-ambiente-com-env-file.md](./01-padroes-de-raciocinio-e-execucao/docs/variaveis-de-ambiente-com-env-file.md) | `.env` com `node --env-file-if-exists --import tsx`: o bug da arena, outros usos e montagem do zero |
| [roteiro-de-implementacao.md](./01-padroes-de-raciocinio-e-execucao/docs/roteiro-de-implementacao.md) | Ponte `tasks.md` × referência, divergências e armadilhas já identificadas |
| [glossario.md](./01-padroes-de-raciocinio-e-execucao/docs/glossario.md) | Termos novos do projeto |
