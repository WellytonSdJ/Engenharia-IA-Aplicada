# Módulo 04 — Criação de Agentes Autônomos

Do "agente que chama tools" para o **agente que raciocina, lembra, se autocorrige e é observável**. O módulo inteiro constrói um único projeto que evolui unidade a unidade: o **OpsPilot**, um copiloto de plantão / incident commander do e-commerce fictício "Mercadinho", com LangChain/LangGraph sobre OpenRouter.

Diferente dos módulos anteriores, o desenvolvimento aqui é conduzido por **Spec-Driven Development com Spec Kit**: cada feature nasce como spec numerada (`specs/NNN-slug/`), passa por plano e tarefas, e só então é implementada pelo agente de código (GitHub Copilot). O módulo 03 já tinha introduzido as peças — agentes declarativos (`.agent.md`) e Agent Skills (`SKILL.md`); o Spec Kit junta essas peças num processo.

> A primeira unidade deste módulo, sobre operar agentes de código (projeto de notas via CLI), foi **pulada** neste repositório. Por isso a numeração aqui começa na segunda unidade do módulo: o projeto `01` deste repositório trata de padrões de raciocínio e execução.

## Projetos

| # | Projeto | Tema | Status | Descrição |
|---|---------|------|--------|-----------|
| 01 | [padroes-de-raciocinio-e-execucao](./01-padroes-de-raciocinio-e-execucao/) | Padrões de Raciocínio e Execução | ✅ specs 001 (núcleo ReAct/Plan-and-Execute), 002 (Reflection) e 003 (fix replanner) implementadas; bench C1/C2/C3 pronto. Spec 004 (`POST /chat`) especificada (specify → plan → tasks), mas a implementação acontece no projeto 02 | Kickoff do OpsPilot: estratégias ReAct e Plan-and-Execute atrás de uma interface comum, trace tipado, métricas, tools mock sobre store em memória, arena de comparação e bench com acerto medido no estado; depois, a camada de Reflection |
| 02 | [function-calling-e-tool-use](./02-function-calling-e-tool-use/) | Function Calling e Tool Use | ✅ specs 004 a 008 implementadas (99 testes): `POST /chat`, persistência SQLite com tool de runbook, tool de status de provedores externos, servidor MCP e conversa persistente | Evolução do OpsPilot: endpoint HTTP de chat, persistência real via `node:sqlite` com tool de runbook, tool de status de provedores externos, servidor MCP de operações e conversa persistente no `/chat` |
| 03 | [memoria-e-reflexao-em-agentes-autonomos](./03-memoria-e-reflexao-em-agentes-autonomos/) | Memória e Reflexão em Agentes Autônomos | ✅ specs 009 (memória semântica) e 010 (refletor de aprendizado) implementadas (129 testes sem rede + 3 testes com o modelo real de embeddings) | Memória de longo prazo do OpsPilot: embeddings locais (`all-MiniLM-L6-v2`), vector store em SQLite com `remember`/`recall`/`forget` por `userId`, memórias injetadas no `/chat`, `POST /memories`, refletor que aprende fatos duráveis em segundo plano e tool `forget_preference` |

## Roteiro do módulo (orientação, não implementado ainda)

Os próximos projetos seguem a trilha de temas do módulo. Como o OpsPilot é um projeto único e evolutivo, cada novo projeto parte do anterior. A lista abaixo é só orientação — cada um será documentado a fundo quando for de fato estudado e trazido para este repositório.

| # | Tema | Especificações previstas do OpsPilot |
|---|------|----------------------------|
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
| **SQLite** | nativo (`node:sqlite`) | Usado a partir do projeto 02 (spec 005) como persistência real via `DatabaseSync`, sem servidor externo; testes usam `:memory:` e o bench segue no store em memória |
| **`@huggingface/transformers`** | 4.2.x | A partir do projeto 03 (spec 009): roda o modelo de embeddings `all-MiniLM-L6-v2` localmente (sem API key). A primeira execução baixa ~23 MB; `npm test` não depende disso (usa `FakeEmbedder`) |

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
- Function calling e design de tools ("o schema é o prompt", 6 regras de descrição) com uma fonte única de definições para o agente e o MCP
- Persistência real sem servidor externo (`node:sqlite`, prepared statements, transações), tool de runbook, tool externa resiliente (timeout, retry seletivo, erro como observação), servidor MCP de operações e conversa persistente no `/chat`
- Memória de curto prazo (histórico por `conversationId`) × longo prazo (fatos por `userId`)
- Embeddings locais com Transformers.js, similaridade de cosseno como produto escalar de vetores normalizados, lazy singleton do modelo
- Vector store "na mão" com SQLite (vetor em BLOB, busca por força bruta), dedup semântico (> 0,92) e recall top-3 com piso (≥ 0,3)
- Reflection reaproveitada como loop de aprendizado (`{ hasLearning, fact }`), fire-and-forget e tool `forget_preference` com `userId` por closure
- (Roteiro futuro) Gerenciamento de contexto, grafos de produção, observabilidade, aprovação humana, publicação e multi-agentes

## Documentação de conceitos (projeto 01)

Documentação aprofundada disponível em [`01-padroes-de-raciocinio-e-execucao/docs/`](./01-padroes-de-raciocinio-e-execucao/docs/). Os docs de conceito foram escritos antes da implementação, e cada doc registra onde a implementação final divergiu do desenho inicial. O roteiro registra o que mudou depois da implementação.

Os projetos seguintes nascem como cópia do anterior e **não repetem** essa documentação: linkam de volta para cada doc abaixo e documentam só o que é novo.

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md) | Por onde começar: o OpsPilot, trilha de leitura, mapa do código de referência e comandos |
| [01-spec-driven-com-spec-kit.md](./01-padroes-de-raciocinio-e-execucao/docs/01-spec-driven-com-spec-kit.md) | O processo Spec Kit, a constituição, o que já produzimos e os comandos de apoio |
| [02-contrato-de-estrategia-e-trace.md](./01-padroes-de-raciocinio-e-execucao/docs/02-contrato-de-estrategia-e-trace.md) | `ReasoningStrategy`, `TraceEvent` tipado, `buildTraceFromMessages` e métricas |
| [03-tools-e-store-deterministico.md](./01-padroes-de-raciocinio-e-execucao/docs/03-tools-e-store-deterministico.md) | Tools com `tool()` + Zod, store injetado, seed validado e testes sem rede |
| [04-react.md](./01-padroes-de-raciocinio-e-execucao/docs/04-react.md) | ReAct com `createReactAgent`, `modelFactory` e limite de iterações |
| [05-plan-and-execute.md](./01-padroes-de-raciocinio-e-execucao/docs/05-plan-and-execute.md) | `StateGraph` planner → executor → replanner, schemas planos e modo sem replanner |
| [06-reflection.md](./01-padroes-de-raciocinio-e-execucao/docs/06-reflection.md) | Reflection como decorator: crítico, feedback injetado e métricas somadas |
| [07-arena-e-bench.md](./01-padroes-de-raciocinio-e-execucao/docs/07-arena-e-bench.md) | Arena lado a lado e bench com acerto no estado (C1/C2/C3), com lista de motivos quando o cenário falha |
| [09-variaveis-de-ambiente-com-env-file.md](./01-padroes-de-raciocinio-e-execucao/docs/09-variaveis-de-ambiente-com-env-file.md) | `.env` com `node --env-file-if-exists --import tsx`: o bug da arena, outros usos e montagem do zero |
| [08-roteiro-de-implementacao.md](./01-padroes-de-raciocinio-e-execucao/docs/08-roteiro-de-implementacao.md) | Ponte `tasks.md` × referência, divergências e armadilhas já identificadas |
| [10-glossario.md](./01-padroes-de-raciocinio-e-execucao/docs/10-glossario.md) | Termos novos do projeto |

## Documentação de conceitos (projeto 02)

Documentação aprofundada em [`02-function-calling-e-tool-use/docs/`](./02-function-calling-e-tool-use/docs/):

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./02-function-calling-e-tool-use/docs/00-START-HERE.md) | O que muda nesta unidade, trilha de leitura, mapa do código, como rodar e exercícios |
| [01-function-calling-e-design-de-tools.md](./02-function-calling-e-tool-use/docs/01-function-calling-e-design-de-tools.md) | O modelo pede e o código executa; "o schema é o prompt"; as 6 regras de descrição; `list_incidents`, `consultar_runbook`; fonte única de tools |
| [04-api-chat-http.md](./02-function-calling-e-tool-use/docs/04-api-chat-http.md) | `POST /chat`: Zod na fronteira, registry de estratégias, `reflect`, timeout e erro → status HTTP |
| [02-persistencia-sqlite.md](./02-function-calling-e-tool-use/docs/02-persistencia-sqlite.md) | `node:sqlite`, DDL idempotente, `CHECK`, prepared statements, transações, seed e `:memory:` |
| [03-tool-externa-resiliente.md](./02-function-calling-e-tool-use/docs/03-tool-externa-resiliente.md) | `check_provider_status`: timeout, retry seletivo, Zod na resposta, erro como observação, `fetch` injetável |
| [05-servidor-mcp.md](./02-function-calling-e-tool-use/docs/05-servidor-mcp.md) | O OpsPilot como provedor MCP: 3 tools da mesma fonte, stdio sem `console.log`, testes em memória, prova circular |
| [06-conversa-persistente.md](./02-function-calling-e-tool-use/docs/06-conversa-persistente.md) | `conversationId`, janela de 12 mensagens, gravar só após sucesso, 422 `UNKNOWN_CONVERSATION` |
| [07-glossario.md](./02-function-calling-e-tool-use/docs/07-glossario.md) | Termos novos do projeto |

## Documentação de conceitos (projeto 03)

O projeto 03 nasce como cópia do 02 e, do mesmo jeito, **não repete** a documentação anterior: linka para o 01/02 e documenta só memória e aprendizado. Cada doc abre com um bloco "Em 30 segundos" e tem checkpoints ao longo do texto. O [`00-START-HERE.md`](./03-memoria-e-reflexao-em-agentes-autonomos/docs/00-START-HERE.md) inclui uma tabela que liga cada arquivo da referência do curso ao arquivo correspondente aqui.

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./03-memoria-e-reflexao-em-agentes-autonomos/docs/00-START-HERE.md) | O que muda na unidade, trilha de leitura, mapa do código, mapa referência → nosso e como rodar |
| [01-embeddings-e-similaridade.md](./03-memoria-e-reflexao-em-agentes-autonomos/docs/01-embeddings-e-similaridade.md) | Embedding, cosseno como produto escalar, `all-MiniLM-L6-v2` local e lazy singleton |
| [02-memoria-semantica.md](./03-memoria-e-reflexao-em-agentes-autonomos/docs/02-memoria-semantica.md) | Curto × longo prazo, vector store em SQLite, `remember`/`recall`/`forget`, limiares e memórias no `/chat` |
| [03-refletor-de-aprendizado.md](./03-memoria-e-reflexao-em-agentes-autonomos/docs/03-refletor-de-aprendizado.md) | Reflection como loop de aprendizado, durável × pontual × segredo, fire-and-forget |
| [04-forget-preference.md](./03-memoria-e-reflexao-em-agentes-autonomos/docs/04-forget-preference.md) | Tool presa ao usuário do turno por closure (e por que a referência usa `AsyncLocalStorage`) |
| [05-testando-memoria.md](./03-memoria-e-reflexao-em-agentes-autonomos/docs/05-testando-memoria.md) | `FakeEmbedder`, testes semânticos fora do `npm test` e teste do "não esperou" |
| [06-glossario.md](./03-memoria-e-reflexao-em-agentes-autonomos/docs/06-glossario.md) | Termos novos da unidade |
