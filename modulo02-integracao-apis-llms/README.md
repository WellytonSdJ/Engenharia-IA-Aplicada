# Módulo 02 — Integração com APIs de LLMs

Integração de aplicações Node.js com APIs de modelos de linguagem (LLMs), cobrindo desde roteamento inteligente entre modelos até RAG com banco de grafos usando LangGraph.

## Projetos

| # | Projeto | Descrição |
|---|---------|-----------|
| 01 | [smart-model-router-gateway](./01-smart-model-router-gateway/) | Gateway HTTP que roteia para o melhor modelo LLM via OpenRouter com base em critérios configuráveis |
| 02 | [langchain-intro](./02-langchain-intro/) | Primeiro grafo LangGraph: StateGraph, nós puros, edges condicionais com Zod state — sem LLM, foco na mecânica do framework |
| 03 | [medical-appointment](./03-medical-appointment/) | Prompt chaining: pipeline identify→execute→respond com LangGraph + LLM. Zod structured output, serviço de domínio, factory pattern |
| 04 | [song-highlights](./04-song-highlights/) | Chatbot musical CLI com grafo LangGraph, checkpointer Postgres, extração de preferências com Zod e persistência em SQLite |
| 05 | [safeguard-prompt-injection](./05-safeguard-prompt-injection/) | Demo de ataques de prompt injection e defesa com guardrails baseados em LLM; RBAC admin vs member via MCP |
| 06 | [rag-neo4j-students](./06-rag-neo4j-students/) | API de análise de vendas que converte linguagem natural em Cypher, executa no Neo4j e retorna respostas analíticas com autocorreção e decomposição multi-step |

## Requisitos gerais

| Requisito | Versão mínima | Projetos |
| --- | --- | --- |
| **Node.js** | 22.6.0+ (projeto 01) / 24.10.0+ (demais) | Todos |
| **npm** | 10+ | Todos |
| **Docker + Docker Compose** | 20+ / 2+ | Projetos 04 e 06 |
| **Conta OpenRouter** | — | Todos exceto 02 |
| **Conta LangSmith** | — | Projetos 02, 04 e 06 (opcional) |

> Cada projeto tem seu próprio `README.md` com requisitos detalhados, variáveis de ambiente e instruções de execução.

## Conceitos abordados

- Integração com a API OpenRouter para acesso a múltiplos modelos LLM
- Roteamento dinâmico de modelos por preço, throughput e latência
- Servidor HTTP com Fastify e validação de schema
- Execução de TypeScript nativa com Node.js (sem compilação)
- Testes E2E com `node:test` (runner nativo do Node.js)
- Grafos de estado com LangGraph (`StateGraph`, nós, edges condicionais, reducers, state com Zod)
- Prompt chaining: pipeline multi-etapa com LangGraph (intent → execução → resposta)
- Saída estruturada de LLMs com validação por Zod (`withStructuredOutput`, `providerStrategy`)
- Factory pattern para injeção de dependências em nós do LangGraph
- Separação entre serviço de domínio (lógica de negócio) e serviço de LLM
- Checkpointer e Store do LangGraph para persistência de sessão no Postgres
- Extração de entidades de conversas em tempo real
- Persistência de preferências de usuário em SQLite com Knex
- Sumarização incremental de conversas para compressão de histórico
- Gerenciamento de sessões por `thread_id` no LangGraph
- Prompt injection: vetores de ataque, demonstração prática e defesa com guardrails baseados em LLM
- RBAC (controle de acesso por papel) com ferramentas MCP de sistema de arquivos
- RAG com banco de grafos: geração de Cypher a partir de linguagem natural com Neo4j
- Self-correction: autocorreção automática de queries com erro usando o schema real do banco
- Decomposição multi-step: quebra de perguntas complexas em sub-queries independentes

## Documentação de conceitos (projeto 02)

Documentação aprofundada dos conceitos aplicados disponível em [`02-langchain-intro/docs/`](./02-langchain-intro/docs/):

| Documento | Conteúdo |
| --- | --- |
| [langgraph-intro.md](./02-langchain-intro/docs/langgraph-intro.md) | StateGraph, state com Zod, nós, edges condicionais, `withLangGraph`, compilação do grafo |
| [langchain-messages.md](./02-langchain-intro/docs/langchain-messages.md) | HumanMessage, AIMessage, BaseMessage, MessagesZodMeta, reducer de mensagens |

## Documentação de conceitos (projeto 03)

Documentação aprofundada dos conceitos aplicados disponível em [`03-medical-appointment/docs/`](./03-medical-appointment/docs/):

| Documento | Conteúdo |
| --- | --- |
| [prompt-chaining.md](./03-medical-appointment/docs/prompt-chaining.md) | Pipeline identify→execute→respond, por que separar etapas, routing condicional |
| [structured-output.md](./03-medical-appointment/docs/structured-output.md) | Zod schemas como contrato, `generateStructured`, `createAgent`, `providerStrategy` |
| [node-factory-pattern.md](./03-medical-appointment/docs/node-factory-pattern.md) | Injeção de dependências em nós via factory functions, `Partial<GraphState>` |
| [domain-vs-llm-service.md](./03-medical-appointment/docs/domain-vs-llm-service.md) | AppointmentService vs OpenRouterService: separação de responsabilidades |

## Documentação de conceitos (projeto 04)

Documentação aprofundada dos conceitos aplicados disponível em [`04-song-highlights/docs/`](./04-song-highlights/docs/):

| Documento | Conteúdo |
| --- | --- |
| [langchain.md](./04-song-highlights/docs/langchain.md) | Abstrações core, ChatOpenAI, structured output, integração com LangGraph |
| [langgraph.md](./04-song-highlights/docs/langgraph.md) | StateGraph, nós, edges, reducers, checkpointer, store, Runtime |
| [conversation-summarization.md](./04-song-highlights/docs/conversation-summarization.md) | Sumarização incremental, trigger por contagem, RemoveMessage |
| [persistence.md](./04-song-highlights/docs/persistence.md) | Postgres (LangGraph) vs SQLite (preferências), quando usar cada um |
| [prompt-injection.md](./04-song-highlights/docs/prompt-injection.md) | O que é, vetores de ataque, mitigações, exposições no projeto |

## Documentação de conceitos (projeto 05)

Documentação aprofundada dos conceitos aplicados disponível em [`05-safeguard-prompt-injection/docs/`](./05-safeguard-prompt-injection/docs/):

| Documento | Conteúdo |
| --- | --- |
| [prompt-injection.md](./05-safeguard-prompt-injection/docs/prompt-injection.md) | Demonstração prática dos ataques: override direto, instrução indireta, por que system prompt não protege |
| [guardrails.md](./05-safeguard-prompt-injection/docs/guardrails.md) | O que são guardrails, safeguard model vs. executor, defesa determinística vs. probabilística, fail closed |
| [mcp.md](./05-safeguard-prompt-injection/docs/mcp.md) | Model Context Protocol: servidor de filesystem, STDIO transport, por que MCP amplifica o risco de injection |
| [rbac.md](./05-safeguard-prompt-injection/docs/rbac.md) | Role-Based Access Control: admin/member, por que RBAC via prompt falha, RBAC via código |
| [langgraph.md](./05-safeguard-prompt-injection/docs/langgraph.md) | Grafo de segurança: guardrails_check → chat/blocked, SafeguardStateAnnotation, roteamento condicional |
