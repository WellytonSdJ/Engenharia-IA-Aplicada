# Módulo 02 — Integração com APIs de LLMs

Integração de aplicações Node.js com APIs de modelos de linguagem (LLMs), cobrindo desde roteamento inteligente entre modelos até RAG com banco de grafos usando LangGraph.

## Projetos

| # | Projeto | Descrição |
|---|---------|-----------|
| 01 | [smart-model-router-gateway](./01-smart-model-router-gateway/) | Gateway HTTP que roteia para o melhor modelo LLM via OpenRouter com base em critérios configuráveis |
| 02 | [song-highlights](./02-song-highlights/) | Chatbot musical CLI com grafo de estados LangGraph, extração de preferências com Zod e persistência em SQLite |
| 03 | [safeguard-prompt-injection](./03-safeguard-prompt-injection/) | Demo de ataques de prompt injection e defesa com guardrails baseados em LLM; RBAC admin vs member via MCP |
| 04 | [rag-neo4j-students](./04-rag-neo4j-students/) | API de análise de vendas que converte linguagem natural em Cypher, executa no Neo4j e retorna respostas analíticas com autocorreção e decomposição multi-step |

## Requisitos gerais

| Requisito | Versão mínima | Projetos |
| --- | --- | --- |
| **Node.js** | 22.6.0+ (projeto 01) / 24.10.0+ (demais) | Todos |
| **npm** | 10+ | Todos |
| **Docker + Docker Compose** | 20+ / 2+ | Projetos 02 e 04 |
| **Conta OpenRouter** | — | Todos |
| **Conta LangSmith** | — | Projetos 02 e 04 (opcional) |

> Cada projeto tem seu próprio `README.md` com requisitos detalhados, variáveis de ambiente e instruções de execução.

## Conceitos abordados

- Integração com a API OpenRouter para acesso a múltiplos modelos LLM
- Roteamento dinâmico de modelos por preço, throughput e latência
- Servidor HTTP com Fastify e validação de schema
- Execução de TypeScript nativa com Node.js (sem compilação)
- Testes E2E com `node:test` (runner nativo do Node.js)
- Grafos de estado com LangGraph (`StateGraph`, nós, edges condicionais, reducers)
- Checkpointer e Store do LangGraph para persistência de sessão no Postgres
- Saída estruturada de LLMs com validação por Zod (`withStructuredOutput`)
- Extração de entidades de conversas em tempo real
- Persistência de preferências de usuário em SQLite com Knex
- Sumarização incremental de conversas para compressão de histórico
- Prompt engineering para extração de dados e sumarização de conversas
- Gerenciamento de sessões por `thread_id` no LangGraph
- Prompt injection: vetores de ataque, demonstração prática e defesa com guardrails baseados em LLM
- RBAC (controle de acesso por papel) com ferramentas MCP de sistema de arquivos
- RAG com banco de grafos: geração de Cypher a partir de linguagem natural com Neo4j
- Self-correction: autocorreção automática de queries com erro usando o schema real do banco
- Decomposição multi-step: quebra de perguntas complexas em sub-queries independentes

## Documentação de conceitos (projeto 02)

Documentação aprofundada dos conceitos aplicados disponível em [`02-song-highlights/docs/`](./02-song-highlights/docs/):

| Documento | Conteúdo |
| --- | --- |
| [langchain.md](./02-song-highlights/docs/langchain.md) | Abstrações core, ChatOpenAI, structured output, integração com LangGraph |
| [langgraph.md](./02-song-highlights/docs/langgraph.md) | StateGraph, nós, edges, reducers, checkpointer, store, Runtime |
| [conversation-summarization.md](./02-song-highlights/docs/conversation-summarization.md) | Sumarização incremental, trigger por contagem, RemoveMessage |
| [persistence.md](./02-song-highlights/docs/persistence.md) | Postgres (LangGraph) vs SQLite (preferências), quando usar cada um |
| [prompt-injection.md](./02-song-highlights/docs/prompt-injection.md) | O que é, vetores de ataque, mitigações, exposições no projeto |

## Documentação de conceitos (projeto 03)

Documentação aprofundada dos conceitos aplicados disponível em [`03-safeguard-prompt-injection/docs/`](./03-safeguard-prompt-injection/docs/):

| Documento | Conteúdo |
| --- | --- |
| [prompt-injection.md](./03-safeguard-prompt-injection/docs/prompt-injection.md) | Demonstração prática dos ataques: override direto, instrução indireta, por que system prompt não protege |
| [guardrails.md](./03-safeguard-prompt-injection/docs/guardrails.md) | O que são guardrails, safeguard model vs. executor, defesa determinística vs. probabilística, fail closed |
| [mcp.md](./03-safeguard-prompt-injection/docs/mcp.md) | Model Context Protocol: servidor de filesystem, STDIO transport, por que MCP amplifica o risco de injection |
| [rbac.md](./03-safeguard-prompt-injection/docs/rbac.md) | Role-Based Access Control: admin/member, por que RBAC via prompt falha, RBAC via código |
| [langgraph.md](./03-safeguard-prompt-injection/docs/langgraph.md) | Grafo de segurança: guardrails_check → chat/blocked, SafeguardStateAnnotation, roteamento condicional |
