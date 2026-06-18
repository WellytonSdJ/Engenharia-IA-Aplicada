# Módulo 02 — Integração com APIs de LLMs

Integração de aplicações Node.js com APIs de modelos de linguagem (LLMs), cobrindo desde roteamento inteligente entre modelos até a construção de agentes conversacionais com memória persistente usando LangGraph.

## Projetos

| # | Projeto | Descrição |
|---|---------|-----------|
| 01 | [smart-model-router-gateway](./01-smart-model-router-gateway/) | Gateway HTTP que roteia para o melhor modelo LLM via OpenRouter com base em critérios configuráveis |
| 02 | [song-highlights](./02-song-highlights/) | Chatbot musical CLI com grafo de estados LangGraph, extração de preferências com Zod e persistência em SQLite |

## Conceitos abordados

- Integração com a API OpenRouter para acesso a múltiplos modelos LLM
- Roteamento dinâmico de modelos por preço, throughput e latência
- Servidor HTTP com Fastify e validação de schema
- Execução de TypeScript nativa com Node.js (sem compilação)
- Testes E2E com `node:test` (runner nativo do Node.js)
- Grafos de estado com LangGraph (`StateGraph`, nós, edges condicionais)
- Saída estruturada de LLMs com validação por Zod
- Extração de entidades de conversas em tempo real
- Persistência de preferências de usuário em SQLite com Knex
- Prompt engineering para extração de dados e sumarização de conversas
- Gerenciamento de sessões por `thread_id` no LangGraph
