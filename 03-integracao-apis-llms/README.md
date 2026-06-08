# Módulo 3 — Integração com APIs de LLMs

Este módulo explora como integrar aplicações Node.js com APIs de modelos de linguagem (LLMs), abordando roteamento inteligente, gerenciamento de providers e boas práticas de configuração.

## Projetos

| # | Projeto | Descrição |
|---|---|---|
| 01 | [Smart Model Router Gateway](./01-smart-model-router-gateway/) | Gateway HTTP que roteia para o melhor modelo LLM via OpenRouter com base em critérios configuráveis |

## Conceitos abordados

- Integração com a API OpenRouter para acesso multi-modelo
- Roteamento dinâmico por preço, throughput e latência
- Servidor HTTP com Fastify e validação de schema
- Execução de TypeScript nativa com Node.js (sem compilação)
- Testes E2E com `node:test` (runner nativo)
