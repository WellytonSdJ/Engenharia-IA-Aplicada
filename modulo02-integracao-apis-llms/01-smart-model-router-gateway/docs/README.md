# Documentação — Smart Model Router Gateway

Documentação de estudo do projeto `01-smart-model-router-gateway`, o primeiro projeto do
módulo 02.

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

## O que o projeto demonstra

Um gateway HTTP construído com Fastify que recebe perguntas via `POST /chat` e as encaminha,
através do OpenRouter, para o melhor modelo LLM dentro de uma lista de candidatos — "melhor"
definido por um critério configurável (preço, throughput ou latência). O foco do projeto é
o **roteamento entre modelos**, não a construção de fluxos conversacionais complexos (isso
começa no projeto 02 com LangGraph).

## Documentos

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura, mapa do código, como rodar |
| [model-routing.md](./model-routing.md) | O conceito central: roteamento de modelos por preço, throughput ou latência |
| [openrouter-sdk.md](./openrouter-sdk.md) | Como o `@openrouter/sdk` unifica o acesso a múltiplos provedores de LLM |
| [fastify.md](./fastify.md) | Servidor HTTP com validação de schema nativa |
| [testes-e2e-injecao-dependencia.md](./testes-e2e-injecao-dependencia.md) | Testes E2E com `node:test` e injeção de dependência via `configOverride` |
| [glossario.md](./glossario.md) | Referência rápida dos termos novos |

## Contexto do projeto

| Tecnologia | Papel no projeto |
| --- | --- |
| **Fastify** | Servidor HTTP, rota `POST /chat`, validação de schema do body |
| **OpenRouter (`@openrouter/sdk`)** | Acesso unificado a múltiplos modelos LLM e roteamento entre eles |
| **TypeScript** | Tipagem do projeto, executado nativamente pelo Node.js sem compilação prévia |
| **Node.js `--experimental-strip-types`** | Execução direta de `.ts` sem `tsc`/`ts-node` |
| **`node:test`** | Runner de testes E2E nativo do Node.js |
