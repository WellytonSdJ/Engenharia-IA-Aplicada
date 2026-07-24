# Módulo 03 — MCP na Prática

Aprofundamento em **Model Context Protocol (MCP)** — o protocolo aberto que padroniza como LLMs se conectam a ferramentas, dados e sistemas externos. O módulo 02 já havia introduzido MCP de forma pontual (um único servidor filesystem, em `05-safeguard-prompt-injection`); este módulo assume esse conceito como conhecido e vai fundo: múltiplos servidores MCP simultâneos, agentes com autonomia real de orquestração, MCP do zero, MCP como camada de modernização de APIs legadas, segurança e governança (RBAC, JWT, Service Tokens, rate limiting), publicação de servidores MCP e integração com LangChain.

Material de referência do curso disponível em [`docs/`](./docs/) (dois PDFs fornecidos pela pós-graduação — não confundir com a pasta `docs/` de cada subprojeto, que é a documentação de estudo gerada para este repositório).

## Projetos

| # | Projeto | Status | Descrição |
|---|---------|--------|-----------|
| 01 | [multiple-mcp-tools](./01-multiple-mcp-tools/) | ✅ concluído | Agente autônomo que resolve um pipeline de dados inteiro (CSV → JSON → MongoDB → relatório) combinando 2 servidores MCP (filesystem + MongoDB) com uma tool nativa do LangChain, orquestrado por `createAgent` em vez de nós de grafo explícitos |

## Roteiro do módulo (orientação, não implementado ainda)

Os próximos subprojetos deste módulo já têm pasta reservada no repositório original do curso. A lista abaixo é só para orientação — cada um será documentado a fundo somente quando for de fato estudado e trazido para este repositório.

| # | Pasta de referência | Tema (a partir do nome/README disponível) |
|---|---------------------|--------------------------------------------|
| 02 | `02-google-trends-agent` | Agente com prompt chaining e structured output (Zod), usando um serviço externo (Google Trends) como tool |
| 03 | `03-dev-instructions-agents` | Instruções para agentes de desenvolvimento (ex: `.github/` — copilot/agent instructions) |
| 04 | `04-skills` | Skills — outra forma de empacotar capacidades para agentes, em contraste com tools/MCP |
| 05 | `05-mcps-do-zero` | Construção de um servidor MCP do zero (`ciphersuite-mcp`): tools, resources, prompts, testes automatizados via MCP client |
| 06 | `06-your-legacy-api-as-mcp` | Transformar uma API legada (Fastify + MongoDB CRUD) em um servidor MCP — abstração de domínio em vez de espelhar endpoints |
| 07 | `07-api-security-auth-rate-limiting` | Segurança e governança: RBAC, JWT, Service Tokens, rate limiting — aplicado tanto na Web API quanto no servidor MCP |
| 08 | `08-publishing-mcps-private-npm` | Publicação de servidores MCP no NPM Registry (público) e Verdaccio (privado); outros transports além de STDIO |
| 09 | `09-using-mcp-with-langchain` | Uso do MCP construído (customers MCP server) como tool de um agente LangChain.js |

## Requisitos gerais

| Requisito | Versão mínima | Observação |
| --- | --- | --- |
| **Node.js** | 24.10.0+ | Execução nativa de `.ts` sem compilação (`node --env-file`) |
| **npm** | 10+ | — |
| **Docker + Docker Compose** | 20+ / 2+ | MongoDB (projeto 01); demais bancos conforme cada subprojeto futuro |
| **Conta OpenRouter** | — | Acesso ao LLM via API compatível com OpenAI |
| **Conta LangSmith** | — | Opcional — tracing do grafo e do agente |

> Cada projeto tem seu próprio `README.md`/`docs/` com requisitos detalhados, variáveis de ambiente e instruções de execução.

## Conceitos abordados

- MCP (Model Context Protocol): servidores, clientes, tools, resources e prompts como contrato padronizado entre LLMs e sistemas externos
- Múltiplos servidores MCP conectados simultaneamente no mesmo cliente (`MultiServerMCPClient`)
- Combinação de tools vindas de MCP com tools nativas do LangChain no mesmo agente
- Orquestração autônoma: delegar ao modelo a decisão de qual tool chamar e em qual ordem, em vez de nós de grafo explícitos
- Observabilidade do loop de decisões do agente via callbacks do LangChain (`handleChatModelStart`, `handleLLMEnd`, `handleToolStart`, `handleToolEnd`)
- (Roteiro futuro) Construção de servidores MCP do zero: tools, resources, prompts e testes automatizados via MCP client
- (Roteiro futuro) Transformação de APIs legadas em servidores MCP, com abstração de domínio
- (Roteiro futuro) Segurança e governança em MCP: RBAC, autenticação JWT, Service Tokens, rate limiting
- (Roteiro futuro) Publicação e distribuição de servidores MCP (NPM Registry / Verdaccio) e diferentes transports
- (Roteiro futuro) Integração de servidores MCP com agentes LangChain.js

## Documentação de conceitos (projeto 01)

Documentação aprofundada dos conceitos aplicados disponível em [`01-multiple-mcp-tools/docs/`](./01-multiple-mcp-tools/docs/):

| Documento | Conteúdo |
| --- | --- |
| [mcp-multiplos-servidores.md](./01-multiple-mcp-tools/docs/mcp-multiplos-servidores.md) | Dois servidores MCP (filesystem + MongoDB) no mesmo cliente, e uma tool nativa do LangChain misturada com tools de MCP |
| [agente-autonomo-vs-orquestracao-explicita.md](./01-multiple-mcp-tools/docs/agente-autonomo-vs-orquestracao-explicita.md) | Por que o grafo tem só 2 nós e como o agente passa a decidir sozinho a sequência de execução do pipeline |
| [observabilidade-agent-loop.md](./01-multiple-mcp-tools/docs/observabilidade-agent-loop.md) | Callbacks do LangChain para enxergar o loop de decisões do agente em tempo real |
