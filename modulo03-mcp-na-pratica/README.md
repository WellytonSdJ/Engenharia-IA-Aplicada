# Módulo 03 — MCP na Prática

Aprofundamento em **Model Context Protocol (MCP)** — o protocolo aberto que padroniza como LLMs se conectam a ferramentas, dados e sistemas externos. O módulo 02 já havia introduzido MCP de forma pontual (um único servidor filesystem, em `05-safeguard-prompt-injection`); este módulo assume esse conceito como conhecido e vai fundo: múltiplos servidores MCP simultâneos, agentes com autonomia real de orquestração, MCP do zero, MCP como camada de modernização de APIs legadas, segurança e governança (RBAC, JWT, Service Tokens, rate limiting), publicação de servidores MCP e integração com LangChain.

Material de referência do curso disponível em [`docs/`](./docs/) (dois PDFs fornecidos pela pós-graduação — não confundir com a pasta `docs/` de cada subprojeto, que é a documentação de estudo gerada para este repositório).

## Projetos

| # | Projeto | Status | Descrição |
|---|---------|--------|-----------|
| 01 | [multiple-mcp-tools](./01-multiple-mcp-tools/) | ✅ concluído | Agente autônomo que resolve um pipeline de dados inteiro (CSV → JSON → MongoDB → relatório) combinando 2 servidores MCP (filesystem + MongoDB) com uma tool nativa do LangChain, orquestrado por `createAgent` em vez de nós de grafo explícitos |
| 02 | [google-trends-agent](./02-google-trends-agent/) | ✅ concluído | Agente LangGraph de prompt chaining linear (`researcher` → `responder`) que responde perguntas de estratégia de conteúdo usando dados reais do Google Trends via SerpAPI, embrulhado como tool nativa do LangChain |
| 03 | [dev-instructions-agents](./03-dev-instructions-agents/) | ✅ concluído | Instruções declarativas (`.agent.md`) para agentes de desenvolvimento do GitHub Copilot: um agente genérico de código e um pipeline de 3 agentes (planner → generator → healer) para automação de testes Playwright |
| 04 | [skills](./04-skills/) | ✅ concluído | Agent Skills — pacotes de instrução em Markdown (`SKILL.md`) carregados sob demanda pelo próprio agente, em contraste com tools/MCP; 3 skills instaladas (`ffmpeg`, `find-skills`, `neo4j-cypher-guide`) |
| 05 | [mcps-do-zero](./05-mcps-do-zero/) | ✅ concluído | Servidor MCP construído do zero (`ciphersuite-mcp`) com o SDK oficial: tools de criptografia AES-256-CBC, resource, prompt e testes via `Client` MCP real sobre transporte STDIO |

## Roteiro do módulo (orientação, não implementado ainda)

Os próximos subprojetos deste módulo já têm pasta reservada no repositório original do curso. A lista abaixo é só para orientação — cada um será documentado a fundo somente quando for de fato estudado e trazido para este repositório.

| # | Pasta de referência | Tema (a partir do nome/README disponível) |
|---|---------------------|--------------------------------------------|
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
| **Conta OpenRouter** | — | Acesso ao LLM via API compatível com OpenAI (projetos 01 e 02) |
| **Conta SerpAPI** | — | Dados reais do Google Trends (projeto 02; opcional, com fallback para fixture local) |
| **Conta LangSmith** | — | Opcional — tracing do grafo e do agente |

> Cada projeto tem seu próprio `README.md`/`docs/` com requisitos detalhados, variáveis de ambiente e instruções de execução.

## Conceitos abordados

- MCP (Model Context Protocol): servidores, clientes, tools, resources e prompts como contrato padronizado entre LLMs e sistemas externos
- Múltiplos servidores MCP conectados simultaneamente no mesmo cliente (`MultiServerMCPClient`)
- Combinação de tools vindas de MCP com tools nativas do LangChain no mesmo agente
- Orquestração autônoma: delegar ao modelo a decisão de qual tool chamar e em qual ordem, em vez de nós de grafo explícitos
- Observabilidade do loop de decisões do agente via callbacks do LangChain (`handleChatModelStart`, `handleLLMEnd`, `handleToolStart`, `handleToolEnd`)
- Prompt chaining linear (sem condicionais/loop) combinando uma tool nativa do LangChain com dados de uma API externa (SerpAPI/Google Trends)
- `providerStrategy` vs. `tools` no mesmo método de geração — como a presença/ausência de um schema Zod muda o comportamento do agente
- Agentes de desenvolvimento declarativos do GitHub Copilot (`.agent.md`): persona, ferramentas restritas e fluxo de trabalho definidos em Markdown, sem código
- Pipeline de agentes para automação de testes E2E (planner → generator → healer) com Playwright
- Agent Skills como alternativa ao MCP: instruções em Markdown carregadas sob demanda pelo próprio agente, sem servidor nem protocolo, instaladas/versionadas por um gerenciador de skills (lockfile)
- Construção de servidores MCP do zero: `McpServer`, `registerTool`, `registerResource`, `registerPrompt`, transporte STDIO e testes via `Client` MCP real
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

## Documentação de conceitos (projeto 02)

Documentação aprofundada dos conceitos aplicados disponível em [`02-google-trends-agent/docs/`](./02-google-trends-agent/docs/):

| Documento | Conteúdo |
| --- | --- |
| [google-trends-tool.md](./02-google-trends-agent/docs/google-trends-tool.md) | Como o SerpAPI é embrulhado como tool do LangChain (`google_trends`), o schema Zod de entrada, e o fallback para fixture local |
| [structured-output-zod.md](./02-google-trends-agent/docs/structured-output-zod.md) | O mecanismo `providerStrategy` vs. `tools` do `OpenRouterService`, os schemas Zod definidos no projeto, e por que eles não estão conectados às chamadas atuais |
| [prompt-chaining-researcher-responder.md](./02-google-trends-agent/docs/prompt-chaining-researcher-responder.md) | O grafo linear de 2 nós (sem condicionais, sem loop) e como o texto flui de um nó ao outro via `state.trendsData` |

## Documentação de conceitos (projeto 03)

Documentação aprofundada dos conceitos aplicados disponível em [`03-dev-instructions-agents/docs/`](./03-dev-instructions-agents/docs/):

| Documento | Conteúdo |
| --- | --- |
| [custom-agents-copilot.md](./03-dev-instructions-agents/docs/custom-agents-copilot.md) | Formato `.agent.md`: frontmatter, persona, ferramentas restritas — instruções declarativas em vez de código |
| [pipeline-playwright-agents.md](./03-dev-instructions-agents/docs/pipeline-playwright-agents.md) | Como `playwright-test-planner`, `playwright-test-generator` e `playwright-test-healer` colaboram num pipeline sequencial de automação de testes |

## Documentação de conceitos (projeto 04)

Documentação aprofundada dos conceitos aplicados disponível em [`04-skills/docs/`](./04-skills/docs/):

| Documento | Conteúdo |
| --- | --- |
| [formato-skill-md.md](./04-skills/docs/formato-skill-md.md) | Estrutura de um `SKILL.md`: frontmatter, corpo, arquivos de `references/`, e como o agente decide carregar cada um |
| [skills-vs-mcp-e-gerenciamento.md](./04-skills/docs/skills-vs-mcp-e-gerenciamento.md) | Skills como alternativa ao MCP, e como elas são instaladas/versionadas (Skills CLI, `skills-lock.json`, `skills.sh`) |

## Documentação de conceitos (projeto 05)

Documentação aprofundada dos conceitos aplicados disponível em [`05-mcps-do-zero/docs/`](./05-mcps-do-zero/docs/):

| Documento | Conteúdo |
| --- | --- |
| [construindo-mcp-server-do-zero.md](./05-mcps-do-zero/docs/construindo-mcp-server-do-zero.md) | Como um servidor MCP é construído com o SDK oficial: `McpServer`, `registerTool`, `registerResource`, `registerPrompt` |
| [transporte-stdio-e-testes-mcp-client.md](./05-mcps-do-zero/docs/transporte-stdio-e-testes-mcp-client.md) | Transporte STDIO (servidor como subprocesso) e como os testes conectam um `Client` MCP real ao servidor para validar o comportamento ponta a ponta |
| [criptografia-aes-256-cbc.md](./05-mcps-do-zero/docs/criptografia-aes-256-cbc.md) | As decisões de criptografia por trás das tools: derivação de chave com `scrypt`, IV aleatório por chamada, formato de saída |
