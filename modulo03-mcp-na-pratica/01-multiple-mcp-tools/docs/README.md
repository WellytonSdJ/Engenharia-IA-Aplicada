# Documentação — Multiple MCP Tools

Documentação de estudo do projeto `01-multiple-mcp-tools`, o primeiro do módulo 3 (MCP na Prática).

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura ordenada, mapa do código, fluxo do projeto |
| [mcp-multiplos-servidores.md](./mcp-multiplos-servidores.md) | Dois servidores MCP (filesystem + MongoDB) no mesmo cliente, e uma tool nativa do LangChain misturada com tools de MCP |
| [agente-autonomo-vs-orquestracao-explicita.md](./agente-autonomo-vs-orquestracao-explicita.md) | Por que o grafo tem só 2 nós e como o agente passa a decidir sozinho a sequência de execução do pipeline |
| [observabilidade-agent-loop.md](./observabilidade-agent-loop.md) | Callbacks do LangChain (`handleChatModelStart`, `handleLLMEnd`, `handleToolStart`, `handleToolEnd`) para enxergar o loop de decisões do agente |
| [glossario.md](./glossario.md) | Todos os termos novos deste projeto — referência rápida |

---

## Contexto do projeto

Agente de processamento de dados de vendas que resolve um pipeline completo (converter CSV → salvar JSON → inserir no MongoDB → consultar → escrever relatório) de forma autônoma, com:

- **LangGraph** orquestrando apenas 2 nós: extração de intenção e execução autônoma
- **`@langchain/mcp-adapters`** conectando dois servidores MCP simultâneos: `@modelcontextprotocol/server-filesystem` e `mongodb-mcp-server`
- Uma **tool nativa do LangChain** (`csv_to_json`) combinada com as tools vindas de MCP no mesmo agente
- **LangChain** (`createAgent`) e **OpenRouter** para chamadas ao LLM, com dois modos de uso: extração estruturada (`providerStrategy`) e execução com ferramentas (`tools`)
- **Callbacks do LangChain** para observar em tempo real quais tools o agente decide chamar
- **Fastify** servindo o endpoint `POST /chat`
- **MongoDB + mongo-express** via Docker Compose como backend de persistência e inspeção
