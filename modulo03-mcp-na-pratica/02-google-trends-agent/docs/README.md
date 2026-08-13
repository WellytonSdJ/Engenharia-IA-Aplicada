# Documentação — Google Trends Agent

Documentação de estudo do projeto `02-google-trends-agent`, o segundo do módulo 3 (MCP na Prática).

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura ordenada, mapa do código, fluxo do projeto |
| [google-trends-tool.md](./google-trends-tool.md) | O SerpAPI embrulhado como tool do LangChain (`google_trends`), schema Zod de entrada, parsing dos dados e fallback para fixture local |
| [structured-output-zod.md](./structured-output-zod.md) | O ternário `providerStrategy` vs `tools` no `OpenRouterService`, os schemas Zod definidos no projeto e por que hoje eles não são usados nas chamadas reais |
| [prompt-chaining-researcher-responder.md](./prompt-chaining-researcher-responder.md) | O grafo linear de 2 nós (`researcher` → `responder`), sem condicionais nem loop, e como o dado flui entre eles |
| [glossario.md](./glossario.md) | Todos os termos novos deste projeto — referência rápida |

---

## Contexto do projeto

Agente que responde perguntas de estratégia de conteúdo para criadores de vídeo, cruzando a pergunta do usuário com dados reais do Google Trends, com:

- **LangGraph** orquestrando um prompt chaining explícito de 2 nós: pesquisa (`researcher`) e resposta (`responder`)
- Uma **tool nativa do LangChain** (`google_trends`) que embrulha o SerpAPI, com schema Zod validando o input
- **`@langchain/mcp-adapters`** conectando o servidor MCP filesystem (herdado do padrão do projeto `01-multiple-mcp-tools`, mas não usado pelo domínio deste projeto)
- **LangChain** (`createAgent`) e **OpenRouter** para as chamadas ao LLM
- **Fastify** servindo o endpoint `POST /chat`
- Fixture local (`data/trendingData.ts`) como fallback quando o SerpAPI está desabilitado ou falha
