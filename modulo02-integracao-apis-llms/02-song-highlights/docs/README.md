# Documentação — Song Highlights

Documentação de estudo do projeto `02-song-highlights`.

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura ordenada, mapa do código, fluxo do projeto |
| [langchain.md](./langchain.md) | LangChain, ChatOpenAI, mensagens, structured output, OpenRouter, temperatura |
| [langgraph.md](./langgraph.md) | StateGraph, nós, edges, reducers, checkpointer, store, Runtime, LangSmith |
| [conversation-summarization.md](./conversation-summarization.md) | Por que sumarizar, sumarização incremental, RemoveMessage, dois sistemas de persistência do resumo |
| [persistence.md](./persistence.md) | Postgres (LangGraph) vs SQLite (preferências), quando usar cada um |
| [prompt-injection.md](./prompt-injection.md) | Vetores de ataque, guardrails, MCP, mitigações, exposições no projeto |
| [glossario.md](./glossario.md) | Todos os termos do módulo com definições — referência rápida |

---

## Contexto do projeto

Chatbot de recomendação musical com:

- **LangGraph** orquestrando o fluxo entre nós (chat → savePreferences → summarize)
- **OpenRouter** como gateway de LLMs (via interface compatível com OpenAI)
- **LangChain** fornecendo as abstrações de mensagens, modelos e structured output
- **Postgres** para memória de sessão (checkpointer/store do LangGraph)
- **SQLite** para preferências persistidas entre sessões (via Knex)
