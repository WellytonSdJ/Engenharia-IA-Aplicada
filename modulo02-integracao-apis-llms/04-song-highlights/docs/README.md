# Documentação — Song Highlights

Documentação de estudo do projeto `02-song-highlights`.

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura ordenada, mapa do código, fluxo do projeto |
| [01-langchain.md](./01-langchain.md) | LangChain, ChatOpenAI, mensagens, structured output, OpenRouter, temperatura |
| [02-langgraph.md](./02-langgraph.md) | StateGraph, nós, edges, reducers, checkpointer, store, Runtime, LangSmith |
| [03-conversation-summarization.md](./03-conversation-summarization.md) | Por que sumarizar, sumarização incremental, RemoveMessage, dois sistemas de persistência do resumo |
| [04-persistence.md](./04-persistence.md) | Postgres (LangGraph) vs SQLite (preferências), quando usar cada um |
| [05-prompt-injection.md](./05-prompt-injection.md) | Vetores de ataque, guardrails, MCP, mitigações, exposições no projeto |
| [06-glossario.md](./06-glossario.md) | Todos os termos do módulo com definições — referência rápida |

---

## Contexto do projeto

Chatbot de recomendação musical com:

- **LangGraph** orquestrando o fluxo entre nós (chat → savePreferences → summarize)
- **OpenRouter** como gateway de LLMs (via interface compatível com OpenAI)
- **LangChain** fornecendo as abstrações de mensagens, modelos e structured output
- **Postgres** para memória de sessão (checkpointer/store do LangGraph)
- **SQLite** para preferências persistidas entre sessões (via Knex)

---

**➡️ Próximo passo: [00 · Por onde começar](./00-START-HERE.md)**
