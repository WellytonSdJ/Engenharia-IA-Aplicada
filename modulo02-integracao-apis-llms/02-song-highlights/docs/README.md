# Documentação — Song Highlights

Documentação de estudo do projeto `02-song-highlights`, cobrindo os conceitos aplicados até o momento.

## Índice

| Documento | O que cobre |
|---|---|
| [langchain.md](./langchain.md) | O que é LangChain, abstrações core, ChatOpenAI, mensagens, structured output |
| [langgraph.md](./langgraph.md) | StateGraph, nós, edges, reducers, checkpointer, store, Runtime |
| [conversation-summarization.md](./conversation-summarization.md) | Por que sumarizar, sumarização incremental, como está implementado aqui |
| [persistence.md](./persistence.md) | Dois sistemas de persistência do projeto (Postgres vs SQLite), quando usar cada um |
| [prompt-injection.md](./prompt-injection.md) | O que é, vetores de ataque, como proteger, onde o projeto está exposto |

## Contexto do projeto

Chatbot de recomendação musical com:
- **LangGraph** orquestrando o fluxo entre nós (chat → savePreferences → summarize)
- **OpenRouter** como gateway de LLMs (via interface compatível com OpenAI)
- **LangChain** fornecendo as abstrações de mensagens, modelos e structured output
- **Postgres** para memória de sessão (checkpointer/store do LangGraph)
- **SQLite** para preferências persistidas entre sessões (via Knex)
