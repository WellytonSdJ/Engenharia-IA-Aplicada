# Documentação — LangChain Intro

Documentação de estudo do projeto `02-langchain-intro`.

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura, mapa do código, como rodar e ver o que importa |
| [01-langgraph-intro.md](./01-langgraph-intro.md) | StateGraph, state com Zod, nós, edges condicionais, compilação do grafo |
| [02-langchain-messages.md](./02-langchain-messages.md) | HumanMessage, AIMessage, BaseMessage, MessagesZodMeta, por que mensagens em vez de strings |
| [03-glossario.md](./03-glossario.md) | Termos novos deste projeto — referência rápida |

---

## Contexto do projeto

O menor projeto LangGraph possível que ainda faz algo real: um roteador de comandos de texto (uppercase / lowercase / fallback) exposto via API HTTP.

Tecnologias:
- **LangGraph** (`@langchain/langgraph`) — orquestra o fluxo entre nós via `StateGraph`
- **LangChain** (`langchain`) — fornece os tipos de mensagem (`HumanMessage`, `AIMessage`)
- **Zod** — define e valida o estado do grafo em tempo de execução
- **Fastify** — servidor HTTP que recebe a pergunta e retorna a resposta

> Este projeto não chama nenhuma API de LLM. O "roteamento" é determinístico (verificação de substring). O objetivo é aprender a mecânica do LangGraph sem a variabilidade de um modelo.

---

**➡️ Próximo passo: [00 · Por onde começar](./00-START-HERE.md)**
