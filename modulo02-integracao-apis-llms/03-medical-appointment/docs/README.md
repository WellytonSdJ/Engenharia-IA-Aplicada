# Documentação — Medical Appointment

Documentação de estudo do projeto `03-medical-appointment`.

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura, mapa do código, fluxo do projeto |
| [01-prompt-chaining.md](./01-prompt-chaining.md) | O padrão central: múltiplas chamadas LLM em pipeline via grafo |
| [02-structured-output.md](./02-structured-output.md) | Zod schemas para forçar saída estruturada do LLM, `generateStructured`, `createAgent`, `providerStrategy` |
| [03-node-factory-pattern.md](./03-node-factory-pattern.md) | Como injetar dependências (LLM client, services) em nós via factory functions |
| [04-domain-vs-llm-service.md](./04-domain-vs-llm-service.md) | Separação entre `AppointmentService` (lógica de negócio) e `OpenRouterService` (chamadas ao LLM) |
| [05-glossario.md](./05-glossario.md) | Termos novos deste projeto — referência rápida |

---

## Contexto do projeto

Sistema de agendamento médico por linguagem natural. O usuário digita em linguagem livre ("quero marcar com o Dr. Alicio amanhã às 14h") e o sistema:

1. Usa um LLM para extrair a intenção e os dados estruturados (quem, quando, o quê)
2. Executa a ação de negócio (agendar ou cancelar) via código determinístico
3. Usa outro LLM para gerar a resposta amigável em português

Tecnologias:
- **LangGraph** (`@langchain/langgraph`) — orquestra o pipeline via `StateGraph` (ver [01-langgraph-intro.md](../../02-langchain-intro/docs/01-langgraph-intro.md))
- **LangChain** (`langchain`) — `ChatOpenAI`, `createAgent`, `providerStrategy`, tipos de mensagem
- **OpenRouter** — gateway de LLMs (já visto em `01-smart-model-router-gateway`)
- **Zod** — define e valida os schemas de saída esperados do LLM
- **AppointmentService** — CRUD in-memory de consultas médicas (sem banco de dados)

---

**➡️ Próximo passo: [00 · Por onde começar](./00-START-HERE.md)**
