# Por onde começar

Este projeto implementa um chatbot musical com memória persistente usando LangGraph.
Se você está chegando agora, leia nesta ordem:

---

## Trilha de leitura

### 1. Contexto — o que estamos construindo e por quê

Antes de qualquer código, entenda o cenário:

> Estamos construindo um **wrapper** — um produto que "embrulha" uma API de LLM
> e entrega uma experiência específica. Neste caso: recomendação musical com memória do usuário.

O que diferencia um wrapper fraco de um forte não é a chamada à API. Qualquer dev faz isso em dias.
O diferencial está em: saída estruturada, validação, segurança, memória e observabilidade.

```
LLM não é mágica. LLM não é solução isolada. LLM é componente.
```

---

### 2. Leia os documentos nesta ordem

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [langchain.md](./langchain.md) | Fundação: o que é LangChain, como funciona a integração com o modelo, o que é structured output. Sem isso, o código dos nós não faz sentido. |
| 2 | [langgraph.md](./langgraph.md) | Como o fluxo é orquestrado: state, nós, edges, checkpointer. É a arquitetura central do projeto. |
| 3 | [conversation-summarization.md](./conversation-summarization.md) | Por que e como o histórico é comprimido — o problema de memória que o projeto resolve. |
| 4 | [persistence.md](./persistence.md) | Os dois bancos de dados do projeto (Postgres e SQLite) e por que cada um existe. |
| 5 | [prompt-injection.md](./prompt-injection.md) | Como o sistema pode ser atacado, o que já está protegido, e o que falta. Leitura essencial antes de colocar qualquer chatbot em produção. |
| 6 | [glossario.md](./glossario.md) | Referência rápida de todos os termos. Consulte quando encontrar algo que não reconhece. |

---

### 3. Mapa do código

Depois de ler os docs, o código vai fazer mais sentido nesta ordem:

```
src/config.ts                          → configurações e limites do sistema
src/services/openrouterService.ts      → como o LLM é chamado
src/services/memoryService.ts          → configuração do Postgres (LangGraph)
src/services/preferencesService.ts     → SQLite, merge e leitura de preferências
src/graph/graph.ts                     → definição do state e montagem do grafo
src/graph/nodes/edgeConditions.ts      → quem decide para onde o grafo vai
src/graph/nodes/chatNode.ts            → nó principal: resposta + extração
src/graph/nodes/savePreferencesNode.ts → persiste preferências detectadas
src/graph/nodes/summarizationNode.ts   → comprime histórico quando necessário
src/prompts/v1/chatResponse.ts         → prompt do chat (schema + system prompt)
src/prompts/v1/summarization.ts        → prompt de sumarização
src/graph/factory.ts                   → monta tudo junto
src/index.ts                           → ponto de entrada, loop de conversa
```

---

### 4. O fluxo em uma linha

```
usuário → chatNode → [savePreferences?] → [summarize?] → resposta
```

Toda invocação começa no `chatNode`. As edges condicionais decidem os próximos passos
baseadas no que o LLM detectou na resposta (preferências novas? histórico longo?).

---

### 5. Conceitos do curso que vão além deste projeto

Este projeto cobre os módulos 2, 4 e 5 do curso. Os docs aqui focam no que está implementado.
O arquivo de resumo do Obsidian (`POS - MODULO 02 - RESUMO...`) cobre os demais módulos:
- **Módulo 3:** Prompt Chaining, sistema de agendamento com linguagem natural
- **Módulo 5:** Guardrails e MCP (Model Context Protocol) — leia [prompt-injection.md](./prompt-injection.md), seção Guardrails
- **Módulo 6:** RAG com Neo4j, Query Planner, Cypher Generator
- **Módulo 7:** Multimodalidade, Langfuse, Evaluation Tests
