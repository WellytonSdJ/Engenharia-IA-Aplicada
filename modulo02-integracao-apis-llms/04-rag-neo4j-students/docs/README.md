# Documentação — RAG Neo4j Students

Documentação de estudo do projeto `04-rag-neo4j-students`.

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura ordenada, mapa do código, fluxo do projeto |
| [rag-com-grafos.md](./rag-com-grafos.md) | O que é RAG, por que grafos, como Neo4j e Cypher se encaixam |
| [langgraph-pipeline.md](./langgraph-pipeline.md) | StateGraph, nós, edges condicionais, estado do grafo, roteamento |
| [text-to-cypher.md](./text-to-cypher.md) | Como o LLM gera, valida e corrige queries Cypher — o núcleo do sistema |
| [multi-step.md](./multi-step.md) | Decomposição de perguntas complexas em sub-queries sequenciais |
| [glossario.md](./glossario.md) | Todos os termos do módulo com definições — referência rápida |

---

## Contexto do projeto

API HTTP de análise de vendas de uma academia online com:

- **LangGraph** orquestrando 6 nós em um pipeline de RAG com autocorreção
- **Neo4j** armazenando o grafo de alunos, cursos, compras e progresso
- **LangChain** e **OpenRouter** para chamadas ao LLM com saída estruturada (Zod)
- **Fastify** servindo o endpoint `POST /sales`
- **Cypher** como linguagem intermediária entre pergunta em linguagem natural e banco de grafos
