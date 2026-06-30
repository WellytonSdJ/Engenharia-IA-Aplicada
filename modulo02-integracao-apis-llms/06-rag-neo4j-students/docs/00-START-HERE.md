# Por onde começar

Este projeto implementa uma API de análise de vendas usando RAG sobre um banco de grafos Neo4j.
Se você está chegando agora, leia nesta ordem:

---

## Trilha de leitura

### 1. Contexto — o que estamos construindo e por quê

Antes de qualquer código, entenda o cenário:

> Estamos construindo um **pipeline de RAG sobre grafo** — o usuário faz uma pergunta em linguagem natural,
> o sistema converte essa pergunta em Cypher (a linguagem de query do Neo4j), executa no banco,
> e devolve uma resposta analítica em prosa.

O que diferencia esse projeto de um chatbot comum não é a geração de texto. É a capacidade de transformar linguagem natural em código executável, validar esse código, corrigi-lo quando falha, e sintetizar múltiplos resultados em uma resposta coerente.

```
Pergunta em português → Cypher → Neo4j → Prosa analítica
```

---

### 2. Leia os documentos nesta ordem

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [rag-com-grafos.md](./rag-com-grafos.md) | Fundação: o que é RAG, por que grafo é diferente de vetor, como o Neo4j organiza o domínio de vendas. Sem isso, o propósito do projeto fica vago. |
| 2 | [langgraph-pipeline.md](./langgraph-pipeline.md) | A arquitetura central: como o StateGraph orquestra os 6 nós, o que é o estado compartilhado, como as edges condicionais decidem o fluxo. |
| 3 | [text-to-cypher.md](./text-to-cypher.md) | O núcleo técnico: como o LLM gera Cypher a partir de schema real, como a validação com `EXPLAIN` funciona, e como a autocorreção fecha o loop quando a query falha. |
| 4 | [multi-step.md](./multi-step.md) | Como perguntas complexas são decompostas em sub-queries sequenciais e os resultados são sintetizados ao final. |
| 5 | [glossario.md](./glossario.md) | Referência rápida de todos os termos. Consulte quando encontrar algo que não reconhece. |

---

### 3. Mapa do código

Depois de ler os docs, o código vai fazer mais sentido nesta ordem:

```
src/config.ts                                   → limites: tentativas de correção, sub-perguntas máximas, modelo LLM
src/services/neo4jService.ts                    → conexão lazy com Neo4j, schema, validação, execução de queries
src/services/openrouterService.ts               → LLM via OpenRouter com saída estruturada (Zod)
src/graph/graph.ts                              → definição do GraphState e montagem do StateGraph
src/graph/nodes/extractQuestionNode.ts          → extrai a pergunta da última mensagem
src/graph/nodes/queryPlannerNode.ts             → analisa complexidade: simples vs. multi-step
src/graph/nodes/cypherGeneratorNode.ts          → gera Cypher com schema real + contexto de negócio
src/graph/nodes/cypherExecutorNode.ts           → valida com EXPLAIN, executa, detecta erros
src/graph/nodes/cypherCorrectionNode.ts         → reescreve query com base no erro do Neo4j
src/graph/nodes/analyticalResponseNode.ts       → gera resposta em prosa; trata erro, sem resultados, multi-step
src/prompts/v1/queryAnalyzer.ts                 → schema + prompt para análise de complexidade
src/prompts/v1/cypherGenerator.ts               → schema + prompt para geração de Cypher (com exemplos)
src/prompts/v1/cypherCorrection.ts              → schema + prompt para correção de Cypher
src/prompts/v1/analyticalResponse.ts            → schema + 4 templates de resposta
src/prompts/v1/salesContext.ts                  → regras de negócio injetadas nos prompts de Cypher
src/graph/factory.ts                            → monta serviços e grafo
src/server.ts                                   → Fastify com rota POST /sales
src/index.ts                                    → entry point, sobe servidor e faz requisição de teste
```

---

### 4. O fluxo em uma linha

```
pergunta → extrair → planejar → gerar Cypher → executar → [corrigir?] → responder
```

O grafo sempre começa em `extractQuestion`. As edges condicionais após `cypherExecutor` decidem:
corrigir query com erro, buscar próximo step de uma query multi-step, ou gerar a resposta final.

---

### 5. Conceitos do curso que este projeto cobre

Este projeto é o módulo de RAG com grafos. Os conceitos implementados aqui:

- **Módulo 6 (RAG com Neo4j):** Query Planner, Cypher Generator, Cypher Correction — o núcleo deste projeto
- **LangGraph:** StateGraph com edges condicionais e estado compartilhado (ver módulo 2)
- **Structured Output com Zod:** validação de respostas LLM em todos os nós (ver módulo 2)
- **Self-correction:** loop automático de correção quando a query gerada é inválida — padrão agentic
