# Engenharia de IA Aplicada

Repositório de estudos da pós-graduação em **Engenharia de IA Aplicada**. Cada pasta corresponde a um módulo do curso e contém projetos desenvolvidos ao longo das aulas.

## Estrutura

```
engenharia-ia-aplicada/
├── modulo01-fundamentos-de-ia-e-llms-para-programadores/
│   ├── 01-neural-network/              # Classificador com TensorFlow.js
│   └── 02-Sistemas-Recomendacao/       # Recomendação de produtos no browser
└── modulo02-integracao-apis-llms/
    ├── 01-smart-model-router-gateway/  # Gateway HTTP com roteamento de modelos
    ├── 02-langchain-intro/             # Primeiro grafo LangGraph (sem LLM)
    ├── 03-medical-appointment/         # Prompt chaining + structured output
    ├── 04-song-highlights/             # Chatbot musical com memória (LangGraph)
    ├── 05-safeguard-prompt-injection/  # Guardrails e segurança com LLM
    └── 06-rag-neo4j-students/          # RAG com Neo4j e Cypher
```

## Módulos

### Módulo 01 — Fundamentos de IA e LLMs para Programadores

Introdução prática a redes neurais com TensorFlow.js, cobrindo arquitetura de modelos, codificação de features, treinamento no browser e inferência.

| Projeto | Descrição |
| ------- | --------- |
| [01-neural-network](./modulo01-fundamentos-de-ia-e-llms-para-programadores/01-neural-network/) | Rede neural que classifica usuários em categorias (premium, medium, basic) com TensorFlow.js |
| [02-Sistemas-Recomendacao](./modulo01-fundamentos-de-ia-e-llms-para-programadores/02-Sistemas-Recomendacao/) | Sistema de recomendação de produtos para e-commerce com TensorFlow.js, Web Worker e arquitetura MVC |

### Módulo 02 — Integração com APIs de LLMs

Integração de aplicações Node.js com APIs de modelos de linguagem, do roteamento inteligente até RAG com banco de grafos, passando por LangGraph, prompt chaining e segurança.

| Projeto | Descrição |
| ------- | --------- |
| [01-smart-model-router-gateway](./modulo02-integracao-apis-llms/01-smart-model-router-gateway/) | Gateway HTTP com Fastify que roteia requisições para o melhor modelo LLM via OpenRouter (por preço, throughput ou latência) |
| [02-langchain-intro](./modulo02-integracao-apis-llms/02-langchain-intro/) | Primeiro grafo LangGraph: StateGraph + Zod state + nós puros + edges condicionais, sem chamadas ao LLM |
| [03-medical-appointment](./modulo02-integracao-apis-llms/03-medical-appointment/) | Prompt chaining com LangGraph: LLM extrai intent, código agenda/cancela, LLM gera resposta — Zod structured output e factory pattern |
| [04-song-highlights](./modulo02-integracao-apis-llms/04-song-highlights/) | Chatbot musical CLI com grafo LangGraph, checkpointer Postgres, extração de preferências com Zod e persistência em SQLite |
| [05-safeguard-prompt-injection](./modulo02-integracao-apis-llms/05-safeguard-prompt-injection/) | Demonstração de ataques de prompt injection e defesa com guardrails baseados em LLM; RBAC admin vs member via MCP |
| [06-rag-neo4j-students](./modulo02-integracao-apis-llms/06-rag-neo4j-students/) | API de análise de vendas que c