# Engenharia de IA Aplicada

Repositório de estudos da pós-graduação em **Engenharia de IA Aplicada**. Cada pasta corresponde a um módulo do curso e contém projetos desenvolvidos ao longo das aulas.

## Estrutura

```
engenharia-ia-aplicada/
├── modulo01-fundamentos-de-ia-e-llms-para-programadores/
│   ├── 01-neural-network/          # Classificador com TensorFlow.js
│   └── 02-Sistemas-Recomendacao/   # Recomendação de produtos no browser
└── modulo02-integracao-apis-llms/
    ├── 01-smart-model-router-gateway/  # Gateway HTTP com roteamento de modelos
    └── 02-song-highlights/             # Chatbot musical com memória (LangGraph)
```

## Módulos

### Módulo 01 — Fundamentos de IA e LLMs para Programadores

Introdução prática a redes neurais com TensorFlow.js, cobrindo arquitetura de modelos, codificação de features, treinamento no browser e inferência.

| Projeto | Descrição |
| ------- | --------- |
| [01-neural-network](./modulo01-fundamentos-de-ia-e-llms-para-programadores/01-neural-network/) | Rede neural que classifica usuários em categorias (premium, medium, basic) com TensorFlow.js |
| [02-Sistemas-Recomendacao](./modulo01-fundamentos-de-ia-e-llms-para-programadores/02-Sistemas-Recomendacao/) | Sistema de recomendação de produtos para e-commerce com TensorFlow.js, Web Worker e arquitetura MVC |

### Módulo 02 — Integração com APIs de LLMs

Integração de aplicações Node.js com APIs de modelos de linguagem, abordando roteamento inteligente, grafos de estado com LangGraph e persistência de memória.

| Projeto | Descrição |
| ------- | --------- |
| [01-smart-model-router-gateway](./modulo02-integracao-apis-llms/01-smart-model-router-gateway/) | Gateway HTTP com Fastify que roteia requisições para o melhor modelo LLM via OpenRouter (por preço, throughput ou latência) |
| [02-song-highlights](./modulo02-integracao-apis-llms/02-song-highlights/) | Chatbot musical CLI com grafo de estados LangGraph, extração de preferências com Zod e persistência em SQLite |

## Stack

- **TensorFlow.js** — redes neurais e sistemas de recomendação no browser e Node.js
- **Node.js** + **TypeScript** — backend e integrações com APIs (execução nativa de `.ts` sem compilação)
- **LangGraph** — grafos de estado para fluxos conversacionais com LLMs (nós, edges condicionais, checkpointer, store)
- **LangChain** — integração com modelos de linguagem e saída estruturada (`withStructuredOutput`)
- **Fastify** — servidor HTTP com validação de schema via JSON Schema
- **OpenRouter** — acesso a múltiplos modelos LLM com roteamento automático
- **Zod** — validação e tipagem de saídas estruturadas dos LLMs
- **PostgreSQL** — persistência de sessão do LangGraph (checkpointer e store)
- **SQLite / knex** — persistência leve de preferências de usuário entre sessões
- **Web Workers** — treinamento de modelos no browser sem bloquear a UI

## Conceitos de segurança abordados

- **Prompt injection** — vetores de ataque em sistemas com LLM, mitigações via structured output e separação de contextos (módulo 02)
