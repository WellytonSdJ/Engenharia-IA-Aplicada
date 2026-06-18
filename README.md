# Engenharia de IA Aplicada

Repositório de estudos da pós-graduação em **Engenharia de IA Aplicada**. Cada pasta contém um projeto independente desenvolvido ao longo do curso.

## Módulos

### Módulo 1 — Redes Neurais

| Projeto | Descrição |
| ------- | --------- |
| [01-neural-network](./01-neural-network/) | Rede neural com TensorFlow.js que classifica usuários em categorias (premium, medium, basic) |

### Módulo 2 — Sistemas de Recomendação

| Projeto | Descrição |
| ------- | --------- |
| [02-Sistemas-Recomendacao](./02-Sistemas-Recomendacao/) | Sistema de recomendação de produtos para e-commerce usando TensorFlow.js com Web Worker, arquitetura MVC e codificação one-hot de features |

### Módulo 3 — Integração com APIs de LLMs

| Projeto | Descrição |
| ------- | --------- |
| [01-smart-model-router-gateway](./03-integracao-apis-llms/01-smart-model-router-gateway/) | Gateway HTTP com Fastify que roteia requisições para o melhor modelo LLM via OpenRouter com base em critérios configuráveis (preço, throughput ou latência) |

## Stack utilizada

- **TensorFlow.js** — redes neurais e sistemas de recomendação no browser e Node.js
- **Node.js** + **TypeScript** — backend e integrações com APIs
- **Fastify** — servidor HTTP com validação de schema
- **OpenRouter** — roteamento multi-modelo para LLMs
- **Web Workers** — treinamento de modelos sem bloquear a UI
