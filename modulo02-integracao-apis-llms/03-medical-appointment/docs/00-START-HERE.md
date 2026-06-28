# Por onde começar

Este projeto combina pela primeira vez **LangGraph** com **chamadas reais ao LLM**. No projeto anterior (`02-langchain-intro`) o grafo existia mas os nós eram funções puras sem acesso a nenhum modelo. Aqui, dois nós chamam o LLM — um para entender o que o usuário quer, outro para gerar a resposta.

---

## O que estamos construindo e por quê

O padrão central deste projeto chama-se **prompt chaining** — encadear múltiplas chamadas ao LLM em sequência, onde a saída de uma alimenta a próxima.

```
Usuário: "quero cancelar minha consulta com a Dra. Ana amanhã às 14h"
         ↓
[LLM #1 — identifyIntent]
  Recebe: texto livre
  Retorna: { intent: "cancel", professionalId: 2, datetime: "...", patientName: "..." }
         ↓
[CÓDIGO — cancellerNode]
  Recebe: dados estruturados
  Executa: appointmentService.cancelAppointment(...)
  Retorna: { actionSuccess: true } ou { actionError: "..." }
         ↓
[LLM #2 — messageGeneratorNode]
  Recebe: resultado da ação + contexto
  Retorna: mensagem amigável em português
         ↓
Resposta: "Sua consulta com a Dra. Ana Pereira foi cancelada com sucesso."
```

A lógica de negócio (checar disponibilidade, inserir na lista) é **determinística** — código TypeScript comum. O LLM faz o que ele faz bem: entender linguagem natural e gerar texto. Essa separação é o padrão arquitetural mais importante deste módulo.

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [prompt-chaining.md](./prompt-chaining.md) | O padrão central do projeto. Entenda o pipeline antes de entrar no código de cada nó. |
| 2 | [structured-output.md](./structured-output.md) | Como o LLM retorna dados estruturados em vez de texto livre. É o que torna o prompt chaining confiável: sem isso, você teria que parsear texto de LLM manualmente. |
| 3 | [node-factory-pattern.md](./node-factory-pattern.md) | Como os nós recebem o LLM client e o AppointmentService como dependência — necessário para entender por que os nós são funções que retornam funções. |
| 4 | [domain-vs-llm-service.md](./domain-vs-llm-service.md) | A separação entre o serviço de agendamento (lógica de negócio) e o serviço de LLM (chamadas ao modelo). |
| 5 | [glossario.md](./glossario.md) | Termos novos deste projeto. |

Para fundamentos de LangGraph (StateGraph, nodes, edges), consulte o projeto anterior: [`02-langchain-intro/docs/langgraph-intro.md`](../../02-langchain-intro/docs/langgraph-intro.md).

---

## Mapa do código

```
src/config.ts                              → config do OpenRouter (modelos, temperatura, headers)
src/services/openRouterService.ts          → wrapper do ChatOpenAI com método generateStructured
src/services/appointmentService.ts         → CRUD in-memory de consultas + lista de profissionais

src/prompts/v1/identifyIntent.ts           → schema Zod (IntentSchema) + templates de prompt para extração de intent
src/prompts/v1/messageGenerator.ts         → schema Zod (MessageSchema) + templates para geração de resposta

src/graph/graph.ts                         → estado (AppointmentStateAnnotation) + montagem do grafo
src/graph/nodes/identifyIntentNode.ts      → chama LLM para extrair intent e dados da consulta
src/graph/nodes/schedulerNode.ts           → valida campos e agenda via appointmentService
src/graph/nodes/cancellerNode.ts           → valida campos e cancela via appointmentService
src/graph/nodes/messageGeneratorNode.ts    → chama LLM para gerar resposta amigável
src/graph/factory.ts                       → exporta o grafo compilado para uso externo

src/index.ts                               → CLI: lê pergunta do stdin, invoca o grafo, exibe resposta
tests/router.e2e.test.ts                   → testes E2E que chamam a API real do OpenRouter
```

---

## O fluxo em uma linha

```
START → identifyIntent → [schedule | cancel | message] → message → END
```

O nó `identifyIntent` (LLM) preenche `state.intent`. A edge condicional roteia para `schedule` ou `cancel` (código) ou diretamente para `message` (se intent desconhecida). Todos convergem para o nó `message` (LLM) que gera a resposta final.

---

## Como rodar e ver o que importa

```bash
# 1. Instalar dependências
npm install

# 2. Configurar variáveis de ambiente
cp .env.example .env
# preencha: OPENROUTER_API_KEY

# 3. Rodar interativamente via CLI
npm run dev
# Digite: "schedule an appointment with Dr. Alicio tomorrow at 2pm for João Silva"
# Observe os logs: 🔍 Identifying intent... → ✅ → 📅 Scheduling... → 💬 Generating...

# 4. Rodar testes E2E (chama API real — requer OPENROUTER_API_KEY)
npm test
```

Observe os logs do console ao rodar. Cada nó emite emoji de status (🔍 ✅ 📅 💬 ❌) que tornam o fluxo do grafo visível em tempo real.
