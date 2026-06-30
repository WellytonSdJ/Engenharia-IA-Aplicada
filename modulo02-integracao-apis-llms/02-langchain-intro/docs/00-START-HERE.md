# Por onde começar

Este projeto introduz o **LangGraph** — o framework de grafos de estado para fluxos com LLMs. Não existe LLM aqui ainda; o objetivo é entender a mecânica antes de adicionar a variabilidade de um modelo.

---

## O que estamos construindo e por quê

No projeto anterior (`01-smart-model-router-gateway`) você chamava a API do OpenRouter diretamente: uma requisição entrava, você montava o payload, chamava o modelo e retornava a resposta. Funcionou. Mas esse padrão escala mal quando o fluxo precisa de ramificações — "se o usuário quer X, faz A; se quer Y, faz B".

LangGraph resolve isso com **grafos de estado**: você define nós (funções) e arestas (quem chama quem), e o framework executa o fluxo correto automaticamente.

```
Sem LangGraph:              Com LangGraph:
if intent == "upper"        START
  return upper(text)          → identifyIntent
elif intent == "lower"          → [condicional]
  return lower(text)              uppercase → chatResponse → END
else                              lowercase → chatResponse → END
  return fallback                 fallback  → chatResponse → END
```

À esquerda: lógica de fluxo misturada com lógica de negócio. À direita: o grafo separa cada responsabilidade em um nó isolado. Quando o fluxo crescer (mais intenções, loops, retry), o grafo escala; o if/else vira espaguete.

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [langgraph-intro.md](./langgraph-intro.md) | A base: o que é `StateGraph`, como o estado funciona com Zod, como nós e edges são definidos. Leia antes do código — os arquivos fazem muito mais sentido depois. |
| 2 | [langchain-messages.md](./langchain-messages.md) | Por que o estado guarda `messages: BaseMessage[]` em vez de `messages: string[]`. O sistema de mensagens do LangChain é o formato padrão que todos os projetos seguintes vão usar. |
| 3 | [glossario.md](./glossario.md) | Referência rápida dos termos novos. |

---

## Mapa do código

Leia nesta ordem depois dos docs:

```
src/graph/graph.ts                        → define o estado (GraphState) e monta o grafo
src/graph/nodes/identifyIntentNode.ts     → nó de entrada: detecta comando por substring
src/graph/nodes/upperCaseNode.ts          → transforma state.output para maiúsculas
src/graph/nodes/lowerCaseNode.ts          → transforma state.output para minúsculas
src/graph/nodes/fallbackNode.ts           → resposta padrão para comandos desconhecidos
src/graph/nodes/chatResponseNode.ts       → empacota state.output como AIMessage
src/graph/factory.ts                      → exporta a função graph() para o LangGraph server
src/server.ts                             → Fastify: POST /chat → graph.invoke() → retorna output
src/index.ts                              → ponto de entrada, sobe o servidor na porta 3000
tests/router.e2e.test.ts                  → testa os 3 caminhos (upper, lower, fallback)
```

---

## O fluxo em uma linha

```
POST /chat → identifyIntent → [upper | lower | fallback] → chatResponse → output
```

O nó `identifyIntent` lê a última mensagem, detecta a intenção por substring e armazena `command` no estado. A edge condicional usa `state.command` para rotear para o nó correto. Todos convergem para `chatResponse` que empacota o resultado como `AIMessage`.

---

## Como rodar e ver o que importa

```bash
# 1. Instalar dependências
npm install

# 2. Rodar os testes (não precisa de .env — nenhuma API externa)
npm test

# 3. Subir o servidor e testar manualmente
cp .env.example .env   # preencha LANGSMITH_API_KEY se quiser rastreamento
npm run dev

# Em outro terminal:
curl localhost:3000/chat \
  --data '{"question": "make this UPPER please"}' \
  -H "Content-type: application/json"
# → "MAKE THIS UPPER PLEASE"

curl localhost:3000/chat \
  --data '{"question": "convert to lower"}' \
  -H "Content-type: application/json"
# → "convert to lower"

curl localhost:3000/chat \
  --data '{"question": "hey there"}' \
  -H "Content-type: application/json"
# → "Unknown command. Try 'make this uppercase' or 'convert to lowercase'"
```

Os testes cobrem exatamente esses três cenários — leia `tests/router.e2e.test.ts` para ver como o Fastify é testado com `.inject()` sem precisar subir o servidor de verdade.
