# LangChain Intro — Primeiro Grafo LangGraph

O menor projeto LangGraph possível que ainda faz algo real: um roteador de comandos de texto (uppercase / lowercase / fallback) exposto via API HTTP. Não há chamada a nenhum LLM — o objetivo é aprender a mecânica do `StateGraph` (nós, edges condicionais, state com Zod) sem a variabilidade de um modelo de linguagem.

## O que o projeto faz

- Recebe uma pergunta em texto livre via `POST /chat` (ex: "make this uppercase")
- `identifyIntent` decide deterministicamente (verificação de substring) se o comando é `uppercase`, `lowercase` ou `unknown`
- O nó correspondente transforma o texto; um roteamento condicional decide qual nó executar
- `chatResponse` empacota o resultado como `AIMessage` e finaliza o grafo

## Arquitetura — Grafo de Estados (LangGraph)

```
START
  │
  ▼
identifyIntent ──command?──► uppercase ──┐
                 ├──────────► lowercase ──┼──► chatResponse ──► END
                 └──────────► fallback  ──┘
```

### Nós

| Nó | O que faz |
| --- | --- |
| `identifyIntent` | Lê a última mensagem do usuário e define `command` (`uppercase`/`lowercase`/`unknown`) por verificação de substring |
| `uppercase` | Converte `output` para maiúsculas |
| `lowercase` | Converte `output` para minúsculas |
| `fallback` | Retorna mensagem de ajuda quando o comando não é reconhecido |
| `chatResponse` | Empacota `output` como `AIMessage` e acrescenta ao histórico `messages` |

### Roteamento condicional

Após `identifyIntent`, `addConditionalEdges` decide o próximo nó com base em `state.command`; os três caminhos convergem em `chatResponse` antes do `END`.

## State (Zod)

```typescript
const GraphState = z.object({
  messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta), // reducer de mensagens do LangGraph
  output: z.string(),
  command: z.enum(['uppercase', 'lowercase', 'unknown']),
})
```

## Como rodar

```bash
npm install
npm run dev
```

```bash
curl localhost:3000/chat \
  --data '{"question": "uppercase this"}' \
  -H "Content-type: application/json"
```

Também é possível inspecionar o grafo visualmente com o LangGraph Studio:

```bash
npm run langgraph:serve
```

## Documentação de conceitos

Documentação aprofundada em [`docs/`](./docs/) — comece por [`docs/00-START-HERE.md`](./docs/00-START-HERE.md).

| Documento | Conteúdo |
| --- | --- |
| [langgraph-intro.md](./docs/langgraph-intro.md) | StateGraph, state com Zod, nós, edges condicionais, `withLangGraph`, compilação do grafo |
| [langchain-messages.md](./docs/langchain-messages.md) | HumanMessage, AIMessage, BaseMessage, MessagesZodMeta, reducer de mensagens |
| [glossario.md](./docs/glossario.md) | Termos novos deste projeto — referência rápida |
