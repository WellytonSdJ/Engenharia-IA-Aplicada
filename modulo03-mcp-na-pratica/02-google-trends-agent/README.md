# Google Trends Agent

Agente LangGraph que responde perguntas de estratégia de conteúdo para criadores de vídeo ("esse título está em alta?", "que tópicos eu deveria explorar?"), usando o Google Trends (via SerpAPI) como fonte real de dados através de uma tool do LangChain.

## O que o projeto faz

- Recebe uma pergunta em linguagem natural sobre um tema/título de vídeo via `POST /chat` (Fastify)
- Um primeiro nó (`researcher`) delega a um agente LangChain (`createAgent`) a extração de palavras-chave e a chamada da tool `google_trends`, que busca dados reais no SerpAPI (volume de busca, tendência de alta/queda, tópicos relacionados em alta)
- Um segundo nó (`responder`) usa os dados coletados para gerar uma recomendação de conteúdo em português, com base em dados reais em vez de opinião do modelo
- Se o SerpAPI estiver desabilitado (`serpAPIConfig.disabled`) ou falhar, o serviço cai para uma fixture local (`data/trendingData.ts`) em vez de quebrar o fluxo
- O mesmo `MultiServerMCPClient` que conecta o servidor MCP filesystem também é usado aqui (herdado do padrão do projeto `01-multiple-mcp-tools`), mas a tool de domínio deste projeto (`google_trends`) é uma tool nativa do LangChain, não uma tool MCP

## Arquitetura do grafo

Diferente do `01-multiple-mcp-tools` (um agente autônomo com um único nó de execução), este projeto é um **prompt chaining linear e explícito** de 2 nós, sem bordas condicionais e sem loop:

```
START
  │
  ▼
researcher ──► responder ──► END
```

### Nós

| Nó | O que faz |
| --- | --- |
| `researcher` | Recebe a última mensagem do usuário e chama `OpenRouterService.generateStructured(...)` sem schema — o que faz o agente rodar em **modo tools** (MCP filesystem + `google_trends`). O modelo decide extrair as palavras-chave e chamar `google_trends` uma única vez, guiado só pelo system prompt. O texto final do agente é salvo em `state.trendsData`. |
| `responder` | Recebe `question` + `trendsData` e chama `generateStructured(...)` novamente, também sem schema, para gerar a resposta final em português — uma análise dos dados de tendência com recomendações acionáveis. O texto vira a última `AIMessage` do grafo. |

> Nota importante: os dois nós usam o **mesmo método** `generateStructured`, que tem um comportamento ternário — chamado com um schema Zod ele usa `providerStrategy` (saída estruturada); chamado sem schema (como acontece nos dois nós deste projeto) ele usa `tools`. Os schemas Zod deste projeto (`KeywordsSchema`, `VideoTrendsSchema`) existem em `src/prompts/v1/`, mas nenhum dos dois nós os passa como argumento — hoje eles não afetam o comportamento do agente. Veja [`docs/structured-output-zod.md`](./docs/structured-output-zod.md) para o detalhe.

## Como rodar

### Pré-requisitos

- Node.js >= 24.10.0
- Conta [OpenRouter](https://openrouter.ai/) (LLM)
- Conta [SerpAPI](https://serpapi.com/) (dados reais do Google Trends — opcional se `serpAPIConfig.disabled = true`, aí usa a fixture local)

### Instalação

```bash
npm install
```

### Variáveis de ambiente

Copie `.env.example` para `.env` e preencha:

| Variável | Obrigatória | Descrição |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Sim | Chave de acesso à API OpenRouter (LLM) |
| `SERPAPI_API_KEY` | Sim* | Chave do SerpAPI para consultar o Google Trends real. *Só é usada se `config.serpAPIConfig.disabled` for `false` em `src/config.ts` |
| `LANGSMITH_API_KEY` | Não | Habilita rastreamento com LangSmith |
| `LANGCHAIN_TRACING_V2` | Não | Ativa o tracing (`true`/`false`) |
| `LANGCHAIN_PROJECT` | Não | Nome do projeto no LangSmith |

### Executar

```bash
npm start
```

Sobe o servidor Fastify em `http://0.0.0.0:3000` e, no próprio `src/index.ts`, dispara automaticamente uma chamada de exemplo a `POST /chat` para demonstrar o fluxo no console.

Para testar manualmente:

```bash
curl -X POST \
  -H 'Content-type: application/json' \
  --data '{"question": "Estou pensando em criar um video sobre Web AI, quais titulos você me recomendaria?"}' \
  localhost:3000/chat
```

### LangGraph Studio

```bash
npm run langgraph:serve
```

Usa a configuração em `langgraph.json`, que expõe o grafo `google_trends` a partir de `src/graph/factory.ts`.

### Testes

Scripts de teste (`npm test`, `npm run test:unit`, `npm run test:e2e`) estão declarados no `package.json`, mas não há arquivos em `tests/` neste momento.

## Documentação de conceitos

Documentação aprofundada em [`docs/`](./docs/) — comece por [`docs/00-START-HERE.md`](./docs/00-START-HERE.md).

| Documento | Conteúdo |
| --- | --- |
| [google-trends-tool.md](./docs/google-trends-tool.md) | Como o SerpAPI é embrulhado como tool do LangChain (`google_trends`), o schema Zod de entrada, e o fallback para fixture local |
| [structured-output-zod.md](./docs/structured-output-zod.md) | O mecanismo `providerStrategy` vs `tools` do `OpenRouterService`, os schemas Zod definidos no projeto, e por que eles não estão conectados às chamadas atuais |
| [prompt-chaining-researcher-responder.md](./docs/prompt-chaining-researcher-responder.md) | O grafo linear de 2 nós (sem condicionais, sem loop) e como o texto flui de um nó ao outro via `state.trendsData` |
| [glossario.md](./docs/glossario.md) | Termos novos deste projeto |
