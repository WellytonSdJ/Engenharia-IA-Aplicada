# Por onde começar

Este é o segundo projeto do módulo 3 (MCP na Prática). Se você está chegando agora, leia nesta ordem.

---

## O que estamos construindo e por quê

> Estamos construindo um agente que responde "esse título de vídeo está em alta?" com **dados reais**, não com a opinião do modelo.

O projeto pega uma pergunta livre do usuário ("Estou pensando em criar um vídeo sobre Web AI, quais títulos você me recomendaria?"), extrai palavras-chave, consulta o Google Trends de verdade via SerpAPI, e só então gera uma recomendação — o LLM não "chuta" se um tema está em alta, ele consulta uma tool que traz números.

Diferente do `01-multiple-mcp-tools` (um único nó autônomo que decide toda a sequência de execução sozinho), aqui a estrutura volta a ser **prompt chaining explícito**: dois nós fixos, sempre na mesma ordem, sem o modelo decidir "o quê" fazer entre eles — só *dentro* do nó `researcher` o modelo tem liberdade de decidir extrair palavras-chave e chamar a tool.

```
01-multiple-mcp-tools: 1 nó, agente decide toda a sequência do pipeline.
02-google-trends-agent (aqui): 2 nós fixos (researcher → responder),
                                 e só o researcher usa tool calling.
```

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [google-trends-tool.md](./google-trends-tool.md) | A peça central do projeto: como o Google Trends vira uma tool do LangChain, e o que acontece quando a API externa falha ou está desligada. |
| 2 | [prompt-chaining-researcher-responder.md](./prompt-chaining-researcher-responder.md) | Como os 2 nós do grafo se conectam e por que essa cadeia é linear, sem condicionais. |
| 3 | [structured-output-zod.md](./structured-output-zod.md) | Onde entram (e onde na prática *não* entram) os schemas Zod deste projeto — inclui um achado importante sobre código que existe mas não é usado. |
| 4 | [glossario.md](./glossario.md) | Referência rápida de todos os termos novos. Consulte quando encontrar algo que não reconhece. |

---

## Mapa do código

```
data/trendingData.ts                    → fixtures de tendência (alta/queda) usadas quando o SerpAPI está desabilitado ou falha

src/config.ts                           → configuração do modelo (OpenRouter) e do SerpAPI (apiKey, cacheTTL, disabled)
src/index.ts                            → sobe o servidor Fastify e dispara uma chamada de exemplo a /chat
src/server.ts                           → Fastify: POST /chat valida o body e invoca o grafo

src/graph/state.ts                      → GraphAnnotation: messages, trendsData, question, keywords
src/graph/nodes/researcherNode.ts       → chama o LLM em modo tools (MCP filesystem + google_trends)
src/graph/nodes/responderNode.ts        → gera a resposta final a partir de trendsData + question
src/graph/graph.ts                      → StateGraph linear de 2 nós: researcher → responder
src/graph/factory.ts                    → monta o grafo com o OpenRouterService

src/prompts/v1/keywords.ts              → KeywordsSchema (Zod, não usado no momento) + prompt do researcher
src/prompts/v1/videoTrends.ts           → VideoTrendsSchema (Zod, não usado no momento) + prompts do responder

src/services/openRouterService.ts       → ChatOpenAI + createAgent (2 modos: schema vs. tools)
src/services/mcpService.ts              → combina o servidor MCP filesystem + a tool google_trends num único array
src/services/serpApiService.ts          → chama o SerpAPI (engine google_trends), faz parsing e fallback para fixture

src/tools/googleTrendsTool.ts           → tool nativa do LangChain (não-MCP) que expõe o SerpAPIService ao LLM

langgraph.json                          → configuração do LangGraph Studio (graph `google_trends`)
```

---

## O fluxo em uma linha

```
POST /chat → researcher (extrai keywords + chama a tool google_trends) → responder (gera a recomendação) → resposta
```

Não há bordas condicionais nem loop: se o `researcher` falhar, ele captura o erro e segue adiante com uma mensagem de erro em `trendsData` — o `responder` roda de qualquer forma e tenta responder com o que tiver.

---

## Como rodar e ver o que importa

```bash
npm install
npm start
```

Acompanhe o terminal: você vai ver `🔍 Researcher processing...` → (chamada da tool `google_trends`, com log `🔍 Fetching Google Trends data for keywords: [...]`) → `💬 Responder processing...` → a resposta final. Se `config.serpAPIConfig.disabled` estiver `true` em `src/config.ts`, você verá `⚠️  SerpAPIService is disabled. Returning fixture data.` em vez de uma chamada real ao SerpAPI.
