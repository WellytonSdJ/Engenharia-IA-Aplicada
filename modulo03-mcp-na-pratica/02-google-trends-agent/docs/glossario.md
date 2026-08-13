# Glossário

Referência rápida. Para profundidade, vá ao documento específico de cada conceito.

Termos de MCP (MCP, MCP Server, MCP Client, STDIO transport, Tool, Tool calling, Agente, Lazy initialization) e de LangGraph/LangChain/Zod genéricos (StateGraph, State, Node, Edge, `createAgent`, Zod, `z.infer`) já cobertos nos glossários de `05-safeguard-prompt-injection` (módulo 02) e [`01-multiple-mcp-tools`](../../01-multiple-mcp-tools/docs/glossario.md) não são repetidos aqui.

---

## Domínio (agente de tendências para vídeo)

| Termo | Definição |
| --- | --- |
| **SerpAPI** | Serviço externo que expõe dados do Google (incluindo Google Trends) via API HTTP. Usado neste projeto através do pacote `serpapi` (`getJson({ engine: 'google_trends', ... })`). |
| **`google_trends` (tool)** | Tool nativa do LangChain (`src/tools/googleTrendsTool.ts`) que embrulha o `SerpAPIService`. Recebe um array de `keywords` (validado por Zod) e devolve `TrendingData` serializado em JSON. |
| **`SerpAPIService`** | Classe (`src/services/serpApiService.ts`) que chama o SerpAPI por keyword, calcula tendência (`rising`/`stable`/`declining`) comparando médias recentes vs. iniciais da série temporal, e extrai `relatedQueries`/`risingTopics`. |
| **`TrendingData`** | Tipo de retorno de `SerpAPIService.getGoogleTrends`: `{ keywords, relatedQueries, risingTopics, timestamp }`. |
| **Fixture de tendência** | Dados fixos em `data/trendingData.ts` (`risingTrendFixture`, `decliningTrendFixture`) usados no lugar de uma chamada real quando `config.serpAPIConfig.disabled` é `true`. Só `risingTrendFixture` é efetivamente usada hoje pelo código. |
| **`serpAPIConfig.disabled`** | Flag em `src/config.ts` que, quando `true`, faz `SerpAPIService.getGoogleTrends` retornar direto a fixture, sem chamar a rede. |
| **`serpAPIConfig.cacheTTL` / `cache` (Map)** | Campos declarados em `SerpAPIService` para um cache com TTL, mas não implementados — nenhum método lê ou escreve nesse `Map` atualmente. |

---

## Grafo e prompt chaining

| Termo | Definição |
| --- | --- |
| **Prompt chaining linear** | Padrão deste projeto: grafo com bordas fixas (`START → researcher → responder → END`), sem `addConditionalEdges` e sem loop — contraste com um pipeline que repete etapas até uma condição de qualidade. |
| **`researcher` (nó)** | Primeiro nó do grafo. Roda o agente em modo tools para extrair palavras-chave da pergunta do usuário e chamar `google_trends`. Grava o texto final em `state.trendsData`. |
| **`responder` (nó)** | Segundo nó do grafo. Usa `trendsData` + `question` para gerar a recomendação final em português, também via `generateStructured` sem schema. |
| **`GraphAnnotation`** | Schema Zod do estado do grafo (`src/graph/state.ts`): `messages`, `trendsData`, `question`, `keywords` (este último declarado mas não escrito por nenhum nó hoje). |

---

## `OpenRouterService` e structured output

| Termo | Definição |
| --- | --- |
| **`generateStructured`** | Método único do `OpenRouterService` que monta `createAgent` de dois jeitos possíveis: com `responseFormat: providerStrategy(schema)` (schema Zod passado) ou com `tools` (schema omitido). Neste projeto, os dois nós do grafo sempre omitem o schema. |
| **`providerStrategy`** | Helper do pacote `langchain` que configura `createAgent` para pedir ao provedor (OpenRouter/OpenAI-compatible) uma saída validada por um schema Zod. Importado neste projeto, mas nunca efetivamente acionado nas chamadas atuais. |
| **`KeywordsSchema`** | Schema Zod (`src/prompts/v1/keywords.ts`) descrevendo `{ keywords: string[] }`. Existe como scaffolding para uma futura extração estruturada no `researcher`, mas não é passado a `generateStructured` hoje. |
| **`VideoTrendsSchema`** | Schema Zod (`src/prompts/v1/videoTrends.ts`) descrevendo `{ answer: string }`. Mesma situação: definido, mas não conectado à chamada do `responder`. |
| **Schema de input de tool** | A validação Zod que está de fato ativa no projeto: o argumento `keywords` da tool `google_trends`, que restringe o que o modelo pode passar ao chamá-la. |
