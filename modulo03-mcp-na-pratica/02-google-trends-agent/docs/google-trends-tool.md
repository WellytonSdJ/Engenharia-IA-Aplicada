# A tool `google_trends`

## O problema

O LLM sozinho não sabe se um tema está "em alta" agora — seu conhecimento é estático e não tem acesso a dados de busca em tempo real. Para o `researcher` dar uma recomendação de conteúdo confiável, ele precisa de um dado externo real: o Google Trends.

## A tool

`src/tools/googleTrendsTool.ts` embrulha o `SerpAPIService` como uma tool nativa do LangChain (`tool()` de `@langchain/core/tools` — não é uma tool MCP, é código que roda in-process, sem subprocesso):

```typescript
export function createGoogleTrendsTool(serpAPIService: SerpAPIService) {
  return tool(
    async ({ keywords }) => {
      const data = await serpAPIService.getGoogleTrends(keywords);
      return JSON.stringify(data);
    },
    {
      name: 'google_trends',
      description:
        'Get Google Trends data for a list of keywords. Use this to analyze if a video title or topic is trending, rising, or declining in popularity. Always call this when the user shares a video title idea.',
      schema: z.object({
        keywords: z.array(z.string()).describe('Keywords extracted from the video title to analyze'),
      }),
    },
  );
}
```

O `schema` Zod aqui é o que efetivamente restringe a saída do modelo neste projeto: o LLM só consegue chamar essa tool passando um array de strings em `keywords` — não texto livre, não um objeto qualquer. É a única validação estruturada por Zod que está realmente "no caminho" da execução (compare com [`structured-output-zod.md`](./structured-output-zod.md), que trata dos schemas que existem mas não são usados).

A `description` da tool também é o principal mecanismo de controle de comportamento: é ela que instrui o modelo a *sempre* chamar a tool quando o usuário compartilha uma ideia de título — não há verificação de código que force essa chamada.

## Como a tool chega ao agente

`src/services/mcpService.ts` monta a lista de tools disponíveis para o `createAgent`:

```typescript
export const getMCPTools = async () => {
  const mcpClient = new MultiServerMCPClient({
    filesystem: { transport: 'stdio', command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', process.cwd()] },
  });
  const mcpTools = await mcpClient.getTools();

  const serpAPIService = new SerpAPIService(config.serpAPIConfig);
  const googleTrendsTool = createGoogleTrendsTool(serpAPIService);

  return [...mcpTools, googleTrendsTool];
};
```

O servidor MCP filesystem é conectado aqui (mesmo padrão de `01-multiple-mcp-tools`), mas nenhum prompt deste projeto instrui o modelo a ler/escrever arquivos — ele fica disponível como tool, só não é usado pelo domínio (recomendação de conteúdo). O array final mistura uma tool MCP com uma tool nativa, exatamente como em `01-multiple-mcp-tools`.

## `SerpAPIService`: dados reais, parsing e degradação

`src/services/serpApiService.ts` chama o SerpAPI (`engine: 'google_trends'`, janela `now 7-d`) uma vez por keyword e converte a resposta bruta em três estruturas:

| Campo | Como é calculado |
| --- | --- |
| `keywords[].trend` | Compara a média dos 3 últimos pontos da série temporal (`recentAvg`) com a média dos 3 primeiros (`earlyAvg`): `rising` se `recentAvg > earlyAvg * 1.2`, `declining` se `recentAvg < earlyAvg * 0.8`, senão `stable` |
| `relatedQueries` | Junta `related_queries.top` e `related_queries.rising` da resposta do SerpAPI, ordenado por valor, limitado a 10 |
| `risingTopics` | Extrai `related_topics.rising`, convertendo valores como `"+350%"` em número via regex, limitado a 5 |

Duas decisões de resiliência importantes:

- **Falha por keyword não derruba a busca inteira**: cada keyword é buscada num `try/catch` isolado dentro do `for`; se uma falhar, as outras continuam e o erro só vira um `console.warn`.
- **Fallback via `disabled`**: se `config.serpAPIConfig.disabled` for `true`, o método retorna direto `risingTrendFixture` (de `data/trendingData.ts`) sem nenhuma chamada de rede — útil para testar o grafo sem gastar quota do SerpAPI.

> Nota: a classe declara `private cache: Map<string, ...>` e a config tem `cacheTTL: 3600000`, mas nenhum método do serviço lê ou escreve nesse `Map` — hoje é campo morto, sem cache efetivo. Se for implementar cache de verdade, esse é o lugar.
