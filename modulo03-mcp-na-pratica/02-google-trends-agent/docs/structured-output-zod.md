# Structured output com Zod — o que existe vs. o que roda

## O mecanismo: `generateStructured` ternário

`src/services/openRouterService.ts` tem um único método, `generateStructured`, com um comportamento que muda conforme você passa (ou não) um terceiro argumento opcional, `schema`:

```typescript
async generateStructured<T>(
    systemPrompt: string,
    userPrompt: string,
    schema?: z.ZodSchema<T>,
): Promise<{ data?: T | string; }> {
    if (!this.tools.length) {
        this.tools = await getMCPTools();
    }

    const agentConfig = schema ?
        { responseFormat: providerStrategy(schema), tools: [] }   // saída validada por Zod, sem tools
        : { tools: this.tools };                                  // tool calling livre, sem validação de saída

    const agent = createAgent({ ...agentConfig, model: this.llmClient });
    // ...
    return {
        data: (schema ?
            ((data as any).structuredResponse as T) :
            data.messages.at(-1)?.text
        ),
    };
}
```

Esse é o mesmo padrão usado em `01-multiple-mcp-tools`: um método, dois modos de `createAgent` — **saída estruturada** (`providerStrategy(schema)`, sem tools) ou **tool calling** (`tools`, sem `responseFormat`). Passar ou não um schema decide qual dos dois o agente vai rodar.

## O que este projeto define, mas não conecta

Existem dois schemas Zod prontos no projeto:

```typescript
// src/prompts/v1/keywords.ts
export const KeywordsSchema = z.object({
    keywords: z.array(z.string()).describe('The 2 most relevant search keywords extracted from the user question to query Google Trends.'),
});

// src/prompts/v1/videoTrends.ts
export const VideoTrendsSchema = z.object({
    answer: z.string().describe('A clear and concise analysis of the video title idea based on Google Trends data, including whether it is trending, stable, or declining, along with actionable recommendations.'),
});
```

Só que nenhum dos dois nós do grafo os passa para `generateStructured`:

```typescript
// researcherNode.ts — só 2 argumentos, schema fica undefined
const result = await openRouterService.generateStructured(
    getKeywordsSystemPrompt(),
    userQuestion,
);

// responderNode.ts — mesma coisa
const { data } = await openRouterService.generateStructured(
    getResponderSystemPrompt(),
    getResponderUserPrompt(userQuestion, state.trendsData ?? ''),
);
```

Com `schema` sempre `undefined`, o ternário do `agentConfig` **sempre** cai no ramo `{ tools: this.tools }` — os dois nós rodam em modo tool calling, nunca em modo `providerStrategy`. Na prática:

- `researcherNode` deixa o modelo livre para chamar `google_trends` (é isso que ele precisa fazer) — faz sentido rodar em modo tools.
- `responderNode` também roda em modo tools, mesmo não precisando chamar nenhuma tool para gerar a resposta final — ele só não usa esse acesso porque o prompt não pede.

Ou seja: `KeywordsSchema` e `VideoTrendsSchema` são scaffolding — schemas escritos seguindo o mesmo padrão de `01-multiple-mcp-tools` (que os usa de verdade em `intentNode.ts` via `IntentSchema`), mas que neste projeto ainda não foram conectados. O retorno de ambos os nós é sempre texto livre (`data.messages.at(-1)?.text`), não um objeto validado por `structuredResponse`.

## Onde a validação por Zod realmente acontece

A validação estruturada que de fato está ativa neste projeto é outra: o **schema de input da tool `google_trends`** (`z.object({ keywords: z.array(z.string()) })`, em `src/tools/googleTrendsTool.ts`). Isso é o LangChain garantindo que, quando o modelo decide chamar a tool, o argumento chega tipado e validado — ver [`google-trends-tool.md`](./google-trends-tool.md).

## Se for conectar os schemas existentes

Para `KeywordsSchema` virar validação real, seria passar `KeywordsSchema` como terceiro argumento em `researcherNode.ts` — mas isso mudaria o comportamento do nó, porque no modo `providerStrategy` a lista de `tools` fica vazia (`tools: []`), então o modelo não conseguiria mais chamar `google_trends` dentro do mesmo agent. Extrair keywords estruturadas e chamar a tool exigiriam duas chamadas separadas ao LLM (uma para extrair `keywords` via schema, outra para chamar a tool com esse array) — não é uma troca de uma linha.
