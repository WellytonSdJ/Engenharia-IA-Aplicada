# Testes E2E e Injeção de Dependência

## O que é

Duas ideias trabalhando juntas neste projeto:

1. **Teste E2E (end-to-end)**: em vez de testar uma função isolada, o teste sobe o
   comportamento completo do sistema — da requisição HTTP até a chamada real à API do
   OpenRouter — e verifica o resultado observável (o que a rota `/chat` devolve).
2. **Injeção de dependência**: em vez do código de produção decidir sozinho qual configuração
   usar, ele recebe essa configuração de fora (por parâmetro/construtor). Isso permite trocar
   o comportamento em teste sem duplicar ou reescrever o código de produção.

A runner de testes usada é o `node:test`, nativo do Node.js — não há Jest, Vitest ou Mocha
como dependência.

## Como está sendo usado neste projeto

`OpenRouterService` aceita uma configuração opcional no construtor:

```typescript
// src/openrouterService.ts
constructor(configOverride?: ModelConfig) {
  this.config = configOverride ?? config;
  // ...
}
```

E `createServer` aceita a instância do serviço já pronta, em vez de criar a sua própria:

```typescript
// src/server.ts
export const createServer = (routerService: OpenRouterService) => {
  // ...
};
```

Isso significa que o código de produção (`src/index.ts`) e os testes podem montar o mesmo
servidor com configurações diferentes, sem que `server.ts` precise saber a diferença:

```typescript
// tests/router.e2e.test.ts
test('routes to cheapest model by default', async () => {
  const customConfig = {
    ...config,
    provider: {
      ...config.provider,
      sort: { ...config.provider.sort, by: 'price' }
    }
  };
  const routerService = new OpenRouterService(customConfig);
  const app = createServer(routerService);

  const response = await app.inject({
    method: 'POST',
    url: '/chat',
    body: { question: 'What is rate limiting?' }
  });

  assert.equal(response.statusCode, 200);
  const body = response.json() as LLMResponse;
  assert.equal(body.model, 'nvidia/nemotron-3.5-content-safety-20260604:free');
});
```

Note que o teste não faz mock do OpenRouter — ele chama a API de verdade (por isso o projeto
exige `OPENROUTER_API_KEY` válida para rodar os testes) e verifica, no `body.model` da
resposta real, que o critério de roteamento configurado (`price` ou `throughput`) de fato
influenciou qual modelo foi escolhido. É um teste E2E no sentido literal: cobre o caminho
inteiro, sem dublês.

## `app.inject` — chamando a rota sem subir uma porta

`app.inject(...)` é um recurso do Fastify que simula uma requisição HTTP completa (método,
url, body) diretamente contra o app montado em memória, sem precisar de `app.listen(...)` nem
de um cliente HTTP real (`fetch`, `axios`). Isso torna o teste mais rápido e evita conflito de
porta entre execuções.

## Por que isso importa

Sem a injeção de dependência, o único jeito de testar "o que acontece quando o critério é
`price`" seria editar `config.ts` manualmente antes de rodar, testar, e desfazer a edição — um
processo manual e frágil. Com `configOverride`, cada teste monta seu próprio cenário de forma
isolada e explícita, e o código de produção (`index.ts`) continua usando a config real sem
nenhuma alteração.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Configuração injetável | `src/openrouterService.ts` | Parâmetro `configOverride` no construtor |
| Serviço injetável na rota | `src/server.ts` | Parâmetro `routerService` em `createServer` |
| Cenários de teste com config alternativa | `tests/router.e2e.test.ts` | `customConfig` com `sort.by` sobrescrito |
| Chamada de rota sem subir porta | `tests/router.e2e.test.ts` | `app.inject({...})` |
| Runner nativo do Node.js | `package.json` | Scripts `test` e `test:dev` usando `node --test` |
