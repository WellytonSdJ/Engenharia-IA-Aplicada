# Rate Limiting

## O que é

Rate limiting é o mecanismo que limita quantas requisições um cliente pode fazer em um determinado período. Sem isso, um único cliente (ou um bug em loop) pode sobrecarregar o servidor para todos os outros. Com rate limiting: quando o limite é atingido, o servidor rejeita novas requisições com HTTP 429 (Too Many Requests) até a janela de tempo resetar.

---

## Como está configurado neste projeto

```javascript
// nodejs-fastify-mongodb-crud-z/src/config.js
export const REQUESTS_PER_MINUTE = 90   // limite por janela de 1 minuto
```

```javascript
// nodejs-fastify-mongodb-crud-z/src/auth.js
export const rateLimitOptions = {
    max: REQUESTS_PER_MINUTE,     // máximo de requisições por janela
    timeWindow: '1 minute',       // janela de tempo
    // keyGenerator: define como identificar cada "cliente" para o rate limiter
    keyGenerator: (request) =>
        request.headers?.authorization?.replace(/bearer /i, '') ?? request.ip,
}
```

O `keyGenerator` é a parte mais importante: ele determina **por quem** o limite é aplicado. Aqui, é por token de autorização — cada token tem sua própria janela de 90 req/min. Se não houver token (rotas públicas), usa o IP como fallback. Isso significa:
- `admin` com 90 req/min **e** `member` com 90 req/min são janelas separadas
- O mesmo token gasta da mesma cota, independente de quem está usando

---

## Registro do plugin

```javascript
// nodejs-fastify-mongodb-crud-z/src/index.js
import fastifyRateLimit from '@fastify/rate-limit'

await fastify.register(fastifyRateLimit, rateLimitOptions)
```

`fastify.register()` é o sistema de plugins do Fastify. Ao registrar `@fastify/rate-limit`, ele adiciona automaticamente um `onRequest` hook que conta as requisições por `keyGenerator` e responde com 429 quando o limite é atingido. Nenhuma rota precisa ser alterada.

---

## Como o MCP server lida com o 429

No `CustomerHttpClient`, `#assertOk()` mapeia o status 429 para `RateLimitError`:

```typescript
// customers-mcp-z/src/infrastructure/customer-http-client.ts
async #assertOk(res: Response): Promise<void> {
    if (res.status === 401) throw new UnauthorizedError();
    if (res.status === 403) throw new ForbiddenError();
    if (res.status === 429) throw new RateLimitError();
    if (!res.ok) throw new Error(`HTTP ${res.status} - ${res.statusText} - ${await res.text()}`);
}
```

O `RateLimitError` sobe pela cadeia de chamadas até o handler da tool, que o captura e devolve `isError: true`:

```typescript
// qualquer tool (ex: list-customers.ts)
async () => {
    try {
        const customers = await service.listCustomers();
        return { content: [...], structuredContent: { customers } };
    } catch (err) {
        const message = `Failed to list customers. Error: ${err instanceof Error ? err.message : String(err)}`;
        return {
            content: [{ type: "text", text: message }],
            structuredContent: { isError: true, message },   // LLM vê: "Rate limit exceeded..."
        };
    }
}
```

---

## Como os testes verificam o rate limit

```typescript
// customers-mcp-z/tests/tools/customers.test.ts
it('should reach rate limit', async () => {
    let result;
    for (let index = 0; index < 100; index++) {
        result = await client.callTool({ name: 'list_customers', arguments: {} })
        if (result.structuredContent.isError) break;  // para quando atingir o limite
    }

    assert.ok(result.structuredContent.isError, 'Should return isError: true for rate limit exceeded');
    assert.strictEqual(
        result.structuredContent.message,
        'Failed to list customers. Error: Rate limit exceeded. Please try again later.'
    )
})
```

O teste faz até 100 chamadas em loop rápido — com limite de 90/min, ele deve atingir o 429 antes de terminar. O `break` interrompe assim que o erro é detectado, e os asserts verificam que tanto `isError` quanto a mensagem correta chegaram.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Limite configurado | [nodejs-fastify-mongodb-crud-z/src/config.js](../../nodejs-fastify-mongodb-crud-z/src/config.js) | `REQUESTS_PER_MINUTE = 90` |
| `keyGenerator` por token | [nodejs-fastify-mongodb-crud-z/src/auth.js](../../nodejs-fastify-mongodb-crud-z/src/auth.js) | `rateLimitOptions` com fallback para IP |
| Registro do plugin | [nodejs-fastify-mongodb-crud-z/src/index.js](../../nodejs-fastify-mongodb-crud-z/src/index.js) | `fastify.register(fastifyRateLimit, rateLimitOptions)` |
| Tratamento de 429 na infra | [src/infrastructure/customer-http-client.ts](../src/infrastructure/customer-http-client.ts) | `#assertOk()` mapeia 429 → `RateLimitError` |
| Teste de esgotamento | [tests/tools/customers.test.ts](../tests/tools/customers.test.ts) | Loop de 100 chamadas com break no primeiro erro |
