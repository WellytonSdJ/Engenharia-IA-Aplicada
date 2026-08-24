# Erros de Domínio e Tratamento no MCP

## O problema

No projeto anterior (06), quando a API retornava um status inesperado, o `CustomerHttpClient` ou lançava uma exceção genérica (`Error: HTTP 401`) ou simplesmente ignorava o erro. Quem recebesse a exceção no handler da tool não tinha como saber se era falta de autenticação, falta de permissão ou outra coisa — e acabava devolvendo uma mensagem vaga ao LLM.

---

## A solução: hierarquia de erros de domínio

```typescript
// customers-mcp-z/src/domain/errors.ts
export class UnauthorizedError extends Error {
    constructor(message = 'Unauthorized: service token is missing or invalid') {
        super(message);
        this.name = 'UnauthorizedError';   // name é o que aparece na mensagem final ao LLM
    }
}

export class ForbiddenError extends Error {
    constructor(message = 'Forbidden: token does not have sufficient permissions') {
        super(message);
        this.name = 'ForbiddenError';
    }
}

export class RateLimitError extends Error {
    constructor(message = 'Rate limit exceeded. Please try again later.') {
        super(message);
        this.name = 'RateLimitError';
    }
}
```

Cada erro tem uma **semântica específica**: `UnauthorizedError` = token inválido/ausente (401), `ForbiddenError` = token válido mas role insuficiente (403), `RateLimitError` = muitas chamadas (429). Separar assim permite que o código que captura decida de forma precisa o que comunicar.

Os erros ficam em `domain/errors.ts` — não em `infrastructure/` — porque representam conceitos de negócio (não tenho permissão, o limite foi atingido), não detalhes de HTTP.

---

## `#assertOk()`: centralização do mapeamento HTTP → domínio

```typescript
// customers-mcp-z/src/infrastructure/customer-http-client.ts
async #assertOk(res: Response): Promise<void> {
    if (res.status === 401) throw new UnauthorizedError();
    if (res.status === 403) throw new ForbiddenError();
    if (res.status === 429) throw new RateLimitError();
    if (!res.ok) throw new Error(`HTTP ${res.status} - ${res.statusText} - ${await res.text()}`);
}
```

`#assertOk` é chamado depois de cada `fetch()`. O `#` é a sintaxe de **private field nativo do JavaScript** (não é o `private` do TypeScript): o método não existe fora da classe, nem em subclasses. O TypeScript `private` é removido em runtime; `#` é aplicado pelo próprio JavaScript engine.

Todos os métodos do client chamam `#assertOk` logo após a requisição:

```typescript
async listCustomers(): Promise<Customer[]> {
    const res = await fetch(`${this.baseUrl}/customers`, { headers: this.authHeaders });
    await this.#assertOk(res);     // lança o erro tipado se necessário
    return res.json() as Promise<Customer[]>;
}
```

Sem `#assertOk`, cada método teria que verificar `res.status` individualmente — 5 tools × 3 status codes = 15 verificações repetidas.

---

## Propagação: do HTTP ao LLM

A cadeia é: HTTP status → erro de domínio (infra) → exceção capturada no handler da tool → `isError: true` com mensagem para o LLM.

```
fetch() retorna 429
  → #assertOk() lança RateLimitError("Rate limit exceeded. Please try again later.")
    → CustomerService.listCustomers() deixa subir (sem try/catch — não é responsabilidade dela)
      → tool handler captura no catch:
          const message = `Failed to list customers. Error: ${err.message}`
          return { structuredContent: { isError: true, message } }
            → LLM recebe: isError: true, message: "Failed to list customers. Error: Rate limit exceeded..."
```

O `CustomerService` não captura — ele propaga. A responsabilidade de tratar para o usuário (neste caso, o LLM) é da camada MCP (o handler da tool), que sabe como formatar a resposta no protocolo correto.

---

## `beforeEach`/`afterEach` nos testes

Uma mudança sutil nos testes em relação ao projeto 06: de `before`/`after` (por suíte) para `beforeEach`/`afterEach` (por teste).

```typescript
// customers-mcp-z/tests/tools/customers.test.ts
describe('Customer Tools', async () => {
    let client: Client

    beforeEach(async () => {
        const serviceToken = await getServiceToken()  // obtém token real da API
        client = await createTestClient(serviceToken) // sobe novo MCP server por teste
    })

    afterEach(async () => {
        await client.close()  // derruba o processo após cada teste
    })
```

Por que criar cliente por teste e não por suíte? Porque o service token é ligado ao rate limit — se um teste esgota a cota do token, o próximo falha por motivo errado. Criar um token novo (e portanto um cliente com novo token) por teste isola os testes uns dos outros, evitando que o "should reach rate limit" interfira no "should create a customer".

O custo é subir um processo Node por teste — caro, mas necessário para isolamento correto.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Hierarquia de erros | [src/domain/errors.ts](../src/domain/errors.ts) | `UnauthorizedError`, `ForbiddenError`, `RateLimitError` |
| `#assertOk()` | [src/infrastructure/customer-http-client.ts](../src/infrastructure/customer-http-client.ts) | Mapeamento status → erro e sintaxe `#` |
| Propagação sem captura | [src/application/customer-service.ts](../src/application/customer-service.ts) | `CustomerService` não tem try/catch — deixa o erro subir |
| Captura na tool | [src/mcp/tools/list-customers.ts](../src/mcp/tools/list-customers.ts) | `catch (err)` → `structuredContent: { isError: true, message }` |
| Teste de token inválido | [tests/tools/customers.test.ts](../tests/tools/customers.test.ts) | `createTestClient('invalid-token-that-does-not-exist')` |
| `beforeEach`/`afterEach` | [tests/tools/customers.test.ts](../tests/tools/customers.test.ts) | Por que novo token/client por teste |
