# JWT e Service Tokens

## O que são — e por que os dois

Este projeto tem **dois mecanismos de autenticação** servindo a propósitos diferentes:

| Mecanismo | Para quem | Como funciona | Onde é verificado |
| --- | --- | --- | --- |
| JWT | Usuário humano (browser, Postman) | Login com senha → recebe token assinado com `JWT_SECRET` | `request.jwtVerify()` decodifica e valida a assinatura |
| Service Token | Máquina (o `customers-mcp-z`) | Credenciais + `adminSuperSecret` → recebe UUID opaco | `issuedServiceTokens.get(token)` no servidor |

Um usuário humano precisa de um token com expiração, que ele renova fazendo login. Uma máquina (como o MCP server) não tem ninguém digitando senha — ela precisa de uma credencial de longa duração, obtida uma vez (via automação) e reutilizada.

---

## JWT: como funciona neste projeto

O JWT é gerado no endpoint `POST /v1/auth/login`:

```javascript
// nodejs-fastify-mongodb-crud-z/src/auth.js
fastify.post('/v1/auth/login', ..., async (request, reply) => {
    const { username, password } = request.body
    const user = authUsers.find(u => u.username === username && u.password === password)

    if (!user) return reply.code(401).send({ message: 'Invalid credentials' })

    // jwt.sign() do @fastify/jwt: assina o payload com JWT_SECRET
    // O role fica dentro do token — sem consulta extra ao banco por request
    const token = fastify.jwt.sign({ username, role: user.role })
    return reply.send({ token })
})
```

Para verificar rotas protegidas, o hook `onRequest` chama `request.jwtVerify()`:

```javascript
// nodejs-fastify-mongodb-crud-z/src/auth.js
fastify.addHook('onRequest', async (request, reply) => {
    const publicRoutes = ['/v1/health', '/v1/auth/login', '/v1/auth/service-token']
    if (publicRoutes.includes(request.originalUrl)) return

    try {
        await request.jwtVerify()      // valida assinatura + decodifica → request.user = { username, role }
    } catch (error) {
        return reply.code(401).send({ message: 'Unauthorized' })
    }
})
```

`request.jwtVerify()` (método injetado pelo `@fastify/jwt`) faz três coisas de uma vez: extrai o token do header `Authorization: Bearer`, verifica a assinatura com `JWT_SECRET`, e injeta o payload decodificado em `request.user`. Se qualquer etapa falhar, lança exceção.

---

## Service Token: por que é diferente

O JWT é **stateless**: o servidor não armazena nada — qualquer servidor com o mesmo `JWT_SECRET` pode verificar qualquer JWT. O Service Token é **stateful**: o servidor armazena um Map de `token → user`, e o token só é válido enquanto o Map tiver aquela entrada.

```javascript
// nodejs-fastify-mongodb-crud-z/src/auth.js
const issuedServiceTokens = new Map()   // token → { username, role }

fastify.post('/v1/auth/service-token', ..., async (request, reply) => {
    const { username, password, adminSuperSecret } = request.body

    // Dupla verificação: credenciais E o segredo de admin
    if (adminSuperSecret !== ADMIN_SUPER_SECRET) {
        return reply.code(401).send({ message: 'Invalid adminSuperSecret' })
    }
    // ... valida user ...

    const serviceToken = randomUUID()      // UUID opaco — não decodificável
    issuedServiceTokens.set(serviceToken, { username: user.username, role: user.role })
    return reply.send({ serviceToken, role: user.role })
})
```

No `onRequest` hook, a verificação checa o Map antes de tentar `jwtVerify()`:

```javascript
// nodejs-fastify-mongodb-crud-z/src/auth.js
const serviceUser = issuedServiceTokens.get(token)
if (serviceUser) {
    request.user = serviceUser    // injeta manualmente, igual ao jwtVerify() faria
    return                        // interrompe o hook — não precisa verificar JWT
}

try {
    await request.jwtVerify()     // só chega aqui se NÃO for service token
} catch (error) { ... }
```

---

## Como o MCP server usa o Service Token

O `customers-mcp-z` obtém o service token antes de iniciar — e passa para todas as tools via injeção de dependência:

```typescript
// customers-mcp-z/src/index.ts
const SERVICE_TOKEN = process.env.SERVICE_TOKEN ?? "";
if (!SERVICE_TOKEN) {
    console.error('[error]: SERVICE_TOKEN env var is required');
    process.exit(1);
}

// customers-mcp-z/src/mcp/server.ts
const SERVICE_TOKEN = process.env.SERVICE_TOKEN!
const service = new CustomerService(BASE_URL, SERVICE_TOKEN);

// customers-mcp-z/src/infrastructure/customer-http-client.ts
constructor(baseUrl: string, serviceToken: string) {
    this.baseUrl = baseUrl;
    this.authHeaders = { Authorization: `Bearer ${serviceToken}` };
}
```

Nos testes, o service token é obtido da API real antes de subir o MCP server:

```typescript
// customers-mcp-z/tests/helpers.ts
export async function getServiceToken(): Promise<string> {
    const res = await fetch(`${API_URL}/auth/service-token`, {
        method: 'POST',
        body: JSON.stringify({ username: 'erickwendel', password: '123123', adminSuperSecret: 'AM I THE BOSS?' }),
    })
    const { serviceToken } = await res.json()
    return serviceToken
}

export async function createTestClient(serviceToken: string) {
    const transport = new StdioClientTransport({
        command: 'node',
        args: ['--experimental-strip-types', 'src/index.ts'],
        env: { ...process.env, SERVICE_TOKEN: serviceToken },   // injetado no processo filho
    })
    // ...
}
```

`...process.env` espalha o ambiente atual do processo de teste, e `SERVICE_TOKEN: serviceToken` sobrescreve (ou adiciona) a variável — o MCP server filho lê normalmente via `process.env.SERVICE_TOKEN`.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| JWT sign/verify | [nodejs-fastify-mongodb-crud-z/src/auth.js](../../nodejs-fastify-mongodb-crud-z/src/auth.js) | `fastify.jwt.sign()`, `request.jwtVerify()` |
| Service token emitido | [nodejs-fastify-mongodb-crud-z/src/auth.js](../../nodejs-fastify-mongodb-crud-z/src/auth.js) | `randomUUID()` + `issuedServiceTokens.set()` |
| onRequest hook com dupla verificação | [nodejs-fastify-mongodb-crud-z/src/auth.js](../../nodejs-fastify-mongodb-crud-z/src/auth.js) | `issuedServiceTokens.get(token)` antes do `jwtVerify()` |
| SERVICE_TOKEN via env | [src/index.ts](../src/index.ts) | Validação antes de conectar o transport |
| Token injetado no processo filho | [tests/helpers.ts](../tests/helpers.ts) | `env: { ...process.env, SERVICE_TOKEN: serviceToken }` |
| Script para obter tokens manualmente | [getServiceToken.sh](../getServiceToken.sh) | `curl` para `/v1/auth/service-token` com jq |
