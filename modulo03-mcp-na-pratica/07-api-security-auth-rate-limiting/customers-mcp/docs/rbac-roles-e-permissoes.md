# RBAC: Roles e Permissões

## O que é RBAC

RBAC (Role-Based Access Control) é o padrão de controlar o que cada usuário pode fazer baseado no **papel** dele no sistema, não no usuário em si. Em vez de listar "ananeri pode fazer X", você define "quem tem role `member` pode fazer X" e "quem tem role `admin` pode fazer Y". Adicionar um novo usuário com role `member` já herda todas as restrições do role, sem configuração individual.

---

## Os dois roles deste projeto

```javascript
// nodejs-fastify-mongodb-crud-z/src/auth.js
export const authUsers = [{
    username: 'erickwendel',
    password: '123123',
    role: 'admin',   // pode ler e escrever
},
{
    username: 'ananeri',
    password: '1234',
    role: 'member'   // pode só ler (GET)
}]
```

| Role | Permissões |
| --- | --- |
| `admin` | GET, POST, PUT, DELETE |
| `member` | GET apenas (leitura) |

---

## Como o role viaja no token

O role é embutido no payload do JWT no momento do login:

```javascript
// nodejs-fastify-mongodb-crud-z/src/auth.js
const token = fastify.jwt.sign({ username, role: user.role })
```

Isso significa que o servidor não precisa consultar o banco de dados para saber o role a cada request — ele está codificado no próprio token (que é verificado criptograficamente). O mesmo acontece com service tokens: o Map armazena `{ username, role }` e o `onRequest` hook injeta isso em `request.user`.

Depois da verificação, `request.user` tem a forma `{ username: string, role: string }` — disponível para qualquer hook ou handler subsequente.

---

## `requireRole()`: middleware de RBAC

```javascript
// nodejs-fastify-mongodb-crud-z/src/auth.js
export function requireRole(role) {
    // Retorna uma FUNÇÃO, não executa imediatamente.
    // preHandler espera um handler — requireRole('admin') resolve para esse handler.
    return async function (request, reply) {
        if (request.user.role === role) return   // role certo: passa

        return reply.code(403).send({
            message: 'Forbidden: insufficient permissions'
        })
    }
}
```

`requireRole` é uma **factory de middleware**: chama ela com o role necessário e recebe uma função pronta para usar como `preHandler`. Isso é o padrão de closure para configurar comportamento em tempo de declaração de rota.

---

## Aplicação por rota

```javascript
// nodejs-fastify-mongodb-crud-z/src/index.js
// GET /v1/customers — sem preHandler: qualquer token autenticado pode ler
fastify.get('/v1/customers', async (request, reply) => { ... })

// POST /v1/customers — preHandler: [requireRole('admin')] — só admin pode criar
fastify.post('/v1/customers', {
    preHandler: [requireRole('admin')],
    schema: { ... }
}, async (request, reply) => { ... })

// PUT e DELETE seguem o mesmo padrão
```

`preHandler` é um array de funções executadas antes do handler principal — sequencialmente, com curto-circuito: se `requireRole('admin')` responder com 403, o handler principal nunca é chamado.

---

## O que acontece quando um member tenta criar

```
1. POST /v1/customers com token de ananeri (role: member)
2. onRequest → issuedServiceTokens.get(token) → { username: 'ananeri', role: 'member' }
3. request.user = { username: 'ananeri', role: 'member' }
4. preHandler: requireRole('admin')
5. request.user.role === 'member' ≠ 'admin'
6. reply.code(403).send({ message: 'Forbidden: insufficient permissions' })
7. Handler principal nunca executa
```

No MCP server, a tool `create_customer` lança `ForbiddenError` (via `#assertOk`) e o handler devolve `isError: true` com a mensagem do erro para o LLM.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Roles no cadastro de usuários | [nodejs-fastify-mongodb-crud-z/src/auth.js](../../nodejs-fastify-mongodb-crud-z/src/auth.js) | `authUsers` com role por usuário |
| Role embutido no JWT | [nodejs-fastify-mongodb-crud-z/src/auth.js](../../nodejs-fastify-mongodb-crud-z/src/auth.js) | `fastify.jwt.sign({ username, role: user.role })` |
| `requireRole()` middleware | [nodejs-fastify-mongodb-crud-z/src/auth.js](../../nodejs-fastify-mongodb-crud-z/src/auth.js) | Closure retornando handler |
| Aplicação por rota | [nodejs-fastify-mongodb-crud-z/src/index.js](../../nodejs-fastify-mongodb-crud-z/src/index.js) | `preHandler: [requireRole('admin')]` em POST, PUT, DELETE |
| ForbiddenError no MCP | [src/domain/errors.ts](../src/domain/errors.ts) | Erro de domínio mapeado para HTTP 403 |
