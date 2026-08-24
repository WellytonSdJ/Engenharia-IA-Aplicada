# Por onde começar

Este é o sétimo projeto do módulo 3. O projeto anterior (`06-your-legacy-api-as-mcp`) construiu um MCP server que embrulha uma API legada aberta — sem autenticação, sem controle de acesso, sem limite de chamadas. Este projeto resolve isso.

---

## O que estamos construindo e por quê

> A mesma API de customers, agora protegida. O MCP server precisa se autenticar antes de usar qualquer tool — e o role do service token determina o que ele pode fazer.

```
Projeto 06:                              Projeto 07:
API aberta → qualquer um pode acessar.   API protegida com JWT + Service Tokens.
MCP sem credenciais → chama direto.      MCP autentica com SERVICE_TOKEN para chamar a API.
Sem limite de chamadas.                  Rate limiting: 90 req/min por token.
Erros HTTP genéricos.                    Erros tipados: Unauthorized, Forbidden, RateLimitError.
```

O foco não é segurança de produção (as credenciais são hardcoded no código de estudo). O foco é entender os **padrões e mecanismos**: JWT, service tokens, RBAC, rate limiting — e como esses conceitos se encaixam tanto na API REST quanto no servidor MCP.

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [jwt-e-service-tokens.md](./jwt-e-service-tokens.md) | Os dois mecanismos de auth do projeto: JWT para usuários humanos, Service Token para o MCP server. Entender a diferença é o núcleo do projeto. |
| 2 | [rbac-roles-e-permissoes.md](./rbac-roles-e-permissoes.md) | Como os roles `admin`/`member` são embutidos no token e verificados por rota com `requireRole()` |
| 3 | [rate-limiting.md](./rate-limiting.md) | Como `@fastify/rate-limit` limita por token e como os testes verificam que o limite funciona |
| 4 | [erros-de-dominio-e-tratamento.md](./erros-de-dominio-e-tratamento.md) | Como erros HTTP (401, 403, 429) viram erros de domínio tipados e chegam ao LLM como `isError: true` |
| 5 | [glossario.md](./glossario.md) | Referência rápida dos termos novos |

---

## Mapa do código

```
nodejs-fastify-mongodb-crud-z/           ← API legada com segurança adicionada
  src/config.js                          → REQUESTS_PER_MINUTE exportado (usado em auth.js)
  src/db.js                              → conexão MongoDB (igual ao projeto 06)
  src/auth.js                            → JWT, service tokens, RBAC, rateLimitOptions — tudo de auth aqui
  src/index.js                           → Fastify com @fastify/jwt + @fastify/rate-limit registrados;
                                           preHandler: [requireRole('admin')] nas rotas de escrita

customers-mcp-z/                         ← MCP server com autenticação
  src/domain/customer.ts                 → schemas Zod (igual ao projeto 06)
  src/domain/errors.ts                   → UnauthorizedError, ForbiddenError, RateLimitError
  src/infrastructure/customer-http-client.ts → authHeaders no construtor; #assertOk() privado
  src/application/customer-service.ts   → recebe (baseUrl, serviceToken) → repassa ao client
  src/mcp/server.ts                      → SERVICE_TOKEN lido de process.env; service instanciado com token
  src/mcp/tools/                         → tools (mesma estrutura do 06, erros propagados via catch)
  src/mcp/resources/api-info.ts          → resource de documentação (igual ao 06)
  src/mcp/prompts/findCustomer.ts        → prompt de busca (igual ao 06)
  src/index.ts                           → valida SERVICE_TOKEN antes de conectar o transport
  tests/helpers.ts                       → getServiceToken() + createTestClient(serviceToken)
  tests/tools/customers.test.ts          → testa CRUD + token inválido + rate limit
  getServiceToken.sh                     → script para obter service tokens via curl
```

---

## O fluxo em uma linha

```
Agente → STDIO → MCP server (SERVICE_TOKEN no env)
  → tool → CustomerService(serviceToken) → CustomerHttpClient(authHeaders)
    → fetch() com Authorization: Bearer <token>
      → API Fastify → onRequest hook → jwtVerify() OU issuedServiceTokens.get(token)
        → requireRole('admin') (se rota de escrita) → MongoDB
```

---

## Como rodar

```bash
# 1. Subir o MongoDB e a API com segurança
cd nodejs-fastify-mongodb-crud-z
docker compose up -d mongodb
npm install
npm start

# 2. Obter um service token para testar manualmente
bash ../customers-mcp-z/getServiceToken.sh
# copia o "Admin Service Token" da saída

# 3. Rodar os testes do MCP server (obtém o token automaticamente)
cd ../customers-mcp-z
npm install
npm test
```

> Os testes do `customers-mcp-z` chamam `getServiceToken()` automaticamente antes de cada teste. A API legada deve estar rodando em `:9999`.
