# Documentação — 07-api-security-auth-rate-limiting

Este projeto adiciona **segurança e governança** ao sistema do projeto anterior (06): autenticação JWT, Service Tokens para acesso máquina-a-máquina, RBAC (controle de acesso por papel/role) e rate limiting. A API legada continua sendo a mesma — o que muda é que agora ela exige autenticação, e o MCP server precisa se autenticar automaticamente.

Os sub-projetos têm sufixo `-z` (`nodejs-fastify-mongodb-crud-z`, `customers-mcp-z`) para diferenciá-los das versões sem segurança do projeto 06.

---

## O que mudou em relação ao projeto 06

| Componente | Projeto 06 | Projeto 07 |
| --- | --- | --- |
| API REST | Aberta (sem auth) | Protegida com JWT + Service Token |
| RBAC | Não tinha | Admin: CRUD completo. Member: só leitura |
| Rate limiting | Não tinha | 90 req/min por token |
| MCP server | Sem credenciais | Recebe `SERVICE_TOKEN` via env var |
| Erros HTTP | Exceção genérica | Hierarquia: `UnauthorizedError`, `ForbiddenError`, `RateLimitError` |
| Testes | `before`/`after` por suíte | `beforeEach`/`afterEach` + `getServiceToken()` por teste |

---

## Tecnologias novas neste projeto

| Tecnologia | Papel |
| --- | --- |
| `@fastify/jwt` | Plugin JWT para Fastify — `fastify.jwt.sign()` e `request.jwtVerify()` |
| `@fastify/rate-limit` | Rate limiting por token (90 req/min, configurável) |
| Service Token (UUID) | Auth M2M sem JWT — opaco, armazenado em memória no servidor |
| RBAC | `requireRole('admin')` como `preHandler` nas rotas de escrita |

---

## Documentos

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Por onde começar: contexto, trilha, mapa do código e comandos |
| [jwt-e-service-tokens.md](./jwt-e-service-tokens.md) | Dois tipos de autenticação: JWT (usuário humano) vs Service Token (máquina) — por que os dois, como cada um funciona |
| [rbac-roles-e-permissoes.md](./rbac-roles-e-permissoes.md) | Role-Based Access Control: roles admin/member, `requireRole()` middleware, como o role viaja no token |
| [rate-limiting.md](./rate-limiting.md) | Rate limiting com `@fastify/rate-limit`: `keyGenerator` por token, janela de 1 minuto, teste de esgotamento |
| [erros-de-dominio-e-tratamento.md](./erros-de-dominio-e-tratamento.md) | Hierarquia de erros de domínio (`UnauthorizedError`, `ForbiddenError`, `RateLimitError`), `#assertOk()` e propagação até `isError: true` no MCP |
| [glossario.md](./glossario.md) | Termos novos — os cobertos em 05 e 06 não são repetidos |
