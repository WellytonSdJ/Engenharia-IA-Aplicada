# Glossário — 07-api-security-auth-rate-limiting

Termos cobertos em [`05-mcps-do-zero/docs/glossario.md`](../../../05-mcps-do-zero/docs/glossario.md) e [`06-your-legacy-api-as-mcp/customers-mcp/docs/glossario.md`](../../../06-your-legacy-api-as-mcp/customers-mcp/docs/glossario.md) não são repetidos aqui — este glossário cobre só o que é novo neste projeto.

---

## Autenticação e Autorização

| Termo | Definição |
| --- | --- |
| **JWT (JSON Web Token)** | Token assinado digitalmente que carrega claims (dados) no próprio corpo. Verificado sem consultar banco — qualquer parte com o `JWT_SECRET` pode validar. Formato: `header.payload.signature` em base64. |
| **`@fastify/jwt`** | Plugin Fastify que adiciona `fastify.jwt.sign()` (para emitir tokens) e `request.jwtVerify()` (para verificar e decodificar), injetando o payload em `request.user`. |
| **Service Token** | UUID opaco emitido pelo servidor para autenticar aplicações (M2M — machine-to-machine). Diferente do JWT: não decodificável sem o Map do servidor; sem expiração automática; requer `adminSuperSecret` para emissão. |
| **M2M (Machine-to-Machine)** | Autenticação entre sistemas sem interação humana. O `customers-mcp-z` é o "cliente máquina" — ele não faz login interativo, ele usa um service token. |
| **`request.jwtVerify()`** | Método injetado pelo `@fastify/jwt`: extrai o Bearer token, verifica a assinatura com `JWT_SECRET`, injeta payload em `request.user`. Lança exceção se o token for inválido ou expirado. |
| **`issuedServiceTokens`** | `Map<string, {username, role}>` em memória que armazena os service tokens emitidos. Lookup O(1) por token — verificação é só `issuedServiceTokens.get(token)`. |

---

## RBAC

| Termo | Definição |
| --- | --- |
| **RBAC** | Role-Based Access Control — controle de acesso baseado em papéis. Em vez de permissões por usuário, define o que cada papel (role) pode fazer. |
| **Role** | Papel atribuído a um usuário: `admin` (leitura + escrita) ou `member` (leitura apenas) neste projeto. |
| **`requireRole(role)`** | Factory de middleware: recebe o role necessário e retorna um `preHandler` que verifica `request.user.role`. Retorna 403 se o role não bater. |
| **`preHandler`** | Array de funções do Fastify executadas antes do handler principal de uma rota. Se qualquer um responder (ex: 403), o handler principal não é chamado. |
| **HTTP 403 Forbidden** | Requisição autenticada (token válido), mas sem permissão para a operação solicitada — diferente do 401 (não autenticado). |

---

## Rate Limiting

| Termo | Definição |
| --- | --- |
| **Rate Limiting** | Mecanismo que limita o número de requisições que um cliente pode fazer em um período de tempo. Rejeita com HTTP 429 quando excedido. |
| **`@fastify/rate-limit`** | Plugin Fastify que adiciona rate limiting global. Configurado com `max`, `timeWindow` e `keyGenerator`. |
| **`keyGenerator`** | Função que identifica o cliente para o rate limiter. Aqui: extrai o token do header de autorização (limitando por token) com fallback para IP. |
| **HTTP 429 Too Many Requests** | Resposta quando o rate limit é excedido. O cliente deve aguardar o reset da janela antes de tentar novamente. |
| **Janela de tempo (`timeWindow`)** | Período em que as requisições são contadas. Após a janela expirar, o contador zera. Aqui: 1 minuto. |

---

## Erros e Infraestrutura

| Termo | Definição |
| --- | --- |
| **`UnauthorizedError`** | Erro de domínio lançado quando a API retorna HTTP 401 — token inválido ou ausente. |
| **`ForbiddenError`** | Erro de domínio lançado quando a API retorna HTTP 403 — token válido mas role insuficiente para a operação. |
| **`RateLimitError`** | Erro de domínio lançado quando a API retorna HTTP 429 — limite de requisições atingido. |
| **`#assertOk()`** | Método privado JavaScript nativo (`#` prefix) do `CustomerHttpClient`. Centraliza o mapeamento de status HTTP para erros de domínio. Diferente do `private` do TypeScript (só compilação), `#` é aplicado em runtime. |
| **Private field (`#`)** | Sintaxe de campo/método privado nativo do JavaScript. Métodos com `#` não são acessíveis fora da classe, mesmo em subclasses ou via `Object.getOwnPropertyNames`. |
| **`beforeEach`/`afterEach`** | Hooks do Node.js Test Runner executados antes/depois de cada `it`. Diferente de `before`/`after` (executados uma vez por suíte). Aqui necessário para isolar service tokens entre testes. |
| **Env var no processo filho** | `StdioClientTransport` aceita `env` para injetar variáveis de ambiente no processo filho (o MCP server). `{ ...process.env, SERVICE_TOKEN: token }` replica o ambiente atual e adiciona/sobrescreve a variável específica. |
