# Glossário — 06-your-legacy-api-as-mcp

Termos já cobertos em [`05-mcps-do-zero/docs/glossario.md`](../../../05-mcps-do-zero/docs/glossario.md) (McpServer, registerTool, registerResource, registerPrompt, StdioServerTransport, Client, structuredContent, isError, etc.) não são repetidos aqui — este glossário cobre só o que é novo neste projeto.

---

## Arquitetura

| Termo | Definição |
| --- | --- |
| **API Legada** | Uma API REST que já existe e funciona. Neste projeto: `nodejs-fastify-mongodb-crud` (Fastify + MongoDB, rodando em `:9999`). O ponto é que ela não é modificada para suportar MCP — o MCP server é construído em cima dela. |
| **Camada de domínio** | `src/domain/` — contém apenas os schemas Zod e tipos TypeScript do negócio. Não sabe de HTTP, MCP, nem banco de dados. |
| **Camada de infraestrutura** | `src/infrastructure/` — acessa sistemas externos (aqui, a API legada via `fetch()`). Única camada que conhece URLs, métodos HTTP e status codes. |
| **Camada de aplicação** | `src/application/` — lógica de negócio que não é só "repassar chamadas". Aqui: `findCustomer` com busca por substring que a API legada não tem. |
| **Camada MCP** | `src/mcp/` — única camada que importa o SDK `@modelcontextprotocol/sdk`. Registra tools, resources e prompts no `McpServer`. |
| **Padrão um-arquivo-por-tool** | Cada tool tem seu próprio arquivo (`listCustomers.ts`, `createCustomer.ts`, etc.) em vez de todos ficarem em `mcp.ts`. Facilita adicionar/remover tools sem afetar outras. |

---

## Zod

| Termo | Definição |
| --- | --- |
| **`z.infer<typeof Schema>`** | Extrai o tipo TypeScript de um schema Zod automaticamente — evita declarar o type manualmente e mantê-lo sincronizado com o schema. |
| **`.extend()`** | Cria um novo schema herdando todos os campos de outro e opcionalmente sobrescrevendo alguns. Usado em `CustomerUpdateSchema.extend({ _id: z.string() })` para tornar `_id` obrigatório. |
| **`.shape`** | Propriedade de um `z.object()` que expõe o mapa interno de campo→schema. Necessário porque `registerTool` espera esse mapa, não um `z.object()` completo. |
| **`.nullable()`** | Permite que um campo aceite `null` além do tipo normal. Usado em `CustomerSchema.nullable()` para indicar que `get_customer` pode retornar `null` quando o customer não existe. |
| **`CustomerMutationSchema`** | Schema unificado para respostas de mutações (create, update, delete, get, list). Todos os campos opcionais para que diferentes tools usem o mesmo schema sem rejeição por "campos extras" no `structuredContent`. |

---

## API Legada (Fastify + MongoDB)

| Termo | Definição |
| --- | --- |
| **Fastify** | Framework HTTP Node.js focado em performance. Diferencial: validação e serialização via JSON Schema compilado — mais rápido que `JSON.stringify` genérico do Express. |
| **`schema:` por rota** | Fastify permite declarar `body`, `response` em JSON Schema para cada rota. O framework valida a request e serializa a response automaticamente. |
| **`ObjectId.isValid()`** | Método do driver MongoDB que verifica se uma string é um ObjectId válido (24 hex chars). Previne erro de cast antes de consultar o banco. |
| **`ObjectId.createFromHexString()`** | Converte uma string hex em `ObjectId` do MongoDB — necessário porque `findOne({ _id: id })` com string não funciona; o MongoDB armazena IDs como objetos `ObjectId`. |
| **`onClose` hook** | Hook do Fastify executado quando o servidor para. Usado para fechar a conexão com o MongoDB (`dbClient.close()`). |
| **`preHandler` hook** | Hook do Fastify executado antes de qualquer handler de rota. Usado aqui para injetar headers CORS e tratar preflight OPTIONS. |
| **`$set`** | Operador do MongoDB para atualização parcial: `{ $set: { name: "novo" } }` atualiza só o campo `name`, sem apagar os outros. |

---

## Testes

| Termo | Definição |
| --- | --- |
| **`Omit<Customer, '_id'>`** | Tipo TypeScript que exclui um campo de um tipo existente. Usado em `createCustomer(customer: Omit<Customer, '_id'>)` — ao criar, o `_id` ainda não existe. |
| **`structuredContent`** | Ver glossário do `05-mcps-do-zero`. Neste projeto, os testes acessam `result.structuredContent.customers`, `.customer`, `.id`, `.message` — a tipagem é feita com interfaces locais nos testes (ex: `type CustomersResult = { structuredContent: { customers: Customer[] } }`). |
