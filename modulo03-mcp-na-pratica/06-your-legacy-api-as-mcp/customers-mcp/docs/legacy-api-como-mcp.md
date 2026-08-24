# Transformando uma API Legada em Servidor MCP

## O que é

"API legada como MCP" é o padrão de criar uma camada de adaptação MCP em cima de uma API REST que já existe — sem modificar o código original. O MCP server atua como um tradutor: recebe chamadas de tools do agente e as converte em requisições HTTP para a API original.

A motivação é real no mundo corporativo: você tem uma API REST funcionando em produção, com testes, clientes e histórico de mudanças. Reescrever ela pra "falar MCP nativamente" seria arriscado e desnecessário. Em vez disso, você escreve um servidor MCP fino que embrulha a API existente.

```
Antes (sem MCP):          Depois (com MCP layer):
LLM → código hardcoded    Agente → MCP server → API REST → Banco
      ↓                                  ↑
     API REST                    (sem tocar aqui)
      ↓
     Banco
```

---

## A API legada: o que ela faz e o que NÃO muda

`nodejs-fastify-mongodb-crud` é um CRUD de customers com Fastify e MongoDB. Cinco endpoints:

```
GET    /v1/customers          → lista todos (ordenado por nome)
GET    /v1/customers/:id      → busca por ID
POST   /v1/customers          → cria { name, phone }
PUT    /v1/customers/:id      → atualiza { name, phone }
DELETE /v1/customers/:id      → deleta por ID
```

Nada disso muda quando adicionamos o MCP server. A API continua rodando em `:9999` exatamente como antes.

O que o Fastify faz que não é óbvio: cada rota declara um `schema: { body: {...}, response: {...} }` em JSON Schema. O Fastify usa isso para validar a request e serializar a response muito mais rápido que frameworks que fazem `JSON.stringify` genérico. Esse é o motivo de Fastify ser usado em vez de Express em aplicações de alta performance.

---

## O que o MCP server embrulha

O `customers-mcp` expõe cinco tools, um resource e um prompt:

| Primitiva MCP | Nome | Mapeia para |
| --- | --- | --- |
| Tool | `list_customers` | `GET /v1/customers` |
| Tool | `get_customer` | `GET /v1/customers/:id` + busca local por nome/telefone |
| Tool | `create_customer` | `POST /v1/customers` |
| Tool | `update_customer` | `PUT /v1/customers/:id` |
| Tool | `delete_customer` | `DELETE /v1/customers/:id` |
| Resource | `customers://api-info` | documentação da própria API (não tem endpoint equivalente) |
| Prompt | `find_customer_prompt` | instrução pré-montada para acionar `get_customer` |

A relação é quase um-para-um com exceção do `get_customer`: a API legada só busca por ID (`/v1/customers/:id`), mas o `get_customer` aceita também nome e telefone. A inteligência de busca por campos opcionais fica na camada de aplicação (`CustomerService.findCustomer`) — a API legada não precisa saber disso.

---

## Como isso se parece no código

`customers-mcp/src/mcp/tools/listCustomers.ts`:

```typescript
server.registerTool(
    "list_customers",
    {
        description: "List all customers",
        inputSchema: {},       // sem argumentos — lista tudo
        outputSchema: {
            customers: z.array(CustomerSchema).describe('Array of all customers')
        }
    },
    async () => {
        const customers = await service.listCustomers()   // → CustomerService
        return {
            content: [{ type: "text", text: JSON.stringify(customers, null, 2) }],
            structuredContent: { customers }
        }
    }
)
```

O handler faz exatamente uma coisa: chama `service.listCustomers()` e formata o resultado. A tool não sabe que existe uma requisição HTTP por trás. Quem sabe é o `CustomerHttpClient` — e só ele.

---

## Por que não criar as tools chamando `fetch()` direto?

Tecnicamente funcionaria. Mas misturar a lógica de protocolo HTTP dentro do handler da tool cria um problema: se a URL da API mudar, você tem que alterar cada tool individualmente. Se quiser trocar a fonte de dados (de REST para GraphQL, ou para chamada direta ao banco), a mudança se espalharia por todos os handlers.

A separação em camadas resolve isso. A tool chama o service. O service chama o client HTTP. Se a URL muda, só `server.ts` (onde `BASE_URL` é definida) e `customerHttpClient.ts` precisam mudar:

```typescript
// src/mcp/server.ts — única fonte de verdade da URL
const BASE_URL = "http://localhost:9999/v1";
const service = new CustomerService(BASE_URL)
```

---

## O resource como documentação viva

O resource `customers://api-info` não tem equivalente na API legada — é algo novo que o MCP server adiciona. Ele serve como documentação consultável em runtime:

```typescript
// src/mcp/resources/apiInfo.ts
server.registerResource("customers://api-info", "customers://api-info", { ... }, () => ({
    contents: [{
        uri: "customers://api-info",
        mimeType: "text/plain",
        text: `
Customers API
  Base URL : ${baseUrl}
  Endpoints:
    GET    /customers     — list all customers
    ...
  Customer shape: { _id: string, name: string, phone: string }
`
    }]
}))
```

Um agente pode consultar esse resource antes de decidir qual tool chamar — é a forma de o MCP substituir "leia o README antes de usar as tools" por algo que o agente consegue fazer programaticamente.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| API legada (Fastify) | [nodejs-fastify-mongodb-crud/src/index.js](../../nodejs-fastify-mongodb-crud/src/index.js) | Rotas CRUD com `schema:` JSON Schema em cada uma |
| Mapeamento tool → HTTP | [src/infrastructure/customerHttpClient.ts](../src/infrastructure/customerHttpClient.ts) | `fetch()` encapsulado — única camada que conhece HTTP |
| Registro de todas as tools | [src/mcp/server.ts](../src/mcp/server.ts) | `registerListCustomersTool`, `registerGetCustomerTool`, etc. |
| Resource de documentação | [src/mcp/resources/apiInfo.ts](../src/mcp/resources/apiInfo.ts) | `customers://api-info` como documentação da API legada |
| Busca flexível além da API | [src/application/customerService.ts](../src/application/customerService.ts) | `findCustomer` com busca por nome/telefone que a API não tem |
