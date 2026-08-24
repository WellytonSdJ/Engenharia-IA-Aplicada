# Arquitetura em Camadas no MCP

## O que é — e por que importa aqui

No projeto anterior (`05-mcps-do-zero`), tudo ficou em dois arquivos: `service.ts` com a lógica, `mcp.ts` com o registro. Funciona bem quando o domínio é pequeno. Quando o domínio cresce — um CRUD completo com 5 operações, validações, e uma fonte de dados HTTP externa — essa organização vira um arquivo de 200 linhas difícil de testar e modificar.

Este projeto introduz quatro camadas com responsabilidades separadas:

```
src/
  domain/         → tipos e validações (Zod schemas)
  infrastructure/ → acesso a sistemas externos (HTTP, banco, etc.)
  application/    → lógica de negócio (o que o sistema FAZ)
  mcp/            → protocolo MCP (como o sistema é EXPOSTO)
```

Cada camada só conhece as camadas abaixo dela. A camada `mcp` conhece `application`. A `application` conhece `infrastructure`. A `infrastructure` conhece `domain`. Nenhuma delas conhece o MCP SDK exceto a camada `mcp`.

---

## Camada 1: Domain

**Arquivo:** `src/domain/customer.ts`

Define os tipos e schemas do domínio — o que é um Customer, uma query, uma mutação. Todas as outras camadas importam daqui.

```typescript
// src/domain/customer.ts
export const CustomerSchema = z.object({
    _id: z.string().optional(),
    name: z.string(),
    phone: z.string(),
})
export type Customer = z.infer<typeof CustomerSchema>
```

O domínio não sabe que existe MCP, HTTP, nem MongoDB. Se amanhã mudar a fonte de dados de REST para GraphQL, ou de tools para resources, o domínio não muda.

---

## Camada 2: Infrastructure

**Arquivo:** `src/infrastructure/customerHttpClient.ts`

Faz as chamadas HTTP para a API legada. É a única camada que usa `fetch()`, conhece a URL, os métodos HTTP e os status codes.

```typescript
// src/infrastructure/customerHttpClient.ts
export class CustomerHttpClient {
    private baseUrl: string
    constructor(baseUrl: string) { this.baseUrl = baseUrl }

    async listCustomers(): Promise<Customer[]> {
        const res = await fetch(`${this.baseUrl}/customers`)
        return res.json() as Promise<Customer[]>
    }

    async getCustomerById(id: string): Promise<Customer | null> {
        const res = await fetch(`${this.baseUrl}/customers/${id}`)
        if (res.status === 404) return null   // trata 404 aqui, não lança exceção
        return res.json() as Promise<Customer>
    }
    // ...
}
```

Ponto importante: `getCustomerById` retorna `null` em vez de lançar exceção para 404. A decisão de "o que fazer quando o customer não existe" pertence à camada de aplicação, não à infraestrutura.

---

## Camada 3: Application

**Arquivo:** `src/application/customerService.ts`

Contém a lógica de negócio — o que o sistema faz além de simplesmente repassar requisições. O exemplo mais claro é `findCustomer`:

```typescript
// src/application/customerService.ts
async findCustomer(query: CustomerQuery): Promise<Customer | null> {
    if(query._id) return this.client.getCustomerById(query._id)  // atalho: tem ID, vai direto

    const customers = await this.client.listCustomers()           // sem ID: traz tudo
    return (
        customers.find(customer => {
            const entries = Object.entries(query) as [keyof Customer, string][]
            return entries.every(([key, value]) => {
                const customerValue = customer[key]
                return customerValue?.includes(value)             // busca parcial por substring
            })
        })
    ) ?? null
}
```

A API legada só tem `GET /customers/:id` (busca por ID). O `CustomerService` adiciona a lógica de busca por nome ou telefone sem precisar adicionar endpoint na API — ele puxa a lista completa e filtra localmente. Isso é decisão de negócio, não de infraestrutura.

O `CustomerService` não sabe que existe MCP. Ele poderia ser usado por uma API REST, um CLI, um job batch — não importa. A camada de aplicação é agnóstica ao protocolo de entrega.

---

## Camada 4: MCP

**Arquivos:** `src/mcp/server.ts`, `src/mcp/tools/*`, `src/mcp/resources/*`, `src/mcp/prompts/*`

Única camada que conhece o SDK MCP (`@modelcontextprotocol/sdk`). Registra tools, resources e prompts no `McpServer` e traduz os resultados do `CustomerService` para o formato do protocolo (`content`, `structuredContent`, `isError`).

```typescript
// src/mcp/tools/getCustomer.ts
server.registerTool("get_customer", {
    description: "Find a customer by _id, name, or phone number",
    inputSchema: CustomerQuerySchema,
    outputSchema: { customer: CustomerSchema.nullable().describe('...') }
}, async (query: CustomerQuery) => {
    const customer = await service.findCustomer(query)    // chama application, não infra
    return {
        content: [{ type: "text", text: JSON.stringify(customer) }],
        structuredContent: { customer }
    }
})
```

A tool chama `service.findCustomer(query)`. Ela não sabe que por trás existe um `fetch()`, nem que houve uma busca local com `Array.find()`. A camada MCP só sabe orquestrar a resposta no formato que o protocolo espera.

---

## Por que um arquivo por tool?

No `05-mcps-do-zero`, todas as tools ficavam em `mcp.ts`. Aqui cada tool tem seu próprio arquivo (`listCustomers.ts`, `createCustomer.ts`, etc.) — padrão de módulo por funcionalidade.

A vantagem não é óbvia com 5 tools, mas fica clara quando o projeto cresce: cada arquivo é testável de forma independente, as tools podem ser adicionadas/removidas sem alterar outras, e o `server.ts` fica como ponto de composição limpo:

```typescript
// src/mcp/server.ts
registerListCustomersTool(server, service)
registerGetCustomerTool(server, service)
registerCreateCustomersTool(server, service)
registerFindCustomerPrompt(server)
registerApiInfoResource(server, BASE_URL)
registerUpdateCustomersTool(server, service)
registerDeleteCustomersTool(server, service)
```

Lendo `server.ts`, você vê exatamente o que o servidor expõe — sem precisar vasculhar um arquivo de 200 linhas.

---

## Como as camadas se comunicam

```
src/index.ts
    ↓ importa
src/mcp/server.ts
    ↓ instancia CustomerService(BASE_URL)
    ↓ passa server + service para cada register*()
src/mcp/tools/getCustomer.ts
    ↓ chama service.findCustomer(query)
src/application/customerService.ts
    ↓ chama client.getCustomerById(id) OU client.listCustomers()
src/infrastructure/customerHttpClient.ts
    ↓ fetch("http://localhost:9999/v1/customers/...")
API legada (nodejs-fastify-mongodb-crud)
    ↓ MongoDB
```

Cada seta é uma dependência unidirecional — nenhuma camada conhece quem está acima dela.

---

## Referências no projeto

| Camada | Arquivo | O que observar |
| --- | --- | --- |
| Domain | [src/domain/customer.ts](../src/domain/customer.ts) | Schemas compartilhados entre todas as camadas |
| Infrastructure | [src/infrastructure/customerHttpClient.ts](../src/infrastructure/customerHttpClient.ts) | `fetch()` encapsulado; retorno `null` para 404 |
| Application | [src/application/customerService.ts](../src/application/customerService.ts) | `findCustomer` com busca flexível além da API |
| MCP (composição) | [src/mcp/server.ts](../src/mcp/server.ts) | Ponto de entrada com todas as registrações |
| MCP (tool) | [src/mcp/tools/getCustomer.ts](../src/mcp/tools/getCustomer.ts) | Tool que chama service sem saber de HTTP |
