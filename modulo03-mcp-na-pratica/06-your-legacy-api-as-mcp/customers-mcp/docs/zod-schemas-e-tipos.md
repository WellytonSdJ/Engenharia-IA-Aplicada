# Zod: Schemas e Tipos TypeScript

## O que é — e por que está em todo o projeto

Zod é uma biblioteca de validação e inferência de tipos que aparece em três camadas do projeto: no domínio (definindo os schemas), nas tools (como `inputSchema`/`outputSchema`), e nos testes (tipando os resultados). O projeto anterior (`05-mcps-do-zero`) já usou Zod nos schemas de tools — aqui, Zod também vira a fonte da tipagem TypeScript do domínio.

> Conceitos básicos de Zod (campo→schema, `z.string()`, `z.object()`, `.describe()`) já cobertos em [`05-mcps-do-zero/docs/construindo-mcp-server-do-zero.md`](../../../05-mcps-do-zero/docs/construindo-mcp-server-do-zero.md). Este doc cobre só o que é novo.

---

## `z.infer<typeof Schema>` — tipo derivado do schema

Em vez de declarar o schema Zod e depois um type TypeScript separado (com risco de ficarem dessincronizados), `z.infer` extrai o tipo diretamente do schema:

```typescript
// src/domain/customer.ts
export const CustomerSchema = z.object({
    _id: z.string().optional(),
    name: z.string(),
    phone: z.string(),
})

// Customer é inferido como: { _id?: string; name: string; phone: string }
export type Customer = z.infer<typeof CustomerSchema>
```

Se você adicionar um campo ao schema, o tipo TypeScript muda automaticamente — sem precisar lembrar de atualizar os dois lugares. Qualquer desalinhamento entre o que o Zod valida e o que o TypeScript tipifica se torna impossível por construção.

---

## `.extend()` — herdar e sobrescrever campos

`CustomerUpdateSchema` precisa de quase os mesmos campos que `CustomerQuerySchema`, mas com `_id` obrigatório (não opcional):

```typescript
// src/domain/customer.ts
export const CustomerQuerySchema = z.object({
    _id: z.string().optional().describe("MongoDB ObjectId of the customer"),
    name: z.string().optional().describe('Full name of the customer'),
    phone: z.string().optional().describe('phone number of the customer')
})

// .extend() herda todos os campos e sobrescreve _id como obrigatório
export const CustomerUpdateSchema = CustomerQuerySchema.extend({
    _id: z.string().describe("MongoDB ObjectId of the customer"),
})
```

Sem `.extend()`, você teria que copiar todos os campos de `CustomerQuerySchema` manualmente em `CustomerUpdateSchema` — duplicação que desincroniza quando um campo muda.

---

## `.shape` — extraindo o mapa de campos de um z.object()

`registerTool` espera que `inputSchema` e `outputSchema` sejam um **mapa de campo → schema** (objeto plano), não um `z.object()` completo. Quando o schema já foi definido como `z.object()`, `.shape` extrai esse mapa interno:

```typescript
// src/mcp/tools/updateCustomer.ts
server.registerTool("update_customer", {
    inputSchema: CustomerUpdateSchema.shape,   // extrai { _id: z.string(), name: ..., phone: ... }
    outputSchema: CustomerMutationSchema.shape, // extrai { id: ..., message: ..., isError: ..., ... }
}, ...)
```

Sem `.shape`, você passaria um `z.ZodObject` onde o SDK espera um `Record<string, ZodType>` — o TypeScript reclamaria (ou o SDK falharia em runtime).

```
CustomerUpdateSchema            ← z.ZodObject<{ _id, name, phone }>
CustomerUpdateSchema.shape      ← { _id: z.ZodString, name: z.ZodOptional<z.ZodString>, phone: ... }
                                                          ↑ isso que registerTool quer
```

---

## `.nullable()` — output que pode ser null

`get_customer` pode não encontrar o customer pedido — o resultado legítimo é `null`, não um erro. Isso precisa estar no `outputSchema` para que o SDK saiba que null é válido:

```typescript
// src/mcp/tools/getCustomer.ts
outputSchema: {
    customer: CustomerSchema.nullable()
        .describe('Customer details if found, otherwise null!')
}
```

`CustomerSchema.nullable()` produz um schema que aceita `{ _id, name, phone }` **ou** `null`. Sem o `.nullable()`, devolver `null` de `service.findCustomer()` causaria um erro de validação do `outputSchema`.

---

## `CustomerMutationSchema` — schema unificado para respostas de mutação

Create, update e delete retornam formatos parecidos mas não idênticos. Em vez de criar um schema por operação, `CustomerMutationSchema` tem todos os campos opcionais e cada operação usa o subconjunto que precisa:

```typescript
// src/domain/customer.ts
export const CustomerMutationSchema = z.object({
    id: z.string().optional(),       // create, update, delete → retorna o ID afetado
    message: z.string().optional(),  // create → "user X created!" | update → "User X updated!" | delete → "User X deleted!"
    isError: z.boolean().optional(), // true quando a operação falhou (retorno MCP padrão de erro de negócio)
    customer: CustomerSchema.optional(),          // get_customer → retorna o objeto completo
    customers: z.array(CustomerSchema).optional() // list_customers → retorna array
})
```

Todos esses campos estarem no mesmo schema criou um problema em `structuredContent`: o SDK valida o objeto retornado contra o `outputSchema`, e um objeto com campos extras (não declarados no schema da tool específica) causa erro `"data must NOT have additional properties"`. A solução foi incluir `customer` e `customers` no `CustomerMutationSchema` — mesmo que a maioria das tools não use esses campos, precisam estar no schema para que não sejam rejeitados como "campos não declarados".

```typescript
// O FIX que está no comentário do customer.ts:
// customer: CustomerSchema.optional()    ← adicionado para prevenir o erro de outputSchema
// customers: z.array(CustomerSchema)...  ← idem
```

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| `z.infer` | [src/domain/customer.ts](../src/domain/customer.ts) | `type Customer = z.infer<typeof CustomerSchema>` |
| `.extend()` | [src/domain/customer.ts](../src/domain/customer.ts) | `CustomerUpdateSchema = CustomerQuerySchema.extend(...)` |
| `.shape` | [src/mcp/tools/updateCustomer.ts](../src/mcp/tools/updateCustomer.ts) | `inputSchema: CustomerUpdateSchema.shape` |
| `.nullable()` | [src/mcp/tools/getCustomer.ts](../src/mcp/tools/getCustomer.ts) | `customer: CustomerSchema.nullable()` |
| Schema unificado | [src/domain/customer.ts](../src/domain/customer.ts) | `CustomerMutationSchema` com campos opcionais e o FIX comentado |
