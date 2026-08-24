import z from "zod";

// Formato base de um customer — _id é optional porque ao criar ainda não existe ID (gerado pelo MongoDB).
// Mesma shape usada tanto na resposta do banco quanto no output de tools que retornam um customer completo.
export const CustomerSchema = z.object({
    _id: z.string().optional(),
    name: z.string(),
    phone: z.string(),
})

// z.infer<> extrai o tipo TypeScript diretamente do schema Zod — sem duplicar a tipagem manualmente.
export type Customer = z.infer<typeof CustomerSchema>

// Todos os campos opcionais para permitir busca por qualquer combinação (_id, name, phone).
// O .describe() em cada campo é o que o LLM lê ao decidir como preencher o argumento — não é decoração.
export const CustomerQuerySchema = z.object({
    _id: z.string().optional().describe("MongoDB ObjectId of the customer"),
    name: z.string().optional().describe('Full name of the customer'),
    phone: z.string().optional().describe('phone number of the customer')
})
export type CustomerQuery = z.infer<typeof CustomerQuerySchema>

// .extend() herda todos os campos do CustomerQuerySchema e sobrescreve _id como obrigatório.
// Para update, precisamos saber qual registro editar — o _id deixa de ser opcional.
export const CustomerUpdateSchema = CustomerQuerySchema.extend({
    _id: z.string().describe("MongoDB ObjectId of the customer"),
})

export type  CustomerUpdate = z.infer<typeof CustomerUpdateSchema>

// Schema de resposta unificado para mutações (create, update, delete, get).
// Todos os campos são opcionais porque cada operação usa um subconjunto diferente:
// create → id + message | update → id + message | delete → id + message | get → customer | list → customers.
export const CustomerMutationSchema = z.object({
    id: z.string().optional().describe("MongoDB ObjectId of the customer"),
    message: z.string().optional().describe('Confirmation message'),
    isError: z.boolean().optional().describe('Indicates if an error occurred'),

        /* FIX: Tinham faltado estes abaixo para previnir o erro de:
    "   Structured content does not match tool's output schema:
        data must NOT have additional properties"
    */
    customer: CustomerSchema.optional().describe("The found customer"),
    customers: z.array(CustomerSchema).optional().describe("List of customers"),
})

export type CustomerMutation = z.infer<typeof CustomerMutationSchema>
