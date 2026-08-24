import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerQuerySchema } from "../../domain/customer.ts";

export function registerFindCustomerPrompt(server: McpServer) {
    server.registerPrompt(
        "find_customer_prompt",
        {
            description: "Prompt to search a customer using any combination of _id, name or phone",
            // argsSchema reutiliza CustomerQuerySchema.shape — mesmos campos opcionais da query.
            argsSchema:  CustomerQuerySchema.shape
        },
        (query) => ({
            messages: [
                {
                    role: "user",
                    content: {
                        type: "text",
                        // O prompt não chama a tool diretamente — ele monta uma mensagem de usuário
                        // que instrui o agente a usar get_customer ou list_customers.
                        // É o agente que decide qual tool chamar a partir desta instrução.
                        text: `Please find the customer matching the following query using the get_customer or list_customers tool.\nQuery: ${JSON.stringify(query)}`,
                    }
                }
            ]
        })
    )
}
