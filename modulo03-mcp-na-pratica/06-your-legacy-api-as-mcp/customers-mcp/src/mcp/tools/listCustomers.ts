import { type McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerService } from "../../application/customerService.ts";
import z from "zod";
import { CustomerSchema } from "../../domain/customer.ts";


// inputSchema: {} — tool sem argumentos de entrada. O SDK aceita objeto vazio para indicar "sem parâmetros".
export function registerListCustomersTool(
    server: McpServer,
    service: CustomerService
) {

    server.registerTool(
        "list_customers",
        {
            description: "List all customers",
            inputSchema: {},
            outputSchema: {
                // outputSchema define a shape do structuredContent — o SDK valida o retorno contra ela.
                customers: z.array(
                    CustomerSchema
                ).describe('Array of all customers')
            }
        },
        async () => {
            try {
                const customers = await service.listCustomers()
                return {
                    // content: texto livre para o LLM/humano ler
                    // structuredContent: dado tipado para clientes que sabem parsear (testes, agentes avançados)
                    content: [
                        {
                            type: "text",
                            text: JSON.stringify(customers, null, 2)
                        }
                    ],
                    structuredContent: { customers }
                }
            } catch (error) {
                  return {
                    isError: true,
                    content: [
                        {
                            type: "text",
                            text: `Failed to list customers. Error: ${error instanceof Error ? error.message : String(error)}`,
                        },
                    ],
                };
            }
        }
    )
}
