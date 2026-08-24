import { type McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { CustomerService } from "../../application/customerService.ts";
import { CustomerMutationSchema, CustomerUpdateSchema } from "../../domain/customer.ts";

export function registerUpdateCustomersTool(
    server: McpServer,
    service: CustomerService
) {

    server.registerTool(
        "update_customer",
        {
            description:
                "Update an existing customer's name and/or phone number by their _id",

            // .shape extrai o mapa de campos do z.object() — registerTool espera um mapa de campo→schema,
            // não um z.object() completo. Usar .shape evita duplicar os campos já definidos no domínio.
            inputSchema: CustomerUpdateSchema.shape,
            outputSchema: CustomerMutationSchema.shape,
        },
        async (customer) => {
            try {
                const result = await service.updateCustomer(customer)
                return {
                    content: [
                        {
                            type: "text",
                            text: result.message ?? ""
                        }
                    ],
                    structuredContent: result
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
