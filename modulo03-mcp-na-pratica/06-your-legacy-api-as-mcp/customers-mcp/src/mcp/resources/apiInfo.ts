import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

// Resource como documentação viva da API legada: o agente consulta customers://api-info
// antes de chamar tools, exatamente como leria um README — mas de forma programática.
// Resources são somente-leitura e sem argumentos (ao contrário de tools).
export function registerApiInfoResource(
    server: McpServer,
    baseUrl: string
){
    server.registerResource(
        "customers://api-info",
        "customers://api-info",
        {
            description: "describes the customers rest API that this MCP server wraps"
        },
        () => ({
            contents: [
                {
                    uri: "customers://api-info",
                    mimeType: "text/plain",
                    // O texto descreve o contrato da API legada (endpoints, formato de customer).
                    // Isso permite ao LLM entender a fonte de dados sem depender de documentação externa.
                    text: `
Customers API

  Base URL : ${baseUrl}
  Endpoints:
    GET    /customers          — list all customers
    GET    /customers/:id      — get customer by id
    POST   /customers          — create customer  { name, phone }
    PUT    /customers/:id      — update customer  { name, phone }
    DELETE /customers/:id      — delete customer

  Customer shape: { _id: string, name: string, phone: string }
`
                }
            ]
        })
    )
}
