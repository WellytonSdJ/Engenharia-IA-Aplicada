import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { registerListCustomersTool } from "./tools/listCustomers.ts";
import { CustomerService } from "../application/customerService.ts";
import { registerApiInfoResource } from "./resources/apiInfo.ts";
import { registerCreateCustomersTool } from "./tools/createCustomer.ts";
import { registerGetCustomerTool } from "./tools/getCustomer.ts";
import { registerFindCustomerPrompt } from "./prompts/findCustomer.ts";
import { registerUpdateCustomersTool } from "./tools/updateCustomer.ts";
import { registerDeleteCustomersTool } from "./tools/deleteCustomer.ts";

// BASE_URL aponta para a API legada que este MCP server embrulha.
// Trocar aqui reflete em todas as tools sem alterar cada arquivo individualmente.
const BASE_URL = "http://localhost:9999/v1";

// O service é criado aqui e injetado nas tools — todas compartilham a mesma instância (singleton implícito).
const service = new CustomerService(BASE_URL)

export const server = new McpServer({
    name: "@erickwendel/ew-customers-mcp",
    version: "0.0.1",
});

// Cada register* é responsável por registrar uma tool/resource/prompt no servidor.
// Padrão "um arquivo por tool" em vez do mcp.ts monolítico do projeto anterior (05-mcps-do-zero).
registerListCustomersTool(server, service)
registerGetCustomerTool(server, service)
registerCreateCustomersTool(server, service)
registerFindCustomerPrompt(server)
registerApiInfoResource(server, BASE_URL)
registerUpdateCustomersTool(server, service)
registerDeleteCustomersTool(server, service)
