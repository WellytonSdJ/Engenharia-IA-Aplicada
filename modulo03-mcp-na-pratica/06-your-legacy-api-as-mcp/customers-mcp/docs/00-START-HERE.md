# Por onde começar

Este é o sexto projeto do módulo 3 (MCP na Prática). O projeto anterior (`05-mcps-do-zero`) construiu um servidor MCP do zero partindo de uma lógica de domínio simples (criptografia AES). Aqui a pergunta é diferente: **e se o domínio já existe numa API REST que não posso mudar?**

---

## O que estamos construindo e por quê

> A API legada (`nodejs-fastify-mongodb-crud`) é um CRUD de customers com Fastify + MongoDB — existe, funciona, tem testes, e ninguém vai reescrever ela só porque surgiu um novo cliente (o LLM). O `customers-mcp` é a camada que faz a ponte: expõe o mesmo CRUD como tools, resource e prompt MCP, sem tocar na API original.

```
Projeto anterior (05-mcps-do-zero):          Este projeto (06-your-legacy-api-as-mcp):
Domínio criado do zero dentro do MCP.        Domínio já existe (API REST) → camada MCP em cima.
Um arquivo mcp.ts com tudo.                  Quatro camadas separadas (domain / infra / app / mcp).
```

A separação em camadas é o que permite que o MCP server não saiba nada de HTTP, e a API legada não saiba nada de MCP — cada peça tem uma responsabilidade única.

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [legacy-api-como-mcp.md](./legacy-api-como-mcp.md) | A motivação e o padrão central: o que significa "embrulhar uma API como MCP" e como isso se parece no código real |
| 2 | [arquitetura-em-camadas-mcp.md](./arquitetura-em-camadas-mcp.md) | As quatro camadas do projeto e como elas se comunicam — entender isso facilita muito a leitura de cada arquivo |
| 3 | [zod-schemas-e-tipos.md](./zod-schemas-e-tipos.md) | Os schemas Zod compartilhados entre camadas: `z.infer`, `.extend()`, `.shape` — padrões que aparecem em vários arquivos |
| 4 | [glossario.md](./glossario.md) | Referência rápida dos termos novos. Conceitos de servidor MCP já cobertos em `05-mcps-do-zero/docs/` não são repetidos |

---

## Mapa do código

```
nodejs-fastify-mongodb-crud/          ← API legada (não precisa ser modificada)
  src/config.js                       → lê variáveis de ambiente e monta a connection string
  src/db.js                           → conecta ao MongoDB e expõe a collection customers
  src/index.js                        → Fastify com as 5 rotas CRUD + hooks CORS e onClose
  test/api.test.js                    → testes da API legada com Node.js Test Runner

customers-mcp/                        ← servidor MCP que embrulha a API legada
  src/domain/customer.ts              → schemas Zod (Customer, CustomerQuery, CustomerUpdate, CustomerMutation)
  src/infrastructure/
    customerHttpClient.ts             → faz fetch() para a API legada; única camada que conhece HTTP
  src/application/
    customerService.ts                → lógica de negócio (findCustomer com busca flexível)
  src/mcp/
    server.ts                         → instancia McpServer e registra todas as tools/resource/prompt
    tools/listCustomers.ts            → tool list_customers (sem argumentos)
    tools/createCustomer.ts           → tool create_customer (name, phone)
    tools/getCustomer.ts              → tool get_customer (busca por _id, name ou phone)
    tools/updateCustomer.ts           → tool update_customer (_id obrigatório + campos a atualizar)
    tools/deleteCustomer.ts           → tool delete_customer (_id obrigatório)
    resources/apiInfo.ts              → resource customers://api-info (documentação da API legada)
    prompts/findCustomer.ts           → prompt find_customer_prompt (instrui agente a usar get_customer)
  src/index.ts                        → entry point: StdioServerTransport + server.connect()
  tests/helpers.ts                    → createTestClient(): sobe o MCP server como subprocesso
  tests/tools/customers.test.ts       → testa CRUD completo via client.callTool()
  tests/resources/apiInfo.test.ts     → testa listagem do resource via client.listResources()
  tests/prompts/findCustomer.test.ts  → testa o prompt via client.getPrompt()
  .vscode/mcp.json                    → registra o customers-mcp no Copilot Chat do VS Code
```

---

## O fluxo em uma linha

```
Agente (VS Code / Inspector / teste)
  → STDIO → src/index.ts → McpServer
    → tools → CustomerService → CustomerHttpClient → fetch()
      → API legada (Fastify :9999) → MongoDB
```

---

## Como rodar

```bash
# 1. Subir o MongoDB e a API legada
cd nodejs-fastify-mongodb-crud
docker compose up -d mongodb
npm start
# API disponível em http://localhost:9999

# 2. Em outro terminal: testar o servidor MCP ponta a ponta
cd ../customers-mcp
npm install
npm test

# 3. Explorar tools, resource e prompt no MCP Inspector
npm run mcp:inspect
# abre http://localhost:5173 já conectado ao servidor
```

> Pré-requisito: Docker rodando (para o MongoDB da API legada) e Node.js 24+.
