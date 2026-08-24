# Documentação — 06-your-legacy-api-as-mcp

Este projeto demonstra como transformar uma **API REST legada** (Fastify + MongoDB) em um servidor MCP sem modificar uma linha do código original — só construindo uma camada de adaptação em cima.

O sub-projeto `customers-mcp/` é o servidor MCP. O sub-projeto `nodejs-fastify-mongodb-crud/` é a API legada que ele embrulha.

---

## Tecnologias deste projeto

| Tecnologia | Papel no projeto |
| --- | --- |
| `@modelcontextprotocol/sdk` | SDK do servidor MCP (já visto em 05-mcps-do-zero) |
| `zod` | Validação dos schemas de domínio, input e output das tools |
| Fastify | Framework HTTP da API legada (alto desempenho + validação por JSON Schema) |
| MongoDB | Banco de dados da API legada |
| Node.js 24 (native TS) | Execução de `.ts` sem compilação (`--experimental-strip-types`) |
| Node.js Test Runner | Testes tanto da API legada quanto do servidor MCP |

---

## Documentos

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Por onde começar: motivação, trilha de leitura, mapa do código e comandos |
| [legacy-api-como-mcp.md](./legacy-api-como-mcp.md) | O conceito central: por que e como embrulhar uma API REST legada como servidor MCP |
| [arquitetura-em-camadas-mcp.md](./arquitetura-em-camadas-mcp.md) | As quatro camadas do customers-mcp (domain, infrastructure, application, mcp) e por que essa divisão importa |
| [zod-schemas-e-tipos.md](./zod-schemas-e-tipos.md) | Zod no contexto deste projeto: `z.infer`, `.extend()`, `.shape`, `.nullable()`, e `CustomerMutationSchema` unificado |
| [glossario.md](./glossario.md) | Termos novos deste projeto — os já cobertos em 05-mcps-do-zero não são repetidos aqui |
