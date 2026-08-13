# Documentação — MCPs do Zero (ciphersuite-mcp)

Documentação de estudo do projeto `05-mcps-do-zero`, o quinto do módulo 3 (MCP na Prática).

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura ordenada, mapa do código, fluxo do projeto, como rodar |
| [construindo-mcp-server-do-zero.md](./construindo-mcp-server-do-zero.md) | Como um servidor MCP é construído com o SDK oficial: `McpServer`, `registerTool`, `registerResource`, `registerPrompt` |
| [transporte-stdio-e-testes-mcp-client.md](./transporte-stdio-e-testes-mcp-client.md) | Transporte STDIO (servidor como subprocesso) e como os testes conectam um `Client` MCP real ao servidor para validar o comportamento ponta a ponta |
| [criptografia-aes-256-cbc.md](./criptografia-aes-256-cbc.md) | As decisões de criptografia por trás das tools: derivação de chave com `scrypt`, IV aleatório por chamada, formato de saída |
| [glossario.md](./glossario.md) | Todos os termos novos deste projeto — referência rápida |

---

## Contexto do projeto

Servidor MCP construído do zero (`@erickwendel/ciphersuite-mcp`), sem nenhum framework de agente por cima — só o SDK oficial `@modelcontextprotocol/sdk`. Ele expõe:

- **2 tools**: `encrypt_message` e `decrypt_message` (AES-256-CBC)
- **1 resource**: `encryption://info`, descrevendo o algoritmo em texto
- **1 prompt**: `encrypt_message_prompt`, pronto para uso no Copilot Chat (o README raiz também listava um `decrypt_message_prompt`, mas ele não está implementado em `src/mcp.ts` nem coberto pelos testes — só existe a tool `decrypt_message`)
- **Transporte STDIO**, rodando como subprocesso a partir de um cliente MCP (VS Code, MCP Inspector, ou o próprio `Client` de teste)
- **Testes automatizados** que abrem um cliente MCP real via `StdioClientTransport` e chamam as tools/resources/prompts como um agente faria — não são testes unitários das funções de `service.ts`, e sim testes de contrato do protocolo
