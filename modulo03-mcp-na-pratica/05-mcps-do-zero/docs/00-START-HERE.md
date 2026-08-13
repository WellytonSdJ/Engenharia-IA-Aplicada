# Por onde começar

Este é o quinto projeto do módulo 3 (MCP na Prática). Os projetos anteriores (01, e o `mcp.md` de `05-safeguard-prompt-injection` no módulo 02) sempre consumiram servidores MCP prontos (filesystem, MongoDB). Aqui a virada é: **construir um servidor MCP do zero**, do lado de quem expõe as ferramentas, não de quem as consome.

---

## O que estamos construindo e por quê

> Não existe LLM neste projeto. O `ciphersuite-mcp` é só um servidor: ele expõe tools, um resource e prompts via protocolo MCP. Quem decide chamá-los é o cliente (VS Code Copilot Chat, o MCP Inspector, ou o `Client` usado nos testes).

O domínio escolhido é simples de propósito — criptografia simétrica (AES-256-CBC) — para que a atenção fique inteira na mecânica do protocolo: como registrar uma tool com schema de entrada/saída validado por Zod, como expor um resource com uma URI própria, como empacotar um prompt reutilizável, e como tudo isso conversa via STDIO com um processo cliente.

```
Projetos anteriores (01, mcp.md/05):    Este projeto (05-mcps-do-zero):
Cliente MCP → consome servidor pronto.  Servidor MCP → construído do zero, do lado de quem expõe.
```

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [construindo-mcp-server-do-zero.md](./construindo-mcp-server-do-zero.md) | O núcleo do projeto: `McpServer`, `registerTool`, `registerResource`, `registerPrompt` — a API que qualquer servidor MCP feito com o SDK oficial usa. |
| 2 | [transporte-stdio-e-testes-mcp-client.md](./transporte-stdio-e-testes-mcp-client.md) | Como o servidor roda como subprocesso via STDIO, e como os testes abrem um `Client` MCP de verdade para validar tools, resource e prompts ponta a ponta. |
| 3 | [criptografia-aes-256-cbc.md](./criptografia-aes-256-cbc.md) | As decisões de criptografia por trás das duas tools — não é o foco do módulo, mas explica por que o código de `service.ts` é do jeito que é. |
| 4 | [glossario.md](./glossario.md) | Referência rápida de todos os termos novos. Consulte quando encontrar algo que não reconhece. |

---

## Mapa do código

```
src/index.ts     → entry point: cria o StdioServerTransport e conecta o server a ele
src/mcp.ts       → define o McpServer e registra as 2 tools, o resource e os 2 prompts
src/service.ts   → lógica pura de criptografia (encrypt/decrypt), sem nada de MCP

tests/helpers.ts   → createTestClient(): sobe o servidor como subprocesso e conecta um Client MCP nele
tests/mcp.test.ts  → chama tools via client.callTool, lista resources via client.listResources,
                     busca prompts via client.getPrompt — testa o servidor pelo protocolo, não pelas funções internas

.vscode/mcp.json   → registra o ciphersuite-mcp como servidor MCP disponível no Copilot Chat da VS Code
package.json       → scripts: start, dev, test, test:dev, mcp:inspect
```

---

## O fluxo em uma linha

```
Cliente MCP (VS Code / Inspector / teste) → stdin/stdout → src/index.ts → server.connect(transport)
   → McpServer despacha para a tool/resource/prompt registrada em src/mcp.ts → service.ts faz o trabalho real
```

---

## Como rodar

```bash
# 1. Instalar dependências
npm install

# 2. Explorar tools, resource e prompts interativamente numa UI web
npm run mcp:inspect
# abre http://localhost:5173 já conectado ao servidor

# 3. Rodar a suíte de testes (abre um Client MCP real contra o servidor)
npm test
```

Não é preciso build: `node --experimental-strip-types` roda o TypeScript diretamente (ver `scripts` em `package.json`).
