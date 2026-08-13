# Glossário

Referência rápida. Para profundidade, vá ao documento específico de cada conceito.

Termos gerais de MCP (MCP, MCP Server, MCP Client, STDIO transport, Tool, Tool calling) já cobertos no [glossário de `05-safeguard-prompt-injection`](../../../modulo02-integracao-apis-llms/05-safeguard-prompt-injection/docs/glossario.md) não são repetidos aqui — este glossário cobre só o que é novo por este projeto construir um servidor (não consumir um).

---

## SDK do lado servidor (`@modelcontextprotocol/sdk`)

| Termo | Definição |
| --- | --- |
| **`McpServer`** | Classe do SDK oficial que representa um servidor MCP completo. Instanciada uma vez (`new McpServer({ name, version })`) e usada para registrar tudo que o servidor expõe. |
| **`registerTool`** | Método que registra uma tool: nome, `description`, `inputSchema`, `outputSchema` (mapas de campo Zod) e um handler assíncrono que executa a ação e devolve `content`/`structuredContent`. |
| **`registerResource`** | Método que registra um resource: identificador, URI (ex: `encryption://info`), metadados e um handler que devolve `contents` — dado somente-leitura, sem parâmetros de entrada. |
| **`registerPrompt`** | Método que registra um prompt reutilizável: `description`, `argsSchema` e um handler que monta um array de `messages` prontas para entrar no contexto de um agente. |
| **`inputSchema` / `outputSchema` como mapa de campos** | Diferente de um `z.object({...})` completo, `registerTool` espera um objeto onde cada chave já é um schema Zod individual — o SDK monta a validação a partir desse mapa. |
| **`structuredContent`** | Parte da resposta de uma tool com o dado tipado conforme `outputSchema`, para clientes que sabem parsear — sempre devolvida junto de `content` (texto genérico). |
| **`isError: true`** | Convenção MCP para sinalizar que a tool executou, mas o resultado é uma falha de negócio (ex: passphrase errada) — diferente de deixar a exceção estourar como erro de protocolo. |
| **`StdioServerTransport`** | Implementação de transporte usada pelo servidor para falar JSON-RPC via stdin/stdout com o processo que o invocou. |

---

## SDK do lado cliente, usado nos testes

| Termo | Definição |
| --- | --- |
| **`Client`** (`@modelcontextprotocol/sdk/client`) | Classe que representa um cliente MCP. Instanciado com `{ name, version }` e conectado a um transporte via `client.connect(transport)`. |
| **`StdioClientTransport`** | Transporte usado pelo `Client` para spawnar o servidor como subprocesso (`command`/`args`) e falar com ele via stdin/stdout — o mesmo mecanismo que o VS Code usa para subir o `ciphersuite-mcp`. |
| **`client.callTool`** | Chama uma tool registrada no servidor, passando `name` e `arguments`; devolve `content`/`structuredContent`. |
| **`client.listResources`** | Lista os resources disponíveis no servidor conectado. |
| **`client.getPrompt`** | Busca um prompt registrado, já preenchido com os `arguments` passados, devolvendo o array de `messages`. |
| **Teste de contrato via MCP client** | Padrão de teste deste projeto: em vez de testar `encrypt`/`decrypt` de `service.ts` diretamente, os testes abrem um `Client` real contra o servidor real (via STDIO) e chamam tools/resources/prompts como um agente faria — valida o protocolo inteiro, não só a lógica interna. |

---

## Domínio (criptografia)

| Termo | Definição |
| --- | --- |
| **AES-256-CBC** | Cifra de bloco simétrica usada pelas tools `encrypt_message`/`decrypt_message`, via `node:crypto` (`createCipheriv`/`createDecipheriv`). |
| **`scryptSync`** | Função de derivação de chave (KDF) usada para transformar qualquer passphrase de tamanho livre numa chave de 32 bytes exigida pelo AES-256 — deliberadamente cara de computar, o que dificulta força bruta. |
| **Salt fixo (`SALT`)** | Constante hardcoded usada na derivação de chave (`scryptSync(passphrase, SALT, 32)`). Simplificação didática: garante que a mesma passphrase sempre derive a mesma chave, mas não protege contra rainbow tables entre instalações diferentes do servidor. |
| **IV (Initialization Vector)** | 16 bytes aleatórios gerados a cada chamada de `encrypt` (`randomBytes(16)`) — garante que a mesma mensagem cifrada duas vezes com a mesma chave produza saídas diferentes. Não é secreto: viaja junto no formato de saída. |
| **Formato `iv:ciphertext`** | Formato de saída das tools de criptografia: IV e ciphertext, cada um em hexadecimal, concatenados com `:`. Precisa ser mantido inteiro para a decriptação funcionar. |
