# Construindo um servidor MCP do zero

## O objeto central: `McpServer`

Todo o servidor gira em torno de uma instância de `McpServer`, do SDK oficial `@modelcontextprotocol/sdk`:

```typescript
// src/mcp.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

export const server = new McpServer({
    name: '@erickwendel/ciphersuite-mcp',
    version: '0.0.1'
})
```

`name` e `version` são metadados de identidade — é o que um cliente MCP (VS Code, Inspector) mostra ao listar servidores conectados. A partir daqui, tudo que o servidor expõe é registrado chamando métodos nessa instância: `registerTool`, `registerResource`, `registerPrompt`. Não existe roteamento manual de JSON-RPC — o SDK cuida disso; o código só declara "o que existe".

---

## Tools: `registerTool`

Uma tool é uma função que o cliente (tipicamente guiado por um LLM) pode chamar com argumentos e receber um resultado estruturado de volta. O registro tem três partes: nome, definição (schemas + descrição), e o handler.

```typescript
server.registerTool(
    'encrypt_message',
    {
        description: 'Encrypt a message',
        inputSchema: {
            message: z.string().describe("The message to encrypt"),
            encryptionKey: z.string().describe(
                "Any passphrase to use for encryption — the server derives a strong key from it automatically"
            )
        },
        outputSchema: {
            encryptedMessage: z.string().describe(
                "The encrypted message (format: iv:ciphertext)"
            )
        }
    },
    async ({ message, encryptionKey }) => {
        try {
            const encryptedMessage = encrypt(message, encryptionKey)
            return {
                content: [{ type: "text", text: encryptedMessage }],
                structuredContent: { encryptedMessage }
            }
        } catch (error) {
            return {
                isError: true,
                content: [{ type: 'text', text: `Failed to encrypt message! ...` }]
            }
        }
    }
)
```

Pontos que não são óbvios:

- **`inputSchema` e `outputSchema` são objetos Zod "soltos"** (um mapa de campo → schema), não um `z.object({...})` completo. O SDK monta o objeto JSON Schema por trás das cenas a partir desse mapa — é a forma que `registerTool` espera, diferente de como normalmente se usa Zod em outros projetos do repositório.
- **`.describe(...)` em cada campo não é decoração**: é isso que vira a descrição do parâmetro que o cliente (e o LLM por trás dele) vê ao decidir como chamar a tool. Sem `describe`, o LLM só tem o nome do campo para adivinhar o formato esperado.
- **`content` vs `structuredContent`**: toda resposta de tool carrega `content` (texto/blocos genéricos, para exibição humana ou para um LLM que só lê texto) e, quando há `outputSchema`, também `structuredContent` (o mesmo dado, mas tipado, para clientes que sabem parsear). O código sempre popula os dois.
- **Erros não lançam exceção — retornam `isError: true`**: em vez de deixar o `throw` do `service.ts` propagar (o que derrubaria a chamada JSON-RPC como erro de protocolo), o handler captura e devolve um resultado normal com `isError: true` e uma mensagem explicando a causa. Isso é o padrão MCP para "a tool rodou, mas o resultado é uma falha de negócio" — diferente de uma falha de infraestrutura do próprio protocolo.

A tool `decrypt_message` segue exatamente o mesmo formato, espelhando `encrypt_message` (troca `encryptionKey`+`message` na entrada, e `decryptedMessage` na saída).

---

## Resources: `registerResource`

Um resource é dado somente-leitura, identificado por URI, que o cliente pode listar e buscar sob demanda — diferente de uma tool, não recebe argumentos de entrada nem executa uma ação.

```typescript
server.registerResource(
    'encryption://info',
    'encryption://info',
    {
        description: 'Describes the encryption algorithm, key requirements, and output format used by this server',
    },
    () => ({
        contents: [
            {
                uri: "encryption://info",
                mimeType: "text/plain",
                text: `Algorithm : AES-256-CBC\n...`.trim(),
            },
        ]
    })
)
```

Aqui, `registerResource('encryption://info', 'encryption://info', ...)` usa a própria URI como nome — o primeiro argumento é um identificador (pode ser qualquer string), o segundo é a URI real que o cliente usa para buscar o conteúdo. O handler não recebe parâmetros porque o conteúdo é estático: é só a documentação do algoritmo, servida como texto plano, para o cliente (ou o LLM) consultar antes de decidir chamar `encrypt_message`/`decrypt_message` — uma forma de MCP substituir um "leia o README primeiro" por dado consultável em tempo de execução.

---

## Prompts: `registerPrompt`

Um prompt registrado no MCP é um template de mensagem parametrizável que o cliente pode buscar pronto — útil para padronizar como o usuário aciona uma tool, sem precisar redigitar a instrução.

```typescript
server.registerPrompt(
    "encrypt_message_prompt",
    {
        description: "Prompt to encrypt a plain-text message using the encrypt_message tool",
        argsSchema: {
            message: z.string().describe("The message to encrypt"),
            encryptionKey: z.string().describe(...)
        }
    },
    ({ message, encryptionKey }) => ({
        messages: [
            {
                role: 'user',
                content: {
                    type: "text",
                    text: `Please encrypt the following message using the encrypt_message tool.\nMessage: ${message}\nEncryption key: ${encryptionKey}`,
                }
            }
        ]
    })
)
```

O prompt não chama a tool diretamente — ele só monta a mensagem de usuário que, uma vez injetada no contexto de um agente, leva o próprio agente a decidir chamar `encrypt_message`. É por isso que o `argsSchema` espelha o `inputSchema` da tool correspondente: o prompt existe para alimentar a tool com os mesmos dados, só que empacotados como uma frase pronta em vez de argumentos de função.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Instância do servidor | [src/mcp.ts](../src/mcp.ts) | `new McpServer({ name, version })` |
| Tools | [src/mcp.ts](../src/mcp.ts) | `registerTool('encrypt_message', ...)` / `registerTool('decrypt_message', ...)` |
| Resource | [src/mcp.ts](../src/mcp.ts) | `registerResource('encryption://info', ...)` |
| Prompt | [src/mcp.ts](../src/mcp.ts) | `registerPrompt('encrypt_message_prompt', ...)` |
| Lógica de negócio isolada do protocolo | [src/service.ts](../src/service.ts) | `encrypt`/`decrypt` não sabem nada de MCP — os handlers em `mcp.ts` são a única camada que conhece o SDK |
