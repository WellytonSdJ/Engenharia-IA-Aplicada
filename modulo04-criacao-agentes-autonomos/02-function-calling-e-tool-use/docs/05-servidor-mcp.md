# Servidor MCP do OpsPilot — de cliente a provedor de tools

> Este doc cobre a spec 007 ([`007-servidor-mcp-opspilot`](../specs/007-servidor-mcp-opspilot/spec.md)). O básico de MCP (`McpServer`, `registerTool`, transporte stdio, teste com client) foi visto no módulo 03: [01-construindo-mcp-server-do-zero.md](../../../modulo03-mcp-na-pratica/05-mcps-do-zero/docs/01-construindo-mcp-server-do-zero.md) e [02-transporte-stdio-e-testes-mcp-client.md](../../../modulo03-mcp-na-pratica/05-mcps-do-zero/docs/02-transporte-stdio-e-testes-mcp-client.md). Aqui cobrimos só o que muda.
>
> **Em 30 segundos**
>
> - Até agora o OpsPilot **usava** tools. Agora ele também **oferece** tools: qualquer cliente MCP (Copilot, Cursor, outro agente) pode listar alertas e abrir ou resolver incidentes **no mesmo banco** que o `/chat` usa.
> - São só 3 tools no MCP (`list_alerts`, `open_incident`, `resolve_incident`), tiradas da **mesma lista** de definições do agente. Não há schema duplicado.
> - Transporte **stdio**: o stdout é o canal do protocolo, então **`console.log` é proibido** no servidor.
> - `npm run mcp` sobe o servidor. Os testes conversam com ele por um transporte em memória.

---

## O que é: o agente muda de lado do balcão

No módulo 03, nossos agentes eram **clientes** de servidores MCP: consumiam tools de outros. Aqui o OpsPilot vira **servidor**: ele é o "outro".

**Analogia:** um restaurante (o OpsPilot) que, além de servir no salão (`/chat`), passa a atender por **delivery** (MCP). A cozinha é a mesma, o estoque é o mesmo (`SqliteOpsStore`) e o cardápio do delivery é um recorte do cardápio do salão.

```text
              ┌───────────────── OpsPilot ─────────────────┐
 curl ──────▶ │ POST /chat ─▶ agente ─▶ 6 tools ─┐          │
              │                                  ├─▶ SqliteOpsStore ─▶ opspilot.db
 Copilot ───▶ │ MCP (stdio) ─────────▶ 3 tools ──┘          │
              └────────────────────────────────────────────┘
```

| | Agente no `/chat` | Servidor MCP |
| --- | --- | --- |
| Quem decide chamar a tool | o **nosso** modelo (OpenRouter) | o modelo **do cliente** (ex.: Copilot) |
| Tools expostas | 6 | 3 (`list_alerts`, `open_incident`, `resolve_incident`) |
| Transporte | HTTP | stdio |
| Store | `SqliteOpsStore` | `SqliteOpsStore` (mesmo `OPSPILOT_DB`) |

> ✅ **Checkpoint**: MCP aqui **não** é outro agente. É outra **porta** para as mesmas tools e o mesmo banco.

---

## Como está no projeto

```typescript
// src/mcp/server.ts
const MCP_TOOL_NAMES = ["list_alerts", "open_incident", "resolve_incident"] as const;

export function createMcpServer(store: OpsStore): McpServer {
  const server = new McpServer({ name: "opspilot", version: "1.0.0" });
  const definitions = createOpsToolDefinitions(store).filter((definition) =>
    (MCP_TOOL_NAMES as readonly string[]).includes(definition.name),
  );
  for (const definition of definitions) {
    server.registerTool(
      definition.name,
      {
        description: definition.description,
        inputSchema: definition.inputSchema,
      },
      async (input) => ({ content: [{ type: "text", text: await definition.run(input) }] }),
    );
  }
  return server;
}
```

Três coisas para notar:

1. **Uma fonte de verdade.** `createOpsToolDefinitions(store)` é a mesma função que alimenta o agente ([01-function-calling-e-design-de-tools.md](./01-function-calling-e-design-de-tools.md)). Melhorou uma descrição? O Copilot vê a melhoria também.
2. **Allowlist explícita.** `MCP_TOOL_NAMES` decide o que sai pela porta. `check_provider_status` (rede) e `consultar_runbook` ficaram de fora nesta versão.
3. **`createMcpServer(store)` recebe o store.** Produção passa o SQLite real; o teste passa um `:memory:`.

### O entrypoint e a regra de ouro do stdio

```typescript
// src/mcp/server.ts
export async function startStdioServer(): Promise<void> {
  const store = new SqliteOpsStore();
  const server = createMcpServer(store);
  const transport = new StdioServerTransport();
  await server.connect(transport);
}
```

No stdio, **cada byte no stdout precisa ser uma mensagem do protocolo**. Um `console.log("debug")` ali no meio corrompe a conversa com o cliente. Por isso o servidor só usa `console.error` (stderr), inclusive no erro de inicialização.

> ✅ **Checkpoint**: em servidor MCP stdio, `console.log` = bug. Diagnóstico vai para o stderr.

### Testando sem processo nem stdio

```typescript
// src/mcp/server.test.ts
async function connectedClient(store: SqliteOpsStore) {
  const server = createMcpServer(store);
  const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "opspilot-test-client", version: "0.0.0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  return client;
}
```

`InMemoryTransport.createLinkedPair()` liga cliente e servidor por um "cano" em memória. Os testes conferem que o catálogo tem **exatamente** as 3 tools, que o servidor se chama `opspilot` e que cada tool via MCP devolve o mesmo que a tool do agente sobre o mesmo store.

---

## Rodando

```powershell
npm run seed   # garante o banco com o cenário Mercadinho
npm run mcp    # fica esperando um cliente no stdin/stdout
```

O script carrega o `.env` antes (`node --env-file-if-exists=.env --import tsx src/mcp/server.ts`), para o servidor abrir o mesmo `OPSPILOT_DB` do `/chat`.

---

## 🧪 Exercício: a prova circular

A demonstração mais legal desta unidade não está automatizada no projeto, e fica como exercício:

1. Registre o servidor no seu cliente MCP (ex.: `.vscode/mcp.json` no VS Code), com o comando `npm run mcp` na pasta do projeto.
2. Peça ao Copilot: *"abra um incidente sev2 para o checkout"*. Ele chama `open_incident` **via MCP**.
3. Pergunte ao OpsPilot pelo `/chat`: *"quais incidentes estão abertos?"*. O incidente criado pelo Copilot aparece.

Um agente (Copilot) agiu, e outro agente (OpsPilot) enxergou o resultado. Eles não conversam entre si: eles **compartilham o mundo** (o banco). É uma prévia de sistemas multi-agente.

---

## ⚠️ Pegadinhas

- **`console.log` "só para debugar".** Quebra o protocolo. Use `console.error`.
- **Schemas paralelos para o MCP.** Copiar o schema "só para ajustar uma coisinha" cria dois cardápios que vão divergir. Filtre a lista única.
- **Achar que o MCP usa o nosso modelo.** Quem decide chamar as tools no MCP é o modelo do **cliente**. Por isso as descrições boas (as 6 regras) valem ainda mais aqui.
- **Bancos diferentes sem querer.** Rodar o MCP sem o `.env` pode apontar para outro arquivo, e aí a prova circular "não funciona".

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Servidor | [`src/mcp/server.ts`](../src/mcp/server.ts) | `createMcpServer`, `MCP_TOOL_NAMES`, `startStdioServer`, só `console.error` |
| Fonte única | [`src/services/tools.ts`](../src/services/tools.ts) | `createOpsToolDefinitions` |
| Testes | [`src/mcp/server.test.ts`](../src/mcp/server.test.ts) | `InMemoryTransport`, catálogo de 3 tools, paridade com o agente |
| Script | [`package.json`](../package.json) | `"mcp": "node --env-file-if-exists=.env --import tsx src/mcp/server.ts"` |
| Spec | [`specs/007-servidor-mcp-opspilot/spec.md`](../specs/007-servidor-mcp-opspilot/spec.md) | Escopo v1, regra do stdout |

---

⬅️ Anterior: [04 · `POST /chat`](./04-api-chat-http.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [06 · Conversa persistente](./06-conversa-persistente.md)**
