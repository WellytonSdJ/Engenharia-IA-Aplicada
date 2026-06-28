# MCP — Model Context Protocol

## O que é

MCP é um protocolo aberto (criado pela Anthropic) que padroniza como LLMs se conectam a ferramentas e recursos externos. Em vez de cada aplicação inventar sua própria forma de dar "poderes" ao modelo, o MCP define um contrato padrão: um **servidor MCP** expõe ferramentas, e um **cliente MCP** conecta o modelo a esses servidores.

A analogia mais direta é um plugin de navegador: o navegador (o modelo) tem uma interface padrão, e qualquer extensão (servidor MCP) que siga o contrato pode ser instalada sem modificar o navegador em si.

```
Sem MCP:
LLM ──► resposta em texto

Com MCP:
LLM ──► decide usar uma tool ──► MCP client ──► MCP server ──► executa ação real
                                                               (ler arquivo, consultar BD, chamar API)
         ◄───────────────────── resultado ◄──────────────────
```

---

## Como o MCP está sendo usado neste projeto

O projeto usa o pacote `@modelcontextprotocol/server-filesystem` — um servidor MCP oficial que expõe operações de leitura do sistema de arquivos.

```typescript
// src/services/mcpService.ts
import { MultiServerMCPClient } from "@langchain/mcp-adapters"

export const getMCPTools = async () => {
  const mcpClient = new MultiServerMCPClient({
    filesystem: {
      transport: 'stdio',               // comunicação via stdin/stdout
      command: 'npx',
      args: [
        '-y',
        '@modelcontextprotocol/server-filesystem',
        process.cwd()                   // escopo: apenas o diretório atual do projeto
      ]
    },
  })

  return mcpClient.getTools()           // retorna as tools como objetos LangChain
}
```

O `mcpClient.getTools()` retorna um array de tools no formato que o LangChain entende. Essas tools são então passadas para o agente no `chatNode`:

```typescript
// src/services/openrouterService.ts
async generate(systemPrompt: string, userPrompt: string): Promise<string> {
  if (!this.fsAgent) {
    const tools = await getMCPTools()           // busca as tools do servidor MCP
    this.fsAgent = createAgent({
      model: this.llmClient,
      tools,                                    // modelo agora "conhece" as ferramentas
    })
  }

  const messages = [
    new SystemMessage(systemPrompt),
    new HumanMessage(userPrompt),
  ]

  const response = await this.fsAgent.invoke({ messages })
  return String(response.messages.at(-1)?.text ?? '')
}
```

A partir daqui, quando o modelo decide usar uma ferramenta (por exemplo, `read_text_file`), o framework executa automaticamente a chamada ao servidor MCP e injeta o resultado de volta no contexto antes de continuar a geração.

---

## O transporte STDIO

O servidor MCP neste projeto usa o transporte `stdio` — comunicação via entrada e saída padrão entre dois processos:

```
processo Node.js (cliente MCP)
      │
      │ stdin/stdout (texto JSON-RPC)
      │
      ▼
processo npx @modelcontextprotocol/server-filesystem (servidor MCP)
      │
      ▼
sistema de arquivos (restrito ao process.cwd())
```

O servidor MCP roda como um subprocesso separado. Quando o modelo quer usar uma ferramenta, o cliente serializa a chamada em JSON-RPC, envia via stdin, o servidor executa a operação real no filesystem, e retorna o resultado via stdout.

**Por que isso importa para segurança:** o servidor está rodando com as permissões do processo Node.js — o mesmo usuário de sistema que rodou `node src/index.ts`. Não existe sandbox adicional. O escopo `process.cwd()` limita o que o servidor *expõe*, mas dentro desse diretório o servidor tem acesso real.

---

## Por que MCP amplifica o risco de prompt injection

Sem ferramentas, o pior cenário de uma prompt injection é o modelo retornar texto inadequado. Com MCP, o modelo tem agência real:

| Cenário | Sem MCP | Com MCP |
|---|---|---|
| Usuário pergunta pelo `.env` | LLM responde "não posso" ou improvisa um conteúdo | LLM chama `read_text_file('.env')` e retorna o conteúdo real |
| Usuário pede listagem de arquivos | LLM pode inventar uma lista | LLM chama `list_directory('.')` e retorna a estrutura real |
| Injeção bem-sucedida | Resposta indevida em texto | Ação executada no sistema — dado exfiltrado |

O MCP transforma o modelo de **gerador de texto** em **agente com capacidade de agir**. Isso é poderoso para casos legítimos e perigoso se o modelo for manipulado.

---

## Como o escopo limita (parcialmente) o risco

O servidor MCP é inicializado com `process.cwd()`:

```typescript
args: [
  '-y',
  '@modelcontextprotocol/server-filesystem',
  process.cwd()   // ex: /home/user/projeto/03-safeguard-prompt-injection
]
```

Isso significa que o servidor só expõe arquivos dentro desse diretório. O modelo não consegue usar as ferramentas MCP para ler arquivos fora do projeto.

**O que ainda é acessível dentro do escopo:**
- `package.json` — versões, dependências, scripts
- `src/config.ts` — configurações, nomes de modelos
- `.env` — se existir no diretório (chaves de API!)
- `data/users.json` — base de usuários com roles e permissões
- `prompts/system.txt` — o próprio system prompt (útil para ataques mais sofisticados)

Limitar o escopo reduz o raio de dano, mas não elimina o risco. Um atacante que consegue acesso ao `.env` via injeção já causou dano significativo.

---

## MultiServerMCPClient — conectando múltiplos servidores

O `MultiServerMCPClient` foi projetado para conectar o modelo a múltiplos servidores MCP simultaneamente:

```typescript
const mcpClient = new MultiServerMCPClient({
  filesystem: { transport: 'stdio', command: 'npx', args: [...] },
  // outros servidores poderiam ser adicionados aqui:
  // database: { transport: 'stdio', command: 'npx', args: ['mcp-server-postgres', ...] },
  // github: { transport: 'http', url: 'https://...' },
})
```

Cada servidor expõe suas próprias ferramentas, e o modelo pode usar qualquer uma delas no mesmo agente. O `getTools()` agrega todas as ferramentas de todos os servidores conectados em um único array.

**Implicação para segurança:** em sistemas reais com múltiplos servidores MCP, a superfície de ataque se multiplica. Um atacante bem-sucedido em uma injeção pode mover dados entre sistemas, executar queries em bancos de dados, interagir com APIs externas. Isso reforça a necessidade de guardrails antes de qualquer requisição chegar ao agente.

---

## O padrão de lazy initialization do agente

```typescript
// openrouterService.ts
private fsAgent: ReturnType<typeof createAgent> | null = null

async generate(...): Promise<string> {
  if (!this.fsAgent) {
    const tools = await getMCPTools()
    this.fsAgent = createAgent({ model: this.llmClient, tools })
  }
  // reutiliza o agente nas próximas chamadas
}
```

O agente é inicializado na primeira chamada ao `generate` e reutilizado nas seguintes. Isso evita subir um novo processo MCP a cada requisição — o servidor MCP fica vivo enquanto o agente existir.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Configuração do servidor MCP | [src/services/mcpService.ts](../src/services/mcpService.ts) | Transport STDIO, escopo `process.cwd()` |
| Agente com ferramentas MCP | [src/services/openrouterService.ts](../src/services/openrouterService.ts) | `createAgent`, `getMCPTools`, lazy init |
| Nó que usa as ferramentas | [src/graph/nodes/chatNode.ts](../src/graph/nodes/chatNode.ts) | Chama `openRouterService.generate` com sistema de permissões |
| Dependência MCP | [package.json](../package.json) | `@langchain/mcp-adapters` e `@modelcontextprotocol/server-filesystem` |
