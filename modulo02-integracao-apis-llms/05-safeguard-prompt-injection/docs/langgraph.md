# LangGraph neste projeto

> Este documento foca no grafo específico do `03-safeguard-prompt-injection`. Se você ainda não leu sobre LangGraph em geral, leia primeiro o [langgraph.md do projeto 02](../../02-song-highlights/docs/langgraph.md) — ele cobre os fundamentos de StateGraph, nós, edges e checkpointer. Aqui o foco é no que é diferente: um grafo orientado a segurança, sem checkpointer, com roteamento baseado em resultado de validação.

---

## O grafo deste projeto vs. o do projeto 02

O projeto 02 (song-highlights) tem um grafo de **fluxo conversacional**:

```
START → chat → [savePreferences?] → [summarize?] → END
```

Cada nó adiciona valor à conversa: extrai dados, persiste, comprime histórico.

Este projeto tem um grafo de **controle de acesso**:

```
START → guardrails_check → [chat]    → END
                        └─ [blocked] → END
```

A diferença de propósito é fundamental:
- No projeto 02, todos os caminhos são "positivos" — qualquer rota agrega valor
- Neste projeto, existe um caminho "negativo" (`blocked`) que é o resultado correto para inputs maliciosos

---

## O estado: SafeguardStateAnnotation

```typescript
// src/graph/state.ts
export const SafeguardStateAnnotation = z.object({
  messages: withLangGraph(
    z.custom<BaseMessage[]>(),
    MessagesZodMeta
  ),
  user: z.custom<User>(),
  guardrailCheck: z.custom<GuardrailResult | null>().nullable().default(null),
  guardrailsEnabled: z.boolean(),
})
```

Quatro campos, cada um com um papel claro:

| Campo | Tipo | Quem escreve | Quem lê |
|---|---|---|---|
| `messages` | `BaseMessage[]` | `chatNode`, `blockedNode` | `guardrailsCheckNode`, `chatNode` |
| `user` | `User` | `index.ts` (no invoke) | Todos os nós |
| `guardrailCheck` | `GuardrailResult \| null` | `guardrailsCheckNode` | `edgeConditions`, `blockedNode` |
| `guardrailsEnabled` | `boolean` | `index.ts` (no invoke) | `guardrailsCheckNode`, `edgeConditions` |

**`guardrailCheck`** é o campo central: é o "bilhete" que o `guardrailsCheckNode` deixa no estado para a função de roteamento ler. O resultado do modelo safeguard fica aqui:

```typescript
type GuardrailResult = {
  safe: boolean
  reason?: string
  score?: number
  analysis?: string
}
```

---

## Montagem do grafo

```typescript
// src/graph/graph.ts
export function buildChatGraph() {
  const service = new OpenRouterService()

  const workflow = new StateGraph({
    stateSchema: SafeguardStateAnnotation
  })
    .addNode('guardrails_check', createGuardrailsCheckNode(service))
    .addNode('chat', createChatNode(service))
    .addNode('blocked', blockedNode)

    .addEdge(START, 'guardrails_check')

    .addConditionalEdges(
      'guardrails_check',
      (state: GraphState) => routeAfterGuardrails(state),
      {
        chat: 'chat',
        blocked: 'blocked',
      }
    )

    .addEdge('chat', END)
    .addEdge('blocked', END)

  return workflow.compile()
}
```

Sem checkpointer. Sem store. Este grafo não persiste estado entre invocações — cada `invoke` é independente. O projeto é single-shot: recebe uma mensagem, processa, retorna.

---

## O roteamento condicional de segurança

```typescript
// src/graph/nodes/edgeConditions.ts
export function routeAfterGuardrails(state: GraphState): 'chat' | 'blocked' {
  // caso 1: modo --unsafe — pula guardrail completamente
  if (!state.guardrailsEnabled) {
    return 'chat'
  }

  // caso 2: guardrail habilitado, analisa resultado
  const check = state.guardrailCheck
  if (!check || check.safe) {
    return 'chat'    // sem resultado (fallback) ou safe → prossegue
  }

  return 'blocked'   // unsafe detectado → bloqueia
}
```

Três casos possíveis:

1. **`guardrailsEnabled === false`** (flag `--unsafe`): vai direto para `chat`. O `guardrailsCheckNode` ainda executa, mas retorna `{ safe: true }` sem chamar o safeguard model.

2. **`check.safe === true`** (ou `check === null` como fallback): input considerado legítimo, vai para `chat`.

3. **`check.safe === false`**: prompt injection detectado, vai para `blocked`.

**Nota sobre o fallback:** se o `guardrailsCheckNode` falhar (exceção, timeout, etc.), o código captura o erro e retorna `{ safe: false }` — bloqueio por padrão. Isso é o princípio de "fail closed": na dúvida, nega acesso.

```typescript
// guardrailsCheckNode.ts — fail closed
} catch (error) {
  console.error('Guardrails check failed:', error)
  return {
    guardrailCheck: {
      reason: 'Guardrails service unavailable - request blocked for safety',
      safe: false,   // falhou → bloqueia
    }
  }
}
```

---

## O nó guardrails_check

```typescript
// src/graph/nodes/guardrailsCheckNode.ts
export const createGuardrailsCheckNode = (openRouterService: OpenRouterService) => {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      const userPrompt = state.messages.at(-1)?.text!
      const template = PromptTemplate.fromTemplate(prompts.system)
      const systemPrompt = await template.format({
        USER_ROLE: state.user.role,
        USER_NAME: state.user.displayName
      })

      const msg = systemPrompt.concat('\n', userPrompt)   // contexto completo

      const result = await openRouterService.checkGuardRails(
        msg,
        state.guardrailsEnabled,
      )

      return { guardrailCheck: result }     // escreve no estado para a edge condition ler
    } catch (error) {
      return { guardrailCheck: { safe: false, reason: '...' } }
    }
  }
}
```

O nó retorna apenas `guardrailCheck` — um `Partial<GraphState>`. O LangGraph faz merge automático com o estado existente. Os outros campos (`messages`, `user`, `guardrailsEnabled`) continuam intactos.

---

## O nó chat com agente MCP

```typescript
// src/graph/nodes/chatNode.ts
export const createChatNode = (openRouterService: OpenRouterService) => {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    try {
      // fallback para LangGraph Studio (sem state inicial definido)
      if (!state.user) {
        state.user = getUser('ananeri')!
        state.guardrailsEnabled = false
      }

      const userPrompt = state.messages.at(-1)?.text!
      const template = PromptTemplate.fromTemplate(prompts.system)
      const systemPrompt = await template.format({
        USER_ROLE: state.user.role,
        USER_NAME: state.user.displayName
      })

      const response = await openRouterService.generate(systemPrompt, userPrompt)

      return { messages: [new AIMessage(response)] }
    } catch (error) {
      return {
        messages: [new AIMessage('I apologize, but I encountered an error...')],
      }
    }
  }
}
```

O `chatNode` monta o mesmo system prompt que o `guardrailsCheckNode` — mas desta vez para passar ao modelo executor com ferramentas MCP. A diferença é o que acontece depois: `openRouterService.generate` usa o `fsAgent` (que tem as tools MCP), enquanto `checkGuardRails` usa o `safeGuardModel` (sem tools).

---

## O nó blocked: resposta sem LLM

```typescript
// src/graph/nodes/blockedNode.ts
export async function blockedNode(state: GraphState): Promise<Partial<GraphState>> {
  const guardRailCheck = state.guardrailCheck!
  const analysis = guardRailCheck.analysis
    ? `**Analysis:** ${guardRailCheck.analysis}`
    : ''

  const permissions = state.user.permissions?.join(', ') ?? 'None'
  const template = PromptTemplate.fromTemplate(prompts.blocked)
  const blockedMessage = await template.format({
    REASON: guardRailCheck.reason ?? 'Security check failed',
    ANALYSIS: analysis,
    USER_ROLE: state.user.role,
    PERMISSIONS: permissions
  })

  return { messages: [new AIMessage(blockedMessage)] }
}
```

Este nó não é uma factory function — não recebe serviços via injeção de dependência. Ele não precisa: não chama nenhum LLM. Apenas monta uma mensagem a partir do template e do estado atual.

É também o único nó que é `export async function` diretamente em vez de `export const create... = (deps) => async (state) => ...`. Como não tem dependências externas, o padrão de factory não é necessário.

---

## Diferenças em relação ao projeto 02

| Aspecto | Projeto 02 (song-highlights) | Projeto 03 (safeguard) |
|---|---|---|
| **Propósito do grafo** | Fluxo conversacional com memória | Controle de acesso e segurança |
| **Nós** | `chat`, `savePreferences`, `summarize` | `guardrails_check`, `chat`, `blocked` |
| **Checkpointer** | Sim (PostgreSQL) | Não |
| **Persistência** | SQLite + PostgreSQL | Nenhuma |
| **Número de invocações** | Múltiplas (chat contínuo) | Single-shot (uma mensagem) |
| **Roteamento** | Baseado em dados extraídos pelo LLM | Baseado em resultado de validação de segurança |
| **Caminho "negativo"** | Não existe | `blocked` (resultado correto para ataques) |
| **Ferramentas** | Nenhuma | MCP filesystem |

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Definição do grafo | [src/graph/graph.ts](../src/graph/graph.ts) | Montagem dos nós e edges condicionais |
| Estado de segurança | [src/graph/state.ts](../src/graph/state.ts) | `SafeguardStateAnnotation` com `guardrailCheck` |
| Roteamento condicional | [src/graph/nodes/edgeConditions.ts](../src/graph/nodes/edgeConditions.ts) | Os três casos de roteamento |
| Nó de verificação | [src/graph/nodes/guardrailsCheckNode.ts](../src/graph/nodes/guardrailsCheckNode.ts) | Fail closed no catch |
| Nó de chat com MCP | [src/graph/nodes/chatNode.ts](../src/graph/nodes/chatNode.ts) | Fallback para Studio, `generate` com agente |
| Nó de bloqueio | [src/graph/nodes/blockedNode.ts](../src/graph/nodes/blockedNode.ts) | Sem LLM, factory direta |
| Entry point do grafo | [src/graph/factory.ts](../src/graph/factory.ts) | `buildGraph` e `graph` exportado para Studio |
