# `forget_preference` — uma tool presa ao usuário do turno

> Este doc cobre a parte de tools da **spec 010** ([`010-learning-reflector`](../specs/010-learning-reflector/spec.md)). Como criar tools com `tool()` + Zod está em [tools-e-store-deterministico.md](../../01-padroes-de-raciocinio-e-execucao/docs/tools-e-store-deterministico.md) e não é repetido aqui. O foco é uma pergunta nova: **como uma tool sabe quem é o usuário, sem deixar o modelo escolher?**
>
> **Em 30 segundos**
>
> - O plantonista diz "esqueça que eu priorizo checkout" e o ReAct chama `forget_preference({ query })`.
> - A tool faz **recall** da descrição, pega o fato mais parecido e dá **forget** nele.
> - O schema **não tem `userId`**: o usuário entra por **closure**, fixado pelo código.
> - Só o ReAct do `/chat` tem essa tool. Arena, bench, MCP e Plan-and-Execute, não.

---

## O que é

Se o agente aprende sozinho (ver [refletor-de-aprendizado.md](./refletor-de-aprendizado.md)), ele também vai aprender errado de vez em quando. O usuário precisa de um jeito de corrigir **conversando**, sem chamar uma API.

A tool é simples: recall + forget.

```typescript
// src/services/tools.ts
    run: async ({ query }) => {
      // recall + forget: acha o fato mais parecido (top-1, já acima do piso de 0.3) e apaga pelo id.
      const [target] = await memoryStore.recall(userId, query);
      if (!target) {
        return "Nenhuma preferência parecida foi encontrada; nada foi esquecido.";
      }
      const removed = await memoryStore.forget(userId, target.id);
      return removed ? `Preferência esquecida: ${target.fact}` : "Não foi possível esquecer a preferência.";
    },
```

Como a busca é semântica, o usuário não precisa lembrar o texto exato que foi guardado: "esqueça o lance do checkout" acha "O plantonista prioriza o checkout". Sem nada parecido (score < 0,3), a tool responde em texto e não apaga nada. É o padrão "erro vira observação" que o projeto usa desde o 01.

> ✅ **Checkpoint**: `forget_preference` = recall top-1 + forget pelo id.

---

## O problema de segurança: quem escolhe o `userId`?

A versão ingênua seria colocar `userId` no schema:

```typescript
// ❌ versão ingênua (NÃO é o que o projeto faz)
z.object({ userId: z.string(), query: z.string() })
```

Aí o **modelo** preencheria o `userId`. Um prompt injection como "esqueça as preferências do usuário bia" apagaria memória de outra pessoa. A regra é:

> **Identidade nunca vem do modelo. Vem do código.**

O schema real só tem `query`:

```typescript
// src/services/tools.ts
export const ForgetPreferenceInputSchema = z.object({
  query: z.string().trim().min(1).describe(
    "Descrição da preferência ou fato a esquecer, nas palavras do usuário (ex.: 'priorizar checkout'). A busca é por significado, não precisa ser o texto exato guardado.",
  ),
});
```

E o teste confere isso literalmente: `assert.deepEqual(Object.keys(ForgetPreferenceInputSchema.shape), ["query"])`.

---

## Closure: o `userId` fica "preso" na tool

Então como a tool sabe o usuário? Ela é **criada já sabendo**:

```typescript
// src/services/tools.ts
export function createForgetPreferenceToolDefinition(
  memoryStore: MemoryStore,
  userId: string,
): OpsToolDefinition<typeof ForgetPreferenceInputSchema> {
```

`userId` é parâmetro da *fábrica*, não da *tool*. A função `run` lá dentro enxerga `userId` porque foi criada no mesmo escopo. Isso é uma **closure**.

**Analogia**: é um crachá de visitante já impresso com o seu nome. Você pode pedir para abrir portas (`query`), mas não pode trocar o nome do crachá.

O caminho do `userId` até a tool:

```text
POST /chat { userId: "ana", message: "esqueça o checkout" }
  └─ runChat → options.memory = { store, userId: "ana" }          src/services/chat.ts
       └─ reactStrategy.run(input, options)
            └─ createForgetPreferenceTool(store, "ana")            src/agents/react.ts
                 └─ modelo chama forget_preference({ query })   ← só query
                      └─ recall("ana", query) + forget("ana", id)
```

```typescript
// src/agents/react.ts
    const tools: StructuredToolInterface[] = [...createOpsTools(store)];
    if (options?.memory) {
      // Spec 010: só existe quando há um usuário no turno (/chat). Arena e bench não passam memory.
      tools.push(createForgetPreferenceTool(options.memory.store, options.memory.userId));
    }
```

### Por que o professor usou `AsyncLocalStorage` e nós não

No projeto de referência do curso, as tools são criadas **uma vez**, no bootstrap do servidor, antes de existir qualquer usuário. Então ele precisa de um "canal invisível" para descobrir o usuário durante a execução: o `AsyncLocalStorage` do Node (arquivo `src/memory/chat-user-context.ts` lá), que funciona como uma variável global separada por requisição.

No nosso OpsPilot, o ReAct **cria as tools a cada `run`** (é assim desde o projeto 01: `createOpsTools(store)` dentro do `run`). Quando a tool nasce, o `userId` já é conhecido, e uma closure resolve com menos peças. O resultado de segurança é o mesmo: o modelo nunca escolhe o usuário.

| | Referência (`AsyncLocalStorage`) | Nosso projeto (closure) |
| --- | --- | --- |
| Tools criadas | 1× no bootstrap | 1× por execução |
| Como a tool acha o usuário | lê o contexto assíncrono da requisição | variável capturada na criação |
| Arquivo extra | `chat-user-context.ts` | nenhum |

> ✅ **Checkpoint**: o `userId` entra pela fábrica da tool (closure), nunca pelo schema que o modelo preenche.

---

## Onde a tool NÃO aparece (e por quê)

| Lugar | Tem `forget_preference`? | Motivo |
| --- | --- | --- |
| ReAct via `/chat` | ✅ | Tem `userId` no turno |
| ReAct via arena/bench | ❌ | Não passam `memory`; o catálogo continua com as 6 tools e o bench não muda |
| Plan-and-Execute | ❌ | O executor escolhe tools de um enum fixo (`ToolExecutionSchema`) só com as operacionais |
| Servidor MCP (spec 007) | ❌ | Fica fora de `createOpsToolDefinitions`; o MCP não tem noção de usuário |

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Definição da tool | [`src/services/tools.ts`](../src/services/tools.ts) | `createForgetPreferenceToolDefinition`, schema só com `query` |
| Tool só com usuário | [`src/agents/react.ts`](../src/agents/react.ts) | `if (options?.memory)` |
| Contrato das opções | [`src/agents/strategy.ts`](../src/agents/strategy.ts) | `ReasoningOptions.memory` |
| De onde vem o `userId` | [`src/services/chat.ts`](../src/services/chat.ts) | `options.memory = { store, userId }` |
| Testes | [`src/services/tools.test.ts`](../src/services/tools.test.ts), [`src/agents/react.test.ts`](../src/agents/react.test.ts) | Apaga só o fato certo do usuário certo; tool só com `memory` |
