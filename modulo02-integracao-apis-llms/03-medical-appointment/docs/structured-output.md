# Structured Output

## O que é

Structured output (saída estruturada) é a técnica de forçar um LLM a retornar dados em um formato predefinido — um objeto JSON com campos e tipos específicos — em vez de texto livre.

Sem isso, você recebe:
```
"Entendido! O usuário quer agendar uma consulta com o Dr. Alicio da Silva 
amanhã às 14h para João Silva."
```

Com structured output, você recebe:
```json
{
  "intent": "schedule",
  "professionalId": 1,
  "professionalName": "Dr. Alicio da Silva",
  "datetime": "2026-06-29T14:00:00.000Z",
  "patientName": "João Silva",
  "reason": null
}
```

A diferença é que o segundo você pode usar diretamente no código. O primeiro precisa de mais parsing de texto — e LLMs às vezes variam o formato, então parsear texto de LLM é frágil.

---

## O Zod schema como contrato

Neste projeto, cada chamada ao LLM tem um schema Zod que define o que você espera receber:

```typescript
// src/prompts/v1/identifyIntent.ts
import { z } from 'zod';

export const IntentSchema = z.object({
  intent: z.enum(['schedule', 'cancel', 'unknown']).describe('The user intent'),
  professionalId: z.number().optional().describe('ID of the medical professional'),
  professionalName: z.string().optional().describe('Name of the medical professional'),
  datetime: z.string().optional().describe('Appointment date and time in ISO format'),
  patientName: z.string().optional().describe('Patient name extracted from question'),
  reason: z.string().optional().describe('Reason for appointment (for scheduling)'),
});
```

Note o `.describe()` em cada campo: essas strings são passadas ao LLM como parte da instrução de formato. O modelo vê o schema como documentação — "o campo `datetime` deve ser a data/hora em formato ISO".

```typescript
// src/prompts/v1/messageGenerator.ts
export const MessageSchema = z.object({
  message: z.string().min(10).describe('Clear, friendly message for the user')
});
```

O `MessageSchema` é mais simples: força que a resposta tenha ao menos um campo `message` com no mínimo 10 caracteres. Sem isso, o LLM poderia retornar `{}`, `""`, ou qualquer coisa.

---

## Como o generateStructured funciona

A chamada ao LLM com structured output é encapsulada no `OpenRouterService`:

```typescript
// src/services/openRouterService.ts
import { createAgent, HumanMessage, providerStrategy, SystemMessage } from "langchain";

async generateStructured<T>(
    systemPrompt: string,
    userPrompt: string,
    schema: z.ZodSchema<T>
) {
    try {
        const agent = createAgent({
            model: this.llmClient,     // ChatOpenAI apontado para OpenRouter
            tools: [],
            responseFormat: providerStrategy(schema)  // aqui a mágica
        })
        const messages = [
            new SystemMessage(systemPrompt),
            new HumanMessage(userPrompt)
        ]
        const data = await agent.invoke({ messages })
        return {
            success: true,
            data: data.structuredResponse,  // já tipado como T
        }
    } catch (error) {
        return { success: true, error: ... }
    }
}
```

**`providerStrategy(schema)`** instrui o agente a usar a funcionalidade de "response format" do provedor (OpenRouter/OpenAI) para garantir que o output seja o JSON definido pelo schema Zod. O modelo recebe o schema como instrução e a API valida que o output está conforme antes de retornar.

**`data.structuredResponse`** é o objeto já parseado e tipado — você recebe `T` (o tipo inferido do schema), não uma string que precisa de `JSON.parse`.

---

## Por que `createAgent` em vez de `llm.withStructuredOutput()`

O LangChain tem uma API mais direta para structured output:
```typescript
// forma mais simples, vista no 04-song-highlights:
const structured = llm.withStructuredOutput(schema)
const result = await structured.invoke([...messages])
```

Este projeto usa `createAgent` com `providerStrategy`, que é uma abordagem diferente. O `providerStrategy` delega a responsabilidade de garantir o formato ao próprio provedor (OpenRouter/OpenAI) via "response format" na API, em vez de usar function calling ou tool use do LangChain. O resultado é o mesmo — JSON estruturado — mas o mecanismo interno difere.

O `04-song-highlights` usa `withStructuredOutput`. Ambas as abordagens são válidas; este projeto mostra uma alternativa para quando você precisa de controle mais explícito sobre o agent loop.

---

## Como o nó usa o resultado

```typescript
// src/graph/nodes/identifyIntentNode.ts
const result = await llmClient.generateStructured(
    systemPrompt,
    userPrompt,
    IntentSchema,   // contrato: o que esperamos de volta
)

if (!result.success) {
    return { intent: 'unknown', error: result.error }
}

const intentData = result.data!  // IntentSchema inferido: { intent, professionalId, ... }
return { ...intentData }         // espalha direto no estado do grafo
```

`result.data` já é do tipo `z.infer<typeof IntentSchema>`. TypeScript sabe que `result.data.professionalId` é `number | undefined`, não `any`. Isso fecha o ciclo: o schema define o contrato, o LLM preenche, o TypeScript garante que o código downstream usa corretamente.

---

## O que acontece quando o LLM não respeita o schema

O `providerStrategy` usa a API do provedor para forçar o formato — em geral, a API retorna erro se o modelo não conseguir produzir o JSON correto. O `try/catch` do `generateStructured` captura isso e retorna `{ success: true, error: <mensagem> }`.

O nó então decide o que fazer:
- `identifyIntentNode`: retorna `intent: 'unknown'` → grafo vai para `message` com fallback
- `messageGeneratorNode`: retorna `AIMessage("Desculpe, errei!")` como fallback

Falhas do LLM viram estados de erro no grafo, não exceções que derrubam o servidor.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| IntentSchema (Zod) | `src/prompts/v1/identifyIntent.ts` | `z.enum`, `z.number().optional()`, `.describe()` |
| MessageSchema (Zod) | `src/prompts/v1/messageGenerator.ts` | `z.string().min(10)` como validação mínima |
| generateStructured | `src/services/openRouterService.ts` | `createAgent`, `providerStrategy(schema)`, `data.structuredResponse` |
| Uso do resultado | `src/graph/nodes/identifyIntentNode.ts` | `result.data!` tipado, spread no estado |
| Tratamento de falha | `src/graph/nodes/messageGeneratorNode.ts` | Fallback quando `result.error` está presente |
