# Prompt Chaining

## O que é

Prompt chaining é o padrão de encadear múltiplas chamadas ao LLM em sequência, onde a saída de uma chamada alimenta a próxima. Em vez de um único prompt gigante que faz tudo, você divide o trabalho em etapas especializadas.

Analogia: pense numa linha de montagem. Uma estação entende o pedido, outra busca as peças, outra monta, outra embala. Cada uma é boa naquilo que faz. Tentar fazer tudo numa estação só seria caótico.

---

## O problema com um único prompt

Você poderia escrever um único prompt assim:

```
"Dado que o usuário disse '{mensagem}', liste os profissionais disponíveis,
agende a consulta se for possível, e retorne uma mensagem confirmando o agendamento
com todos os detalhes em português."
```

Problemas:
1. O LLM não tem acesso à lista real de profissionais nem aos agendamentos existentes
2. Você não tem como checar disponibilidade sem executar código
3. Mistura linguagem natural com lógica de negócio — e LLMs são ruins em lógica determinística
4. Se qualquer parte falhar, você não sabe qual

---

## Como o prompt chaining resolve

Cada etapa tem uma responsabilidade única:

```
Etapa 1 — Extração (LLM)
  Input:  "Quero marcar com o Dr. Alicio amanhã às 14h para João Silva"
  Output: { intent: "schedule", professionalId: 1, datetime: "2026-06-29T14:00:00Z", patientName: "João Silva" }
  Por que LLM: entender linguagem natural, parsear datas relativas, fazer matching fuzzy de nomes

Etapa 2 — Execução (código)
  Input:  { professionalId: 1, datetime: "...", patientName: "João Silva" }
  Output: { actionSuccess: true, appointmentData: { ... } } ou { actionError: "Horário indisponível" }
  Por que código: checar disponibilidade real, inserir no "banco", garantir consistência

Etapa 3 — Resposta (LLM)
  Input:  { scenario: "schedule_success", details: { professionalName: "Dr. Alicio", datetime: "..." } }
  Output: "Sua consulta com o Dr. Alicio da Silva foi confirmada para amanhã às 14h. Aguardamos sua visita!"
  Por que LLM: gerar texto amigável, personalizado, no idioma correto
```

---

## Como está implementado no projeto

O pipeline é o grafo de estados do LangGraph. O estado (`AppointmentStateAnnotation`) acumula os dados a cada nó:

```typescript
// src/graph/graph.ts
const AppointmentStateAnnotation = z.object({
  messages: withLangGraph(z.custom<BaseMessage[]>(), MessagesZodMeta),

  patientName: z.string().optional(),         // extraído no identifyIntent
  intent: z.enum(['schedule', 'cancel', 'unknown']).optional(),
  professionalId: z.number().optional(),      // extraído no identifyIntent
  professionalName: z.string().optional(),
  datetime: z.string().optional(),
  reason: z.string().optional(),

  actionSuccess: z.boolean().optional(),      // preenchido no scheduler/canceller
  actionError: z.string().optional(),
  appointmentData: z.any().optional(),

  error: z.string().optional(),               // propagação de erros
})
```

Cada campo é `optional()` porque nenhum nó preenche tudo de uma vez. O estado vai sendo completado ao longo do pipeline:

```
Estado inicial:   { messages: [HumanMessage("quero marcar...")] }
Após identifyIntent: + intent, professionalId, datetime, patientName, reason
Após schedule:    + actionSuccess, appointmentData (ou actionError)
Após message:     + messages: [..., AIMessage("Sua consulta foi confirmada...")]
```

---

## O routing: como o grafo decide qual etapa executar

```typescript
// src/graph/graph.ts
.addConditionalEdges(
  'identifyIntent',
  (state: GraphState): string => {
    if (state.error || !state.intent || state.intent === 'unknown') {
      return 'message';   // vai direto para gerar resposta de fallback
    }
    return state.intent   // 'schedule' ou 'cancel'
  },
  { schedule: 'schedule', cancel: 'cancel', message: 'message' }
)
.addEdge('schedule', 'message')   // após executar, sempre gera resposta
.addEdge('cancel', 'message')
```

Se o LLM falhar em extrair a intenção, `state.intent === 'unknown'` e o grafo pula direto para a geração de mensagem (sem tentar executar nada). O nó `message` sabe lidar com o cenário `unknown_error`.

---

## Por que não usar um único nó com toda a lógica

Comparação direta:

```typescript
// ❌ Tudo em um nó
async function singleNode(state) {
  const intent = await llm.call("detecte a intenção: " + state.messages.at(-1).text)
  if (intent.includes("schedule")) {
    // parsear o intent manualmente...
    // checar disponibilidade...
    const response = await llm.call("gere a confirmação...")
    return { messages: [...state.messages, new AIMessage(response)] }
  }
  // ...mais if/else
}

// ✅ Cada nó faz uma coisa
// identifyIntentNode: detecta e estrutura
// schedulerNode: executa com código determinístico
// messageGeneratorNode: gera resposta natural
```

O nó único mistura responsabilidades, é difícil de testar isoladamente e impossível de observar (qual etapa falhou?). Com o grafo, cada nó pode ser testado sozinho e os logs mostram exatamente onde o fluxo está em cada execução.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Estado acumulativo | `src/graph/graph.ts` | Todos os campos `optional()` que vão sendo preenchidos |
| Etapa 1 (LLM) | `src/graph/nodes/identifyIntentNode.ts` | Chama `llmClient.generateStructured` com `IntentSchema` |
| Etapa 2 (código) | `src/graph/nodes/schedulerNode.ts` | Valida com `safeParse`, chama `appointmentService.bookAppointment` |
| Etapa 3 (LLM) | `src/graph/nodes/messageGeneratorNode.ts` | Constrói o cenário (`schedule_success`, `cancel_error`, etc.) e chama LLM |
| Routing do pipeline | `src/graph/graph.ts` | `.addConditionalEdges` após `identifyIntent` |
| Prompts das etapas | `src/prompts/v1/identifyIntent.ts` e `messageGenerator.ts` | Como os templates são construídos como JSON para cada etapa |
