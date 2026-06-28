# Node Factory Pattern

## O que é

Nos projetos anteriores, os nós do LangGraph eram funções simples:

```typescript
// 02-langchain-intro: nó como função direta
export function upperCaseNode(state: GraphState): GraphState { ... }
```

Esse padrão funciona quando o nó não precisa de nenhuma dependência externa. Mas neste projeto, os nós precisam de dois serviços:
- `OpenRouterService` — para chamar o LLM
- `AppointmentService` — para executar a lógica de negócio

Como injetar essas dependências nos nós sem criar acoplamento global?

**Factory function**: uma função que recebe as dependências e **retorna** a função do nó.

```typescript
// 03-medical-appointment: nó como factory
export function createIdentifyIntentNode(llmClient: OpenRouterService) {
    return async (state: GraphState): Promise<Partial<GraphState>> => {
        // usa llmClient aqui
    }
}
```

`createIdentifyIntentNode` não é o nó — é a fábrica que cria o nó. Você chama a fábrica com as dependências e ela retorna a função que o LangGraph vai usar.

---

## Como as dependências chegam até os nós

Tudo começa em `src/graph/graph.ts`, na função `buildAppointmentGraph`:

```typescript
// src/graph/graph.ts
export function buildAppointmentGraph(
    llmClient: OpenRouterService,
    appointmentService: AppointmentService
) {
    const workflow = new StateGraph({ stateSchema: AppointmentStateAnnotation })
        .addNode('identifyIntent', createIdentifyIntentNode(llmClient))
        .addNode('schedule',       createSchedulerNode(appointmentService))
        .addNode('cancel',         createCancellerNode(appointmentService))
        .addNode('message',        createMessageGeneratorNode(llmClient))
        // ...
    return workflow.compile()
}
```

O grafo é construído com as dependências injetadas. `createIdentifyIntentNode(llmClient)` chama a factory e o resultado (a função do nó) é registrado no grafo.

Em `src/index.ts`, quem monta as dependências e constrói o grafo:

```typescript
// src/index.ts (simplificado)
const openRouterService = new OpenRouterService()
const appointmentService = new AppointmentService()
const graph = buildAppointmentGraph(openRouterService, appointmentService)
```

Essa cadeia garante que:
- Os nós não importam os serviços diretamente (sem imports globais)
- As dependências podem ser substituídas nos testes (ex: passar um `OpenRouterService` mockado)
- Cada nó declara explicitamente o que precisa via parâmetros da factory

---

## Validação dentro do nó

Além da injeção de dependências, os nós de ação (scheduler e canceller) também fazem validação do estado antes de executar:

```typescript
// src/graph/nodes/schedulerNode.ts
const ScheduleRequiredFieldsSchema = z.object({
  professionalId: z.number({ required_error: 'Professional ID is required' }),
  datetime: z.string({ required_error: 'Appointment datetime is required' }),
  patientName: z.string({ required_error: 'Patient name is required' }),
});

export function createSchedulerNode(appointmentService: AppointmentService) {
  return async (state: GraphState): Promise<Partial<GraphState>> => {
    const validation = ScheduleRequiredFieldsSchema.safeParse(state)

    if (!validation.success) {
      const errorMessages = validation.error.errors.map(e => e.message).join(', ')
      return {
        actionSuccess: false,
        actionError: errorMessages,
      }
    }

    // só executa se os campos obrigatórios estão presentes
    const appointment = appointmentService.bookAppointment(
      validation.data.professionalId,
      new Date(validation.data.datetime),
      validation.data.patientName,
      state.reason ?? 'general consultation'
    )
    return { actionSuccess: true, appointmentData: appointment }
  }
}
```

`safeParse(state)` valida o objeto de estado completo contra o schema dos campos obrigatórios para aquela ação. Se o LLM não extraiu `professionalId`, `datetime` ou `patientName`, a validação falha e o nó retorna `actionSuccess: false` em vez de deixar o erro explodir.

---

## Por que Partial<GraphState> no retorno

No `02-langchain-intro`, os nós retornavam `GraphState` (o estado completo). Aqui, retornam `Promise<Partial<GraphState>>`:

```typescript
// 02-langchain-intro
export function upperCaseNode(state: GraphState): GraphState {
    return { ...state, output: state.output.toUpperCase() }
}

// 03-medical-appointment
return async (state: GraphState): Promise<Partial<GraphState>> => {
    return { actionSuccess: true, appointmentData: appointment }
    // não precisa retornar o estado completo
}
```

`Partial<GraphState>` significa que o nó só precisa retornar os campos que mudaram. O LangGraph faz o merge com o estado existente automaticamente. Isso é mais seguro que o spread manual `{ ...state, ... }` porque evita sobrescrever campos que o nó não deveria tocar.

Os dois padrões funcionam com LangGraph — este projeto usa o segundo que é mais limpo.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Factory com LLM | `src/graph/nodes/identifyIntentNode.ts` | `createIdentifyIntentNode(llmClient: OpenRouterService)` |
| Factory com service | `src/graph/nodes/schedulerNode.ts` | `createSchedulerNode(appointmentService: AppointmentService)` |
| Montagem do grafo | `src/graph/graph.ts` | `buildAppointmentGraph(llmClient, appointmentService)` — injeção no grafo |
| Composição no entry point | `src/index.ts` | `new OpenRouterService()`, `new AppointmentService()`, `buildAppointmentGraph(...)` |
| Validação no nó | `src/graph/nodes/schedulerNode.ts` | `ScheduleRequiredFieldsSchema.safeParse(state)` antes de executar |
| Partial return | Todos os nós async | `Promise<Partial<GraphState>>` — retorna só os campos que mudaram |
