# Domain Service vs LLM Service

## O que é

Este projeto tem dois tipos de serviço com responsabilidades fundamentalmente diferentes:

| | `AppointmentService` | `OpenRouterService` |
| --- | --- | --- |
| **Tipo** | Serviço de domínio | Serviço de LLM |
| **O que faz** | Lógica de negócio | Chamadas ao modelo |
| **É determinístico?** | Sim — mesma entrada, mesma saída | Não — o LLM pode variar |
| **Tem estado?** | Sim — lista de agendamentos em memória | Não — stateless |
| **Pode falhar como?** | `throw new Error("Horário indisponível")` | Timeout, rate limit, formato inesperado |
| **Testável sem API?** | Sim — puramente em Node.js | Não sem mock |

Separá-los não é só organização — é necessidade arquitetural. Misturar lógica de negócio com chamadas ao LLM cria código que é ao mesmo tempo não-determinístico e difícil de testar.

---

## AppointmentService — o serviço de domínio

```typescript
// src/services/appointmentService.ts
export const professionals = [
    { id: 1, name: 'Dr. Alicio da Silva', specialty: 'Cardiologia' },
    { id: 2, name: 'Dra. Ana Pereira',    specialty: 'Dermatologia' },
    { id: 3, name: 'Dra. Carol Gomes',    specialty: 'Neurologia' },
]

export class AppointmentService {
    checkAvailability(professionalId: number, date: Date): boolean {
        const alreadyBooked = this.getAppointmentsForProfessional(professionalId, date)
        return !alreadyBooked
    }

    bookAppointment(professionalId: number, date: Date, patientName: string, reason: string) {
        if (!this.checkAvailability(professionalId, date)) {
            throw new Error('Horário indisponível para este profissional')
        }
        const newAppointment = { date: date.toISOString(), patientName, reason, professionalId }
        appointments.push(newAppointment)
        return newAppointment
    }

    cancelAppointment(professionalId: number, patientName: string, date: Date) {
        const hasBooked = this.getAppointmentsForProfessional(professionalId, date, patientName)
        if (!hasBooked) {
            throw new Error('Agendamento não encontrado para cancelamento')
        }
        const index = appointments.indexOf(hasBooked)
        appointments.splice(index, 1)
    }
}
```

Características:
- **Estado interno**: `appointments` é um array em memória (substituto de banco de dados)
- **Regras de negócio**: um horário só pode ter um paciente — `checkAvailability` garante isso
- **Erros de negócio**: `throw new Error(...)` com mensagens legíveis que vão parar no estado do grafo
- **Sem LLM**: código puro, testável, previsível

A lista `professionals` é exportada porque o `identifyIntentNode` precisa incluí-la no prompt do LLM para que o modelo saiba quais profissionais existem e possa fazer o matching de nomes.

---

## OpenRouterService — o serviço de LLM

```typescript
// src/services/openRouterService.ts
export class OpenRouterService {
    private llmClient: ChatOpenAI

    constructor(configOverride?: ModelConfig) {
        this.llmClient = new ChatOpenAI({
            apiKey: this.config.apiKey,
            modelName: this.config.models.at(0),
            configuration: { baseURL: 'https://openrouter.ai/api/v1', ... },
            modelKwargs: { models: this.config.models, provider: this.config.provider }
        })
    }

    async generateStructured<T>(systemPrompt, userPrompt, schema: z.ZodSchema<T>) {
        const agent = createAgent({
            model: this.llmClient,
            tools: [],
            responseFormat: providerStrategy(schema)
        })
        const data = await agent.invoke({ messages: [...] })
        return { success: true, data: data.structuredResponse }
    }
}
```

Características:
- **Stateless**: não guarda histórico entre chamadas (diferente do `04-song-highlights` que usa checkpointer)
- **Um método público**: `generateStructured` — toda chamada ao LLM passa por aqui
- **Typed generics**: `<T>` com o schema Zod garante que o retorno é tipado
- **Sem lógica de negócio**: não sabe o que é um agendamento, só sabe chamar o modelo

---

## Como os dois serviços se relacionam no grafo

```
identifyIntentNode → usa OpenRouterService para extrair dados
        ↓
schedulerNode → usa AppointmentService para executar a ação
        ↓
messageGeneratorNode → usa OpenRouterService para gerar a resposta
```

O `AppointmentService` nunca chama o LLM. O `OpenRouterService` nunca acessa a lista de agendamentos. O único ponto de encontro é o **estado do grafo**: o `identifyIntentNode` preenche `professionalId`, `datetime`, `patientName` no estado, e o `schedulerNode` lê esses campos para chamar o service.

---

## O que isso permite

1. **Testar a lógica de negócio sem LLM**: você pode instanciar `AppointmentService` e testar `bookAppointment`, `checkAvailability`, `cancelAppointment` sem nenhuma chave de API

2. **Substituir o LLM sem tocar no domínio**: trocar OpenRouter por Anthropic ou Ollama é uma mudança só em `OpenRouterService` (e `config.ts`) — o `AppointmentService` não precisa saber

3. **Testar o fluxo completo com LLM mockado**: nos testes, você pode injetar um `OpenRouterService` falso que retorna respostas previsíveis, testando o grafo sem depender de API real

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Regras de negócio | `src/services/appointmentService.ts` | `checkAvailability`, `bookAppointment`, `cancelAppointment` com throws |
| Profissionais exportados | `src/services/appointmentService.ts` | `export const professionals` — usado no prompt do identifyIntent |
| LLM client | `src/services/openRouterService.ts` | `ChatOpenAI` apontado para OpenRouter, `generateStructured` genérico |
| Injeção dos dois | `src/graph/graph.ts` | `buildAppointmentGraph(llmClient, appointmentService)` |
| Integração no nó | `src/graph/nodes/identifyIntentNode.ts` | `professionals` no prompt + `llmClient.generateStructured` |
| Execução determinística | `src/graph/nodes/schedulerNode.ts` | `appointmentService.bookAppointment(...)` — sem LLM |
