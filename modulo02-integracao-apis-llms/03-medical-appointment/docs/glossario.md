# Glossário — Medical Appointment

Termos já cobertos nos projetos anteriores não são repetidos aqui. Consulte:
- [`02-langchain-intro/docs/glossario.md`](../../02-langchain-intro/docs/glossario.md) — LangGraph (StateGraph, nós, edges, reducer, MessagesZodMeta)
- [`01-smart-model-router-gateway/docs/`](../../01-smart-model-router-gateway/docs/) — OpenRouter, Fastify

---

## Padrões arquiteturais

| Termo | Definição |
| --- | --- |
| **Prompt chaining** | Padrão de encadear múltiplas chamadas ao LLM em sequência, onde a saída de uma alimenta a próxima. Cada chamada tem uma responsabilidade única. |
| **Serviço de domínio** | Classe que encapsula regras de negócio de forma determinística, sem chamar LLMs. Exemplo: `AppointmentService`. |
| **Serviço de LLM** | Classe que abstrai as chamadas ao modelo, sem lógica de negócio. Exemplo: `OpenRouterService`. |
| **Factory function** | Padrão onde uma função recebe dependências como parâmetros e retorna outra função. Usado para injetar `llmClient` e `appointmentService` nos nós do grafo. |
| **Dependency injection** | Técnica de passar dependências (serviços, configurações) para um componente em vez de o componente criá-las internamente. Torna o código testável e desacoplado. |

## Structured Output

| Termo | Definição |
| --- | --- |
| **Structured output** | Técnica de forçar um LLM a retornar dados em formato JSON com campos e tipos predefinidos (schema Zod), em vez de texto livre. |
| **IntentSchema** | Schema Zod deste projeto que define o contrato da resposta do LLM na etapa de extração de intenção: `intent`, `professionalId`, `datetime`, `patientName`, `reason`. |
| **MessageSchema** | Schema Zod que define o contrato da resposta do LLM na etapa de geração de mensagem: campo `message` com pelo menos 10 caracteres. |
| **createAgent** | Função do LangChain que cria um agente com modelo, tools e formato de resposta configurados. Usada aqui com `responseFormat: providerStrategy(schema)`. |
| **providerStrategy** | Função do LangChain que instrui o agente a usar o mecanismo nativo do provedor (OpenRouter/OpenAI "response format") para garantir saída estruturada conforme o schema Zod. |
| **structuredResponse** | Campo retornado pelo `createAgent.invoke()` que contém o objeto já parseado e tipado pelo schema Zod. |
| **safeParse** | Método do Zod que valida um valor sem lançar exceção — retorna `{ success: true, data }` ou `{ success: false, error }`. Usado nos nós `schedulerNode` e `cancellerNode` para validar os campos obrigatórios no estado. |

## LangChain / OpenRouter

| Termo | Definição |
| --- | --- |
| **ChatOpenAI** | Classe do `@langchain/openai` que implementa a interface de chat do OpenAI. Pode ser apontada para qualquer API compatível com OpenAI — neste projeto aponta para o OpenRouter via `configuration.baseURL`. |
| **modelKwargs** | Parâmetros extras passados diretamente ao payload da API, fora das opções padrão do LangChain. Aqui usado para passar `models` e `provider` específicos do OpenRouter. |
| **Partial<GraphState>** | Tipo TypeScript onde todos os campos são opcionais. Nós async retornam `Partial<GraphState>` — só os campos que mudaram — e o LangGraph faz o merge com o estado existente automaticamente. |

## Domínio

| Termo | Definição |
| --- | --- |
| **Intent** | A intenção do usuário extraída via LLM: `schedule` (agendar), `cancel` (cancelar) ou `unknown` (desconhecida). |
| **Scenario** | String composta de `intent + "_" + resultado` usada pelo `messageGeneratorNode` para selecionar o tom da resposta: `schedule_success`, `cancel_error`, `unknown_undefined`, etc. |
| **In-memory CRUD** | Implementação de banco de dados usando apenas um array JavaScript em memória. Os dados são perdidos quando o processo encerra — substituto de banco real para fins de demonstração. |
