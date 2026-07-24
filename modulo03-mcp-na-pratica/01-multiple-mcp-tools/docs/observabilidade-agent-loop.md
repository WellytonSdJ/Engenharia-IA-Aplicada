# Observabilidade do Loop do Agente (Callbacks)

## O problema que este documento resolve

Depois de ler [agente-autonomo-vs-orquestracao-explicita.md](./agente-autonomo-vs-orquestracao-explicita.md), fica uma pergunta óbvia: se o agente decide sozinho quantas tools chamar e em qual ordem, **como eu vejo o que ele está fazendo de dentro para fora?** Em um grafo explícito, cada nó tem um `console.log` próprio e você sabe exatamente em que etapa está. Dentro do loop interno do `createAgent`, isso não existe por padrão — é uma "caixa preta" que só devolve o resultado final.

A resposta deste projeto: **callbacks do LangChain**, passados na chamada de `agent.invoke`.

## Como está implementado

```typescript
// src/services/openRouterService.ts
const data = await agent.invoke(
    { messages },
    {
        callbacks: [{
            handleChatModelStart(_llm, promptMessages) {
                const lastMsg = promptMessages.at(-1)?.at(-1);
                console.log(`\n🧠 LLM thinking...`);
                console.log(` (last message: "${lastMsg?.content?.toString()}")`);
            },
            handleLLMEnd(output) {
                const msg = (output.generations?.at(0)?.at(0) as ChatGeneration)?.message as AIMessage;
                const toolCalls = msg?.tool_calls;
                if (toolCalls?.length) {
                    console.log(`🎯 Decided to call: ${toolCalls.map((t) => t.name).join(', ')}`);
                }
            },
            handleToolStart(_tool, input, _runId, _parentRunId, _tags, _metadata, runName) {
                console.log(`🔧 Tool called: ${runName} →`, input);
            },
            handleToolEnd(output, _runId, _parentRunId, runName) {
                console.log(`✅ Tool done:   ${runName} →`, output);
            },
        }]
    });
```

Cada callback é um gancho (*hook*) em um momento específico do ciclo de vida do agente:

| Callback | Quando dispara | Para que serve aqui |
| --- | --- | --- |
| `handleChatModelStart` | Antes de cada chamada ao modelo | Mostra qual mensagem o modelo está "vendo" antes de decidir o próximo passo |
| `handleLLMEnd` | Depois que o modelo responde | Mostra se o modelo decidiu chamar uma tool (e qual) ou se já terminou |
| `handleToolStart` | Antes de executar uma tool | Mostra qual tool foi chamada e com quais argumentos |
| `handleToolEnd` | Depois que a tool termina | Mostra o resultado que volta para o contexto do modelo |

## Por que isso é diferente de um `console.log` de nó

Em um grafo com nós explícitos (`identifyIntentNode`, `schedulerNode`, `messageGeneratorNode`), cada `console.log` está dentro de uma função que você escreveu — você sabe de antemão quantas vezes cada log vai aparecer, porque o fluxo é fixo.

Aqui, o número de vezes que `handleToolStart`/`handleToolEnd` disparam depende de quantas tools o modelo decidir chamar naquela execução específica — pode ser 3 chamadas numa rodada e 6 em outra, dependendo de como o modelo interpreta o prompt e os dados de entrada. Os callbacks são a forma de instrumentar um comportamento que **não é fixo em tempo de código**, só em tempo de execução.

```
Execução típica deste projeto (visto no terminal):

🧠 LLM thinking...
🎯 Decided to call: mongodb_delete_many
🔧 Tool called: mongodb_delete_many → { ... }
✅ Tool done:   mongodb_delete_many → { ... }

🧠 LLM thinking...
🎯 Decided to call: csv_to_json
🔧 Tool called: csv_to_json → { csvText: "id,product,price,date..." }
✅ Tool done:   csv_to_json → "[{...}]"

🧠 LLM thinking...
🎯 Decided to call: mongodb_insert_many
...
🧠 LLM thinking...
🎯 Decided to call: mongodb_aggregate
...
🧠 LLM thinking...
🎯 Decided to call: write_file
...
🧠 LLM thinking... (sem tool_calls → resposta final)
```

Cada bloco é uma "volta" do loop interno do agente: pensa, decide, executa, observa o resultado, pensa de novo. Os callbacks tornam essa sequência visível sem precisar instrumentar manualmente cada tool.

## Onde isso se conecta com LangSmith

O projeto também suporta tracing via LangSmith (`LANGSMITH_API_KEY`, `LANGCHAIN_TRACING_V2=true` no `.env`), que já apareceu em projetos anteriores do módulo 02 como visualização opcional de grafos. A diferença é que os callbacks aqui são **impressos direto no terminal**, sem depender de nenhum serviço externo — uma forma mais simples de debugar localmente antes de subir para uma ferramenta de observability completa.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Callbacks no `agent.invoke` | [src/services/openRouterService.ts](../src/services/openRouterService.ts) | `handleChatModelStart`, `handleLLMEnd`, `handleToolStart`, `handleToolEnd` |
| Tipo `ChatGeneration` | [src/services/openRouterService.ts](../src/services/openRouterService.ts) | Usado para tipar `output.generations` dentro de `handleLLMEnd` |
| Onde rodar e ver o log | [src/index.ts](../src/index.ts) → `npm start` | O terminal mostra a sequência completa de pensamentos/tools em tempo real |
