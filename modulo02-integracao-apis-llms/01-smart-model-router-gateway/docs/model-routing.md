# Roteamento de Modelos (Model Routing)

## O que é

Quando você constrói algo com um único LLM fixo (ex: sempre chamar `gpt-4o`), você fica
preso às condições daquele modelo específico: preço, velocidade de resposta e disponibilidade
são as que forem naquele momento. Model routing é a estratégia de **não fixar o modelo**, e
sim fixar um **critério de escolha** — deixando que a decisão de qual modelo usar seja
resolvida em tempo de requisição, com base em uma lista de candidatos.

A analogia mais próxima é um comparador de preços de passagens aéreas: você não escolhe a
companhia aérea de antemão, você diz "quero o voo mais barato" ou "quero o mais rápido", e o
sistema resolve isso a cada busca, entre as opções disponíveis naquele instante.

## Como está sendo usado neste projeto

O roteamento é configurado em `src/config.ts`, com dois elementos principais: a lista de
modelos candidatos e o critério de ordenação:

```typescript
// src/config.ts
export const config: ModelConfig = {
  // ...
  models: [
    "arcee-ai/trinity-large-preview:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
  ],
  // ...
  provider: {
    sort: {
      by: "throughput",
      // by: 'latency',
      // by: 'price',
      partition: "none",
    },
  },
};
```

Esses dois campos (`models` e `provider.sort.by`) são passados diretamente para a API do
OpenRouter dentro de `OpenRouterService.generate`:

```typescript
// src/openrouterService.ts
const response = await this.client.chat.send({
  models: this.config.models,
  messages: [
    { role: "system", content: this.config.systemPrompt },
    { role: "user", content: prompt },
  ],
  stream: false,
  temperature: this.config.temperature,
  maxTokens: this.config.maxTokens,
  provider: this.config.provider as ChatGenerationParams["provider"],
});
```

O OpenRouter recebe a lista `models` e o critério `provider.sort.by`, avalia os candidatos
disponíveis naquele momento e direciona a chamada para o que melhor satisfaz o critério. A
resposta inclui `response.model` — o ID exato do modelo que foi realmente usado — o que
permite auditar, a cada chamada, qual candidato "venceu".

## Os três critérios disponíveis

| `sort.by` | Comportamento |
| --- | --- |
| `throughput` | Seleciona o modelo com maior taxa de tokens por segundo (padrão neste projeto) |
| `price` | Seleciona o modelo mais barato por token |
| `latency` | Seleciona o modelo com menor tempo de resposta |

## Por que isso importa (e onde tem limite)

A vantagem central é desacoplar a aplicação de um provedor/modelo específico: trocar de
critério é uma mudança de configuração, não de código. Isso é útil em produtos reais onde
custo e performance de modelos LLM mudam com frequência — um modelo que hoje é o mais barato
pode não ser amanhã.

O limite é que o roteamento aqui é **apenas por métricas de infraestrutura** (preço,
throughput, latência) — ele não avalia qualidade de resposta, capacidade de raciocínio, ou
adequação do modelo à tarefa. Um modelo mais barato ou mais rápido pode ser pior na resposta.
Esse tipo de escolha (qual modelo é *melhor* para a tarefa, não só mais barato/rápido) fica
fora do escopo deste projeto.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Lista de candidatos e critério | `src/config.ts` | Campos `models` e `provider.sort.by` |
| Envio do critério para a API | `src/openrouterService.ts` | Parâmetro `provider` em `client.chat.send` |
| Auditoria do modelo escolhido | `src/openrouterService.ts` | Campo `response.model` retornado ao cliente |
| Verificação do roteamento | `tests/router.e2e.test.ts` | Testes que trocam `sort.by` e checam `body.model` |
