# OpenRouter SDK

## O que é

OpenRouter é um serviço que expõe uma **API única** na frente de dezenas de provedores de
LLM (OpenAI, Anthropic, modelos open-source hospedados, etc.). Em vez de integrar
diretamente com o SDK de cada provedor — cada um com sua própria autenticação, formato de
request e formato de resposta — a aplicação integra apenas com o OpenRouter, e ele repassa a
chamada para o modelo/provedor de fato escolhido.

A analogia é a de um "roaming" de operadoras de celular: seu aparelho fala com uma torre só,
mas por trás pode estar em qualquer operadora parceira — você não precisa negociar com cada
uma individualmente.

O `@openrouter/sdk` é o cliente oficial em TypeScript/JavaScript para essa API.

## Como está sendo usado neste projeto

O cliente é instanciado uma única vez, dentro do serviço que encapsula toda a comunicação
com a API:

```typescript
// src/openrouterService.ts
import { OpenRouter } from "@openrouter/sdk";
import { config, type ModelConfig } from "./config.ts";
import { type ChatGenerationParams } from "@openrouter/sdk/models";

export class OpenRouterService {
  private client: OpenRouter;
  private config: ModelConfig;

  constructor(configOverride?: ModelConfig) {
    this.config = configOverride ?? config;

    this.client = new OpenRouter({
      apiKey: config.apiKey,
      httpReferer: config.httpReferer,
      xTitle: config.xTitle,
    });
  }

  async generate(prompt: string): Promise<LLMResponse> {
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

    const raw = response.choices.at(0)?.message.content ?? "";
    const content = typeof raw === "string" ? raw : JSON.stringify(raw);
    return { model: response.model, content };
  }
}
```

Pontos a observar:

- `client.chat.send` aceita **uma lista de modelos** (`models`), não um único modelo — é essa
  lista que alimenta o [roteamento](./model-routing.md).
- `httpReferer` e `xTitle` são metadados de identificação da aplicação exigidos pelo
  OpenRouter para exibir no painel de uso/billing — não afetam a geração da resposta.
- A resposta segue o formato `choices[].message.content`, o mesmo shape usado pela API da
  OpenAI — o OpenRouter normaliza a resposta de qualquer provedor para esse formato comum.
- `response.model` devolve o ID exato do modelo que respondeu, essencial para auditar o
  roteamento.

## Por que um wrapper (`OpenRouterService`) em vez de usar o SDK direto na rota

O `OpenRouterService` isola o Fastify de qualquer detalhe do SDK: a rota HTTP em
`server.ts` não sabe nada sobre `@openrouter/sdk`, só chama `routerService.generate(question)`
e recebe `{ model, content }`. Essa separação é o que permite, por exemplo, os testes
instanciarem o serviço com uma configuração diferente (ver
[testes-e2e-injecao-dependencia.md](./testes-e2e-injecao-dependencia.md)) sem tocar no código
da rota.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Instanciação do cliente | `src/openrouterService.ts` | Construtor da classe `OpenRouterService` |
| Chamada ao modelo | `src/openrouterService.ts` | Método `generate`, chamada `client.chat.send` |
| Configuração de credenciais | `src/config.ts` | Campos `apiKey`, `httpReferer`, `xTitle` |
| Dependência declarada | `package.json` | `"@openrouter/sdk": "^0.5.1"` |
