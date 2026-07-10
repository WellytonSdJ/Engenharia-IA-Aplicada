# Por onde começar

Este é o **primeiro projeto do módulo 02**. Se no módulo 01 você treinou redes neurais no
browser com TensorFlow.js, aqui a virada de chave é outra: não estamos mais treinando um
modelo — estamos **consumindo** modelos de linguagem já prontos (LLMs) através de uma API,
e construindo um gateway HTTP que decide, a cada requisição, qual modelo usar.

## O que estamos construindo e por quê

O projeto expõe uma única rota HTTP (`POST /chat`) que recebe uma pergunta e devolve a
resposta de um LLM. A parte interessante não é a rota em si — é o que acontece entre a
requisição chegar e a resposta sair: em vez de fixar "sempre uso o modelo X", o gateway
entrega ao **OpenRouter** uma lista de modelos candidatos e um critério de seleção
(preço, throughput ou latência), e deixa o OpenRouter escolher o melhor candidato daquela
lista no momento da chamada.

Isso resolve um problema real de quem constrói produtos com LLM: modelos diferentes têm
preço, velocidade e disponibilidade diferentes, e essas condições mudam com o tempo. Em vez
de hardcodar um modelo fixo no código, você hardcoda **critérios** — e quem escolhe o
modelo exato é a camada de roteamento.

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [model-routing.md](./model-routing.md) | O conceito central do projeto — por que rotear modelos em vez de fixar um só |
| 2 | [openrouter-sdk.md](./openrouter-sdk.md) | Como o SDK do OpenRouter unifica o acesso a múltiplos provedores de LLM |
| 3 | [fastify.md](./fastify.md) | O framework HTTP usado para expor a rota `/chat` com validação de schema |
| 4 | [testes-e2e-injecao-dependencia.md](./testes-e2e-injecao-dependencia.md) | Como os testes trocam a configuração de roteamento sem tocar no código de produção |
| 5 | [glossario.md](./glossario.md) | Referência rápida dos termos novos |

## Mapa do código

| Arquivo | O que faz |
| --- | --- |
| `src/config.ts` | Objeto `config` central: API key, modelos candidatos, temperatura, `maxTokens`, critério de roteamento (`provider.sort.by`) |
| `src/openrouterService.ts` | `OpenRouterService` — wrapper fino sobre o `@openrouter/sdk`; envia o prompt e devolve `{ model, content }` |
| `src/server.ts` | `createServer(routerService)` — monta o app Fastify com a rota `POST /chat` e o schema de validação do body |
| `src/index.ts` | Entry point — instancia o serviço com a config real, sobe o servidor na porta 3000 e dispara uma requisição de smoke test |
| `tests/router.e2e.test.ts` | Testes E2E com `node:test` — sobem o servidor com configs alternativas e verificam qual modelo foi escolhido |

## O fluxo em uma linha

```
Cliente → POST /chat → Fastify valida o body → OpenRouterService.generate(prompt)
   → OpenRouter API escolhe o melhor modelo da lista pelo critério configurado
   → { model, content } de volta ao cliente
```

## Como rodar e ver o que importa

```bash
npm install
cp .env.example .env   # preencher OPENROUTER_API_KEY
npm run dev
```

Ao subir, o `index.ts` já dispara uma requisição de teste no console — observe o campo
`model` na resposta: é o ID exato do modelo que o OpenRouter escolheu entre os candidatos
configurados em `config.models`. Depois, rode os testes e compare o modelo retornado quando
o critério muda de `price` para `throughput`:

```bash
npm test
```
