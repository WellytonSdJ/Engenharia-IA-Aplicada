# Glossário

Referência rápida dos termos deste projeto. Este é o primeiro projeto do módulo 02 — não há
glossário anterior para referenciar.

## Gateway / Roteamento

| Termo | Definição |
| --- | --- |
| **Model routing** | Estratégia de escolher o modelo LLM a usar em tempo de requisição, com base em critérios (preço, throughput, latência), em vez de fixar um modelo único no código |
| **`sort.by`** | Campo de configuração do OpenRouter que define o critério de seleção entre os modelos candidatos: `price`, `throughput` ou `latency` |
| **Modelo candidato** | Cada item da lista `models` passada ao OpenRouter — um dos modelos que pode ser escolhido para responder |
| **Gateway** | Camada intermediária que recebe a requisição do cliente e decide como/para onde encaminhá-la — aqui, decide qual modelo LLM chamar |

## OpenRouter

| Termo | Definição |
| --- | --- |
| **OpenRouter** | Serviço que expõe uma API única na frente de múltiplos provedores de LLM, permitindo trocar de modelo/provedor por configuração |
| **`@openrouter/sdk`** | Cliente oficial em TypeScript para a API do OpenRouter |
| **`client.chat.send`** | Método do SDK que envia mensagens a um ou mais modelos candidatos e retorna a resposta do escolhido |
| **`httpReferer` / `xTitle`** | Metadados de identificação da aplicação exigidos pelo OpenRouter, usados no painel de uso/billing |

## HTTP / Fastify

| Termo | Definição |
| --- | --- |
| **Fastify** | Framework HTTP para Node.js com validação de schema nativa e alta performance de serialização JSON |
| **Schema de rota** | Definição JSON Schema (`type`, `required`, `properties`) que o Fastify usa para validar o body da requisição antes do handler rodar |
| **`app.inject`** | Recurso do Fastify para simular uma requisição HTTP completa em memória, sem subir uma porta real — usado nos testes |

## Testes e arquitetura

| Termo | Definição |
| --- | --- |
| **Teste E2E (end-to-end)** | Teste que cobre o fluxo completo do sistema (requisição → serviço → API externa → resposta), sem mocks |
| **Injeção de dependência** | Passar uma dependência (ex: configuração, serviço) por parâmetro/construtor, em vez de a própria função/classe criá-la internamente — permite trocar o comportamento em testes |
| **`configOverride`** | Parâmetro opcional do construtor de `OpenRouterService` que permite substituir a configuração padrão por uma configuração customizada (usado nos testes) |
| **`node:test`** | Runner de testes nativo do Node.js, usado neste projeto sem dependência de Jest/Vitest/Mocha |
| **Execução nativa de TypeScript** | Recurso do Node.js 22.6+ (`--experimental-strip-types`) que executa arquivos `.ts` diretamente, sem `tsc` ou `ts-node` |
