# Smart Model Router Gateway

Gateway HTTP construído com **Fastify** e **OpenRouter SDK** que expõe uma API REST para consultar modelos LLM. O roteamento entre modelos é automático: o OpenRouter seleciona o melhor candidato da lista com base no critério configurado (preço, throughput ou latência).

## O que o projeto faz

- Recebe perguntas via `POST /chat` e repassa ao modelo LLM selecionado automaticamente
- Mantém uma lista de modelos candidatos; o OpenRouter escolhe o melhor conforme o critério (`sort.by`)
- Valida o corpo da requisição com schema nativo do Fastify antes de chamar a API
- Retorna a resposta e o nome exato do modelo que foi usado

## Arquitetura — Fluxo de requisição

```
Cliente HTTP
    │
    ▼
POST /chat  (Fastify — validação de schema)
    │
    ▼
OpenRouterService.generate(prompt)
    │   ├── models: lista de candidatos
    │   ├── provider.sort.by: critério de seleção
    │   └── systemPrompt + mensagem do usuário
    ▼
OpenRouter API
    │
    ▼
Modelo LLM selecionado (ex: arcee-ai/trinity-large-preview:free)
    │
    ▼
{ model, content }  →  resposta ao cliente
```

### Responsabilidades por arquivo

| Arquivo                    | Responsabilidade                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| `src/config.ts`            | Objeto `config` com API key, porta, modelos candidatos, temperatura, tokens máximos e critério de sort  |
| `src/openrouterService.ts` | Wrapper do `@openrouter/sdk` — envia o prompt, extrai `content` e `model` da resposta                   |
| `src/server.ts`            | Cria o app Fastify com a rota `POST /chat` e schema de validação                                        |
| `src/index.ts`             | Entry point — instancia serviço e servidor, sobe na porta 3000, dispara uma requisição de teste         |
| `tests/router.e2e.test.ts` | Testes E2E com `node:test` — verificam roteamento por `price` e `throughput`                            |

## Como o roteamento funciona

O `OpenRouterService` passa a lista `models` e o campo `provider.sort.by` para a API do OpenRouter. O OpenRouter avalia os modelos disponíveis na lista e retorna a resposta do que melhor satisfaz o critério:

| `sort.by`    | Comportamento                                         |
| ------------ | ----------------------------------------------------- |
| `throughput` | Seleciona o modelo com maior taxa de tokens por segundo (padrão) |
| `price`      | Seleciona o modelo mais barato por token              |
| `latency`    | Seleciona o modelo com menor tempo de resposta        |

A resposta sempre inclui `response.model` — o ID exato do modelo que foi acionado — o que permite auditar qual candidato foi escolhido a cada chamada.

## Estrutura do projeto

```
src/
├── config.ts                  # Configuração centralizada: modelos, sort, temperatura
├── index.ts                   # Inicialização do servidor + requisição de smoke test
├── server.ts                  # Fastify app com rota POST /chat e validação de schema
└── openrouterService.ts       # Cliente OpenRouter: envia prompt, retorna { model, content }
tests/
└── router.e2e.test.ts         # Testes E2E com chamadas reais à API OpenRouter
```

## API

### `POST /chat`

Envia uma pergunta ao modelo LLM selecionado automaticamente.

**Body:**
```json
{
  "question": "O que é rate limiting?"
}
```

> `question` é obrigatório e deve ter no mínimo 5 caracteres. Requisições com body inválido retornam `400` antes de chegar ao serviço.

**Resposta `200`:**
```json
{
  "model": "arcee-ai/trinity-large-preview:free",
  "content": "Rate limiting é uma técnica para controlar o número de requisições..."
}
```

**Resposta `400`:** Body inválido — `question` ausente ou com menos de 5 caracteres.

**Resposta `500`:** Erro de comunicação com a API OpenRouter.

## Requisitos

### Software

| Requisito   | Versão mínima | Observação                                                                          |
| ----------- | ------------- | ----------------------------------------------------------------------------------- |
| **Node.js** | 22.6.0+       | Suporte nativo a TypeScript via `--experimental-strip-types`; recomendado 24.x      |
| **npm**     | 10+           | Incluído no Node.js                                                                 |

> Node.js 22.6+ executa arquivos `.ts` diretamente sem compilação — não é necessário `ts-node` ou `tsc`.

### Contas e APIs

| Serviço        | Obrigatório | Como obter                                                                           |
| -------------- | ----------- | ------------------------------------------------------------------------------------ |
| **OpenRouter** | Sim         | Crie uma conta em [openrouter.ai](https://openrouter.ai/) e gere uma API key         |

### Variáveis de ambiente

Crie `.env` copiando `.env.example`:

```bash
cp .env.example .env
```

| Variável             | Obrigatória | Descrição                        |
| -------------------- | ----------- | -------------------------------- |
| `OPENROUTER_API_KEY` | Sim         | Chave de acesso à API OpenRouter |

## Configuração

Parâmetros em `src/config.ts`:

| Parâmetro          | Valor padrão                          | Descrição                                              |
| ------------------ | ------------------------------------- | ------------------------------------------------------ |
| `port`             | `3000`                                | Porta do servidor Fastify                              |
| `models`           | `arcee-ai/trinity-large-preview:free` | Lista de modelos candidatos para roteamento            |
| `temperature`      | `0.2`                                 | Criatividade das respostas (0 = mais determinístico)   |
| `maxTokens`        | `100`                                 | Limite de tokens por resposta                          |
| `systemPrompt`     | `"Voce é um assistente..."`           | Instrução de sistema enviada a cada requisição         |
| `provider.sort.by` | `throughput`                          | Critério de seleção do modelo: `price`, `throughput` ou `latency` |

Para testar diferentes critérios de roteamento sem alterar `config.ts`, passe um `configOverride` ao instanciar `OpenRouterService` — é o mesmo padrão usado nos testes E2E.

## Execução

```bash
# Instalar dependências
npm install

# Desenvolvimento (com hot-reload via nodemon)
npm run dev

# Produção
node --experimental-strip-types --env-file .env src/index.ts
```

Ao iniciar, o `index.ts` dispara automaticamente uma requisição de teste (smoke test):

```
POST /chat { question: "qual a resposta pro mundo, o universo e tudo mais?" }
```

A resposta aparece no console com o modelo selecionado e o conteúdo gerado.

## Testes

Os testes E2E usam o runner nativo do Node.js (`node:test`) e fazem chamadas reais à API do OpenRouter — portanto exigem `OPENROUTER_API_KEY` válida no ambiente.

```bash
npm test           # executa todos os testes E2E
npm run test:dev   # modo watch
```

Os testes verificam:

- Roteamento por `price` — confirma que o modelo mais barato da lista é selecionado
- Roteamento por `throughput` — confirma que o modelo com maior vazão é selecionado

O `createServer` aceita uma instância de `OpenRouterService` por injeção, o que permite substituir a configuração nos testes sem alterar o código de produção:

```typescript
const customConfig = { ...config, provider: { sort: { by: 'price' } } }
const routerService = new OpenRouterService(customConfig)
const app = createServer(routerService)
```
