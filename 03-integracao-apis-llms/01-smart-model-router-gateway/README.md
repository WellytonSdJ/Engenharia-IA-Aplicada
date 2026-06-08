# Smart Model Router Gateway

Gateway HTTP que roteia requisições para modelos LLM via [OpenRouter](https://openrouter.ai/), selecionando automaticamente o melhor modelo com base em critérios configuráveis (preço, throughput ou latência).

## Arquitetura

```
Cliente HTTP
    │
    ▼
POST /chat (Fastify)
    │
    ▼
OpenRouterService
    │
    ▼
OpenRouter API ──► Modelo LLM selecionado
```

## Stack

- **Node.js** (ESM nativo, sem compilação TypeScript)
- **Fastify 5** — servidor HTTP
- **@openrouter/sdk** — cliente OpenRouter
- **TypeScript** — tipagem estática

## Configuração

Copie o arquivo de exemplo e preencha a chave da API:

```bash
cp .env.example .env
```

`.env`:
```
OPENROUTER_API_KEY=sua-chave-aqui
```

### Parâmetros principais (`src/config.ts`)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `port` | `3000` | Porta do servidor |
| `models` | `arcee-ai/trinity-large-preview:free` | Lista de modelos candidatos |
| `temperature` | `0.2` | Criatividade das respostas |
| `maxTokens` | `100` | Limite de tokens por resposta |
| `systemPrompt` | `You are a helpful assistant.` | Prompt de sistema |
| `provider.sort.by` | `throughput` | Critério de roteamento: `price`, `throughput` ou `latency` |

## Execução

```bash
# Instalar dependências
npm install

# Desenvolvimento (com hot-reload)
npm run dev

# Produção
node --env-file .env src/index.ts
```

## API

### `POST /chat`

Envia uma pergunta ao modelo LLM selecionado.

**Body:**
```json
{
  "question": "O que é rate limiting?"
}
```

**Validação:** `question` é obrigatório e deve ter no mínimo 5 caracteres.

**Resposta (`200`):**
```json
{
  "model": "arcee-ai/trinity-large-preview:free",
  "content": "Rate limiting é uma técnica..."
}
```

**Resposta (`400`):** Body inválido (validação do schema Fastify).

**Resposta (`500`):** Erro ao comunicar com a API OpenRouter.

## Testes

Testes E2E utilizam o runner nativo do Node.js (`node:test`). Requerem `OPENROUTER_API_KEY` válida pois fazem chamadas reais à API.

```bash
# Executar testes
npm test

# Desenvolvimento (com hot-reload)
npm run test:dev
```

### Cenários cobertos

| Teste | Critério de roteamento | Modelo esperado |
|---|---|---|
| Modelo mais barato | `price` | `arcee-ai/trinity-large-preview:free` |
| Maior throughput | `throughput` | `nvidia/nemotron-3-nano-30b-a3b:free` |

## Estrutura do projeto

```
src/
├── config.ts             # Configurações e variáveis de ambiente
├── index.ts              # Entry point — inicializa servidor
├── openrouterService.ts  # Client OpenRouter e lógica de geração
└── server.ts             # Definição das rotas Fastify

tests/
└── router.e2e.test.ts    # Testes E2E de roteamento
```

## Como o roteamento funciona

O OpenRouter recebe a lista de `models` e o critério `provider.sort.by`, então seleciona automaticamente o modelo disponível que melhor atende ao critério no momento da requisição. Isso permite failover e otimização dinâmica sem lógica manual de seleção.
