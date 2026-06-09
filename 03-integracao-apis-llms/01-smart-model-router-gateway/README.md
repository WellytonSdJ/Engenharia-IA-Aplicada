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

Crie um arquivo `.env` na raiz do projeto com a chave da API:

```
OPENROUTER_API_KEY=sua-chave-aqui
```

### Parâmetros principais (`src/config.ts`)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `port` | `3000` | Porta do servidor |
| `models` | `arcee-ai/trinity-large-preview:free`, `nvidia/nemotron-3-ultra-550b-a55b:free` | Lista de modelos candidatos |
| `temperature` | `0.2` | Criatividade das respostas |
| `maxTokens` | `100` | Limite de tokens por resposta |
| `systemPrompt` | `Voce é um assistente inteligente...` | Prompt de sistema |
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

# Executar em modo watch
npm run test:dev
```
