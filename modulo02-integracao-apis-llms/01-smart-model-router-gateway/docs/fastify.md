# Fastify

## O que é

Fastify é um framework HTTP para Node.js, alternativa ao Express, com dois diferenciais
relevantes para este projeto: **validação de schema nativa** (sem precisar de uma lib externa
como Zod ou Joi na camada HTTP) e alta performance de serialização de JSON.

Se você já usou Express, a diferença mais visível é que, no Fastify, o schema da requisição
faz parte da definição da rota — o framework valida o body *antes* do seu handler rodar, e
rejeita automaticamente requisições inválidas.

## Como está sendo usado neste projeto

O servidor inteiro se resume a uma função que monta o app e registra uma única rota:

```typescript
// src/server.ts
import Fastify from "fastify";
import { OpenRouterService } from "./openrouterService.ts";

export const createServer = (routerService: OpenRouterService) => {
  const app = Fastify({ logger: false });

  app.post('/chat', {
    schema: {
      body: {
        type: 'object',
        required: ['question'],
        properties: {
          question: { type: 'string', minLength: 5 }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { question } = request.body as { question: string };
      const response = await routerService.generate(question);
      return reply.send(response);
    } catch (error) {
      console.error('Error handling /chat request:', error);
      return reply.code(500);
    }
  });

  return app;
};
```

Pontos a observar:

- O `schema.body` usa **JSON Schema puro** (`type`, `required`, `properties`) — não é Zod nem
  nenhuma lib de validação de terceiros. É o formato nativo que o Fastify entende e valida
  internamente antes do handler executar.
- `question` é obrigatório e precisa ter no mínimo 5 caracteres. Uma requisição que viole
  isso nunca chega a executar o `async (request, reply) => {...}` — o Fastify já responde
  `400` sozinho.
- `createServer` recebe o `routerService` **por parâmetro**, em vez de importar um singleton
  fixo. Essa é a mesma técnica de injeção de dependência explorada em
  [testes-e2e-injecao-dependencia.md](./testes-e2e-injecao-dependencia.md).

## Por que schema no Fastify em vez de Zod aqui

Em projetos futuros do módulo (a partir do 03, com `withStructuredOutput`), Zod aparece para
validar a **saída estruturada de um LLM**. Aqui o uso é diferente: é validação da
**entrada HTTP**, e o Fastify já resolve isso nativamente sem dependência extra — por isso
não há Zod neste projeto.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Criação do app e rota | `src/server.ts` | `Fastify({ logger: false })`, `app.post('/chat', ...)` |
| Schema de validação do body | `src/server.ts` | Bloco `schema.body` com `required` e `minLength` |
| Subida do servidor | `src/index.ts` | `app.listen({ port: 3000, host: '0.0.0.0' })` |
| Testes contra o app sem subir porta | `tests/router.e2e.test.ts` | Uso de `app.inject(...)` |
