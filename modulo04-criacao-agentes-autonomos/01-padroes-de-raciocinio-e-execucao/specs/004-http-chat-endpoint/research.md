# Research: Endpoint HTTP de Chat

Não restaram itens `NEEDS CLARIFICATION` no Technical Context. As decisões abaixo registram as escolhas técnicas e as alternativas descartadas.

## R1. Como testar o endpoint por HTTP sem nova dependência

- **Decision**: subir o app com `app.listen(0, "127.0.0.1")`, ler a porta de `server.address()` e chamar com o `fetch` global do Node 22; fechar com `server.close()` em `after`.
- **Rationale**: exercita HTTP real (parsing JSON, status, headers) sem adicionar `supertest`, como manda o princípio V. O tráfego fica restrito ao loopback.
- **Alternatives considered**: `supertest`, que adicionaria uma dependência de desenvolvimento só para isso; chamar o handler com `req`/`res` falsos, que não cobriria o `express.json()` nem o middleware de erro.

## R2. Implementação do timeout de 60 s

- **Decision**: helper puro `withTimeout(promise, ms)`, que faz `Promise.race` entre a execução e um timer que rejeita com `ExecutionTimeoutError`; `clearTimeout` em `finally`; `promise.catch(() => {})` anexado para descartar uma rejeição tardia.
- **Rationale**: atende o 504 sem mexer nas estratégias. Limpar o timer evita prender o event loop nos testes. O `.catch` evita *unhandled rejection* quando a estratégia falha depois do timeout.
- **Alternatives considered**: `AbortSignal.timeout` repassado às estratégias, que exigiria mudar `ReasoningOptions` e todas as estratégias (fora de escopo); `req.setTimeout`/`server.timeout`, que fecha o socket sem permitir o corpo 504.

## R3. Onde mora o registro e como aplicar reflection

- **Decision**: `src/agents/index.ts` exporta `strategyRegistry` (`react`, `plan-and-execute`), o tipo `StrategyRegistry` e `resolveStrategy(name, { reflect }, registry)`. Com `reflect`, retorna `withReflection(base)` com as opções padrão (`maxReflections: 2`).
- **Rationale**: um único ponto que mapeia nome → estratégia, reaproveitável por outras bordas. Aplicar reflection na resolução evita duplicar chaves `reflect:*` e mantém o contrato do corpo (`reflect` booleano) ortogonal ao nome.
- **Alternatives considered**: reutilizar `arenaStrategies`, que já mistura chaves `reflect:*` e faria `strategy: "reflect:react"` + `reflect: true` aplicar reflection duas vezes; migrar a arena agora, que ficou fora de escopo pela spec.

## R4. Lookup seguro de nomes

- **Decision**: usar `Object.hasOwn(registry, name)` antes de indexar.
- **Rationale**: `registry["constructor"]` ou `registry["toString"]` retornariam funções do protótipo e causariam 500 em vez de 422.
- **Alternatives considered**: um `Map`, também seguro, mas destoaria do `Record` usado em `arena.ts` e `runArena`.

## R5. JSON malformado e corpo ausente no Express 5

- **Decision**: `express.json()` na rota e um middleware de erro que reconhece `err.type === "entity.parse.failed"` e responde 400 com uma issue sintética `{ path: [], message: "Malformed JSON body." }`. Um corpo ausente (sem `Content-Type: application/json`) deixa `req.body` como `undefined` no Express 5 e falha naturalmente no Zod.
- **Rationale**: mantém um único formato de 400 (sempre com `issues`) para qualquer corpo inválido.
- **Alternatives considered**: deixar o handler padrão do Express, que responde 400 em HTML e quebra o contrato JSON.

## R6. Formato do 500 e logging

- **Decision**: o corpo é fixo, `{ error: "INTERNAL", message: "Internal server error." }`, e o erro original vai para `console.error` no servidor.
- **Rationale**: erros de configuração mencionam nomes de variáveis de ambiente e erros do provedor podem incluir detalhes de requisição; nenhum deve chegar ao cliente (princípio V).
- **Alternatives considered**: ecoar `error.message` para `OpsPilotError`; descartado porque `AgentConfigurationError` revela a configuração.

## R7. Injeção de modelo para o teste de `reflect`

- **Decision**: `ChatDependencies.model?: BaseChatModel` é repassado a `strategy.run(message, { model })`.
- **Rationale**: `withReflection` resolve o crítico com `resolveModel(options.model)`. Sem injeção, o teste cairia em `createModel()` e exigiria `.env`/rede. Em produção `model` fica `undefined` e o comportamento atual se mantém.
- **Alternatives considered**: registrar no teste uma estratégia fake que já devolve eventos `critique`; descartado porque não provaria que `reflect: true` aplica `withReflection` de verdade.
