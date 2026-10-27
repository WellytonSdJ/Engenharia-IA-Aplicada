# Por onde começar

Este é o terceiro projeto do módulo 04. O anterior foi [`02-function-calling-e-tool-use`](../../02-function-calling-e-tool-use/docs/00-START-HERE.md), e antes dele veio [`01-padroes-de-raciocinio-e-execucao`](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md). Este projeto começa como uma **cópia** do 02 (mesmo `src/`, specs 001–008) e acrescenta duas specs.

---

## O que estamos construindo e por quê

O 01 respondeu "como o agente pensa?". O 02 respondeu "com que mãos ele age?". Esta unidade pergunta:

> **"O que o agente lembra de você, e como ele aprende sozinho?"**

Até aqui, o OpsPilot tinha amnésia entre conversas. A conversa persistente (spec 008) lembrava *o fio de uma conversa*, mas abriu um `conversationId` novo e ele esqueceu tudo: que você prefere resumos curtos, que o checkout é sempre prioridade...

| Spec | O que acrescenta | Por quê |
| --- | --- | --- |
| [009-semantic-memory](../specs/009-semantic-memory/spec.md) | Embeddings locais (`all-MiniLM-L6-v2`), `SqliteMemoryStore` (`remember`/`recall`/`forget`), `userId` obrigatório no `/chat`, memórias injetadas no prompt, `POST /memories` | Memória de **longo prazo** por usuário, buscada **por significado** |
| [010-learning-reflector](../specs/010-learning-reflector/spec.md) | Refletor pós-turno (`{ hasLearning, fact }`) em segundo plano + tool `forget_preference` | O agente **aprende conversando** e você corrige conversando |

> **Mapa com o material do curso**: as nossas specs 009 e 010 são as specs `008-semantic-memory` e `009-learning-reflector` da referência (a nossa numeração está +1 por causa da `003-fix-replanner-output` do projeto 01). A tabela [arquivo da referência → arquivo nosso](#arquivo-da-referência--arquivo-nosso) mostra onde cada arquivo foi parar.

Ficou de fora de propósito: o `scripts/conversa-longa.sh` da referência. O próprio material registra que ele é gancho da próxima unidade (gerenciamento de contexto).

---

## Trilha de leitura

Leia nesta ordem. Cada doc tem um bloco **"Em 30 segundos"** no topo e **checkpoints** ao longo do texto. Se o tempo estiver curto, leia só os blocos de 30 segundos e os checkpoints.

| # | Documento | Por que ler |
| --- | --- | --- |
| 0 | Trilhas do [01](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md#trilha-de-leitura) e do [02](../../02-function-calling-e-tool-use/docs/00-START-HERE.md) | Reflection, tools, store e `/chat` não são repetidos aqui |
| 1 | [embeddings-e-similaridade.md](./embeddings-e-similaridade.md) | A base de tudo: texto → vetor, cosseno, modelo local, lazy singleton |
| 2 | [memoria-semantica.md](./memoria-semantica.md) | O vector store em SQLite, as 3 operações, os limiares e como a memória entra no `/chat` |
| 3 | [refletor-de-aprendizado.md](./refletor-de-aprendizado.md) | Reflection reaproveitada para aprender + fire-and-forget |
| 4 | [forget-preference.md](./forget-preference.md) | Como uma tool sabe o usuário sem o modelo escolher (closure) |
| 5 | [testando-memoria.md](./testando-memoria.md) | `FakeEmbedder`, teste semântico separado e como testar "não esperou" |
| 6 | [glossario.md](./glossario.md) | Consulta rápida dos termos novos |

---

## Mapa do código

Só o que é **novo ou mudou** nesta unidade, em ordem de leitura. O resto está no mapa do [01](../../01-padroes-de-raciocinio-e-execucao/docs/00-START-HERE.md) e na doc do [02](../../02-function-calling-e-tool-use/docs/README.md).

| Arquivo | Spec | O que faz |
| --- | --- | --- |
| [`src/services/embeddings.ts`](../src/services/embeddings.ts) | 009 | Contrato `Embedder` e o modelo real (lazy singleton, import dinâmico) |
| [`src/services/fake-embedder.ts`](../src/services/fake-embedder.ts) | 009 | Double determinístico para testes |
| [`src/services/errors.ts`](../src/services/errors.ts) | 009 | + `InvalidMemoryInputError` e `EmbeddingError` |
| [`src/models/domain.ts`](../src/models/domain.ts) | 009 | + `RecalledMemorySchema` e `RememberResultSchema` |
| [`src/models/memory-store.ts`](../src/models/memory-store.ts) | 009 | `MemoryStore` + `SqliteMemoryStore`: tabela `memories`, BLOB, `dot`, limiares |
| [`src/services/chat.ts`](../src/services/chat.ts) | 009 / 010 | `userId`, recall, `composeInput`, `recalledMemories`, `options.memory` e disparo do refletor |
| [`src/controllers/chat-controller.ts`](../src/controllers/chat-controller.ts) | 009 | `userId` obrigatório; 400/503 para erros de memória |
| [`src/controllers/memory-controller.ts`](../src/controllers/memory-controller.ts) | 009 | `POST /memories` (201 novo / 200 dedup) |
| [`src/http/server.ts`](../src/http/server.ts) | 009 | Registra a rota `/memories` |
| [`src/agents/system-prompt.ts`](../src/agents/system-prompt.ts) | 009 | `OPSPILOT_SYSTEM_PROMPT` (formato + memórias) e `MEMORY_GUIDANCE` |
| [`src/agents/react.ts`](../src/agents/react.ts) | 009 / 010 | Usa o system prompt; adiciona `forget_preference` quando há `memory` |
| [`src/agents/plan-and-execute.ts`](../src/agents/plan-and-execute.ts) | 009 | Planner recebe `MEMORY_GUIDANCE` |
| [`src/agents/strategy.ts`](../src/agents/strategy.ts) | 010 | `ReasoningOptions.memory` |
| [`src/services/learning-reflector.ts`](../src/services/learning-reflector.ts) | 010 | Prompt, schema, refletor LLM e `scheduleLearning` |
| [`src/services/tools.ts`](../src/services/tools.ts) | 010 | + `forget_preference` (fora do catálogo do MCP) |
| [`src/index.ts`](../src/index.ts) | 010 | Liga o refletor real no servidor |
| [`src/models/memory-store.test.ts`](../src/models/memory-store.test.ts) | 009 | Lógica do store com `FakeEmbedder` |
| [`src/models/memory-store.semantic-test.ts`](../src/models/memory-store.semantic-test.ts) | 009 | Modelo real: recall sem palavras em comum |
| [`src/services/learning-reflector.test.ts`](../src/services/learning-reflector.test.ts) | 010 | Aprende × pontual × segredo; falhas engolidas |
| [`src/services/chat.test.ts`](../src/services/chat.test.ts), [`src/http/server.test.ts`](../src/http/server.test.ts), [`src/services/tools.test.ts`](../src/services/tools.test.ts), [`src/agents/react.test.ts`](../src/agents/react.test.ts) | 009 / 010 | Casos novos de memória, fire-and-forget e tool |

### Arquivo da referência → arquivo nosso

A referência do curso põe tudo em `src/memory/`. Aqui os arquivos mantêm **o mesmo nome**, mas cada um vai para a sua camada MVC (regra II da constituição):

| Referência (`04-memoria-e-reflexao-em-agentes-autonomos`) | Nosso projeto |
| --- | --- |
| `src/memory/embeddings.ts` | `src/services/embeddings.ts` |
| `src/memory/fake-embedder.ts` | `src/services/fake-embedder.ts` |
| `src/memory/memory-store.ts` | `src/models/memory-store.ts` |
| `src/memory/learning-reflector.ts` | `src/services/learning-reflector.ts` |
| `src/memory/chat-user-context.ts` (`AsyncLocalStorage`) | não existe: closure ([por quê](./forget-preference.md#por-que-o-professor-usou-asynclocalstorage-e-nós-não)) |
| `src/agents/system-prompt.ts` | `src/agents/system-prompt.ts` |
| `src/agents/tools.ts` (`createForgetPreferenceTool`) | `src/services/tools.ts` |
| `src/chat/run-chat.ts` | `src/services/chat.ts` |
| `src/http/chat-schema.ts` (`rememberRequestSchema`) | `src/controllers/memory-controller.ts` |

Outras diferenças conscientes: `EmbeddingError` vira 503 (a referência usa 500); `created_at` é TEXT ISO (a referência usa INTEGER); o teste com modelo real fica fora do `npm test`.

---

## O fluxo em uma linha

```text
POST /chat {userId, message}
  → recall(userId, message)  → "Relevant memories:" + histórico + mensagem
  → estratégia (ReAct pode chamar forget_preference)
  → append no histórico → 200
  → (em segundo plano) refletor → { hasLearning, fact } → remember
```

---

## Como rodar e ver o que importa

Requisitos e `.env` iguais aos do [01](../../01-padroes-de-raciocinio-e-execucao/docs/variaveis-de-ambiente-com-env-file.md). O `.env` **não** foi copiado do 02: crie o seu a partir do `.env.example`.

```powershell
npm install                  # inclui @huggingface/transformers
npm test                     # 129 testes, sem rede (FakeEmbedder)
npm run test:semantic        # modelo real: baixa ~23 MB na 1ª vez, depois usa o cache
npm run dev                  # servidor com memória + refletor de aprendizado
```

Grave um fato e veja o dedup:

```bash
curl -X POST http://localhost:3000/memories -H "content-type: application/json" \
  -d '{"userId": "ana", "fact": "Always prioritize the checkout service"}'      # 201
curl -X POST http://localhost:3000/memories -H "content-type: application/json" \
  -d '{"userId": "ana", "fact": "Always prioritize the checkout service."}'     # 200, stored: false
```

Veja a memória entrar no turno (`metrics.recalledMemories`) e o agente aprender sozinho:

```bash
curl -X POST http://localhost:3000/chat -H "content-type: application/json" \
  -d '{"userId": "ana", "message": "which service should I look at first?"}'
curl -X POST http://localhost:3000/chat -H "content-type: application/json" \
  -d '{"userId": "ana", "message": "prefiro resumos curtos, sempre"}'   # o refletor grava em segundo plano
```

> ⚠️ `userId` agora é obrigatório no `/chat`: os exemplos `curl` do projeto 02 sem ele recebem 400.
