# Documentação — 03-memoria-e-reflexao-em-agentes-autonomos

Este projeto nasce como cópia do [`02-function-calling-e-tool-use`](../../02-function-calling-e-tool-use/): mesmo OpsPilot, mesmas estratégias (ReAct, Plan-and-Execute, Reflection), mesmo `POST /chat` com SQLite, tools, MCP e conversa persistente. Nada disso é re-explicado aqui.

O que esta unidade acrescenta é **memória de longo prazo e aprendizado**: o agente passa a lembrar fatos sobre cada usuário (buscados por significado, com embeddings locais) e a aprender esses fatos sozinho depois de cada turno, com uma variação do padrão Reflection.

> **Estado atual:** specs 001–008 herdadas e `009-semantic-memory` + `010-learning-reflector` implementadas. `npm test`: 129 testes sem rede; `npm run test:semantic`: 3 testes com o modelo real.

Comece por [00-START-HERE.md](./00-START-HERE.md).

---

## Documentos deste projeto

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | O que muda nesta unidade, trilha de leitura, mapa do código, mapa referência → nosso e como rodar |
| [embeddings-e-similaridade.md](./embeddings-e-similaridade.md) | O que é embedding, cosseno = produto escalar com vetor normalizado, `all-MiniLM-L6-v2` local, lazy singleton |
| [memoria-semantica.md](./memoria-semantica.md) | Curto × longo prazo, vector store em SQLite (BLOB + força bruta), `remember`/`recall`/`forget`, limiares 0,92 e 0,3, `userId` e memórias no `/chat`, `POST /memories` |
| [refletor-de-aprendizado.md](./refletor-de-aprendizado.md) | Reflection apontada para a mensagem do usuário, `{ hasLearning, fact }`, durável × pontual × segredo, fire-and-forget |
| [forget-preference.md](./forget-preference.md) | Tool recall + forget, `userId` por closure (e não pelo schema), por que a referência usa `AsyncLocalStorage` |
| [testando-memoria.md](./testando-memoria.md) | `FakeEmbedder`, `*.semantic-test.ts` fora do `npm test`, teste do "não esperou" com portão |
| [glossario.md](./glossario.md) | Termos novos desta unidade |

## O que já está documentado antes (não repetido aqui)

| Documento | Conteúdo |
| --- | --- |
| [01 / reflection.md](../../01-padroes-de-raciocinio-e-execucao/docs/reflection.md) | O padrão Reflection original (crítico + regeneração), base do refletor de aprendizado |
| [01 / tools-e-store-deterministico.md](../../01-padroes-de-raciocinio-e-execucao/docs/tools-e-store-deterministico.md) | Tools com `tool()` + Zod, store injetado, testes sem rede |
| [01 / contrato-de-estrategia-e-trace.md](../../01-padroes-de-raciocinio-e-execucao/docs/contrato-de-estrategia-e-trace.md) | `ReasoningStrategy`, `ReasoningOptions`, trace e métricas |
| [01 / plan-and-execute.md](../../01-padroes-de-raciocinio-e-execucao/docs/plan-and-execute.md) | Planner, executor e replanner |
| [01 / spec-driven-com-spec-kit.md](../../01-padroes-de-raciocinio-e-execucao/docs/spec-driven-com-spec-kit.md) | O processo Spec Kit e a constituição |
| [02 / README.md](../../02-function-calling-e-tool-use/docs/README.md) | `POST /chat`, SQLite, MCP e conversa persistente (spec 008) |
| [modulo02 / rag-com-grafos.md](../../../modulo02-integracao-apis-llms/06-rag-neo4j-students/docs/rag-com-grafos.md) | Onde "RAG com vetor" apareceu pela primeira vez, só como comparação |

---

## Contexto do projeto

| Tecnologia | Papel neste projeto |
| --- | --- |
| `@huggingface/transformers` (Transformers.js) | **Nova.** Roda o modelo de embeddings em Node, via ONNX, sem API |
| `Xenova/all-MiniLM-L6-v2` | **Novo.** Modelo de 384 dimensões que transforma frases em vetores |
| `node:sqlite` | Já existia; agora também guarda a tabela `memories` com o vetor em BLOB |
| LangChain `withStructuredOutput` | Já existia (crítico da Reflection); agora também no refletor de aprendizado |
| Express 5 + Zod | Já existiam; `/chat` exige `userId` e nasce `POST /memories` |

A dependência nova exigiu a **emenda v1.2.0 da constituição** ([`.specify/memory/constitution.md`](../.specify/memory/constitution.md)), que também oficializa o script `test:semantic`.

### Contrato HTTP desta unidade

| Rota | Mudança | Status novos |
| --- | --- | --- |
| `POST /chat` | `userId` **obrigatório**; resposta ganha `metrics.recalledMemories` | 400 sem `userId`; 503 `EMBEDDING_UNAVAILABLE` |
| `POST /memories` | **nova**: `{ userId, fact }` | 201 gravado; 200 deduplicado; 400; 503 |
