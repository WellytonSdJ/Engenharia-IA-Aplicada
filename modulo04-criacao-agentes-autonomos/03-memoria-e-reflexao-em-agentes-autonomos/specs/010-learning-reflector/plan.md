# Implementation Plan: Refletor de Aprendizado

**Branch**: `010-learning-reflector` | **Date**: 2026-10-26 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/010-learning-reflector/spec.md`

## Summary

Depois de persistir um turno bem-sucedido, `runChat` dispara `scheduleLearning` **sem `await`**: o refletor (`withStructuredOutput(LearningReflectionSchema)`) decide se a mensagem do usuário tem um fato durável e, se tiver, chama `memoryStore.remember`. A tool `forget_preference` entra no catálogo do ReAct quando `ReasoningOptions.memory` vem preenchido (só no `/chat`), com o `userId` preso por closure.

## Technical Context

**Language/Version**: TypeScript ESM, `strict: true`, Node.js 22 LTS

**Primary Dependencies**: existentes (`@langchain/core`, `zod`); nada novo

**Storage**: reutiliza `SqliteMemoryStore` (spec 009)

**Testing**: refletor fake injetável, `MemoryStore` espião, `FakeEmbedder`; sem rede

**Constraints**: a resposta do `/chat` nunca espera o refletor; falhas são best-effort mas registradas no stderr; o MCP não expõe a tool nova

## Constitution Check

| Princípio/Regra | Como o plano atende |
| --- | --- |
| I. TypeScript ESM Estrito | `LearningReflectorFn`, `LearningReflectionSchema`, `ForgetPreferenceInputSchema` explícitos. |
| II. Arquitetura em Camadas | Refletor e tool em `src/services/`; orquestração em `src/services/chat.ts`; ligação do refletor real só em `src/index.ts` (borda). |
| III. Contratos e Erros Explícitos | Saída do LLM validada com Zod; schema da tool sem `userId`; erros viram observação ou "sem aprendizado". |
| IV. Testes Não Negociáveis | `learning-reflector.test.ts`, casos novos em `chat.test.ts`, `tools.test.ts` e `react.test.ts`. |
| V. Integração Segura e Simples | Nenhuma dependência nova; o prompt proíbe aprender segredos. |

**Post-design re-check**: **PASS**.

## Project Structure

```text
src/
├── services/
│   ├── learning-reflector.ts       # NOVO: prompt, schema, createLLMLearningReflector, scheduleLearning
│   ├── learning-reflector.test.ts  # NOVO
│   ├── tools.ts                    # ALTERADO: forget_preference (definição + LangChain tool)
│   ├── tools.test.ts               # ALTERADO
│   ├── chat.ts                     # ALTERADO: learningReflector?, options.memory, fire-and-forget
│   └── chat.test.ts                # ALTERADO
├── agents/
│   ├── strategy.ts                 # ALTERADO: ReasoningOptions.memory
│   ├── react.ts                    # ALTERADO: adiciona forget_preference quando há memory
│   └── react.test.ts               # ALTERADO
└── index.ts                        # ALTERADO: liga createLLMLearningReflector(createModel)
```

**Structure Decision**: na referência, `learning-reflector.ts` e `chat-user-context.ts` ficam em `src/memory/`. Aqui o refletor vai para `src/services/` (mesmo nome de arquivo) e o `chat-user-context.ts` **não existe**: o closure resolve o mesmo problema com menos peças (ver decisões).

## Decisões e rastreabilidade

| Decisão | Atende |
| --- | --- |
| `scheduleLearning` só roda depois de `conversationStore.append`, ou seja, só em turno bem-sucedido | FR-001, edge case |
| `void scheduleLearning(...)` sem `await`: a promise corre em segundo plano e `runChat` retorna | FR-003, SC-003 |
| `scheduleLearning` nunca lança: `try/catch` com `console.error` | FR-006 |
| `createLLMLearningReflector` devolve `{ hasLearning: false, fact: "" }` em qualquer erro do modelo | FR-006 |
| `fact.trim()` vazio com `hasLearning: true` → não grava | edge case |
| `userId` da tool por **closure** (`createForgetPreferenceTool(store, userId)`): nossas tools são criadas a cada `strategy.run`, então o usuário do turno já está à mão. A referência precisa de `AsyncLocalStorage` porque cria as tools uma vez só, no bootstrap | FR-008 |
| `forget_preference` fora de `createOpsToolDefinitions`: o MCP continua com as 3 tools da spec 007 | escopo |
| Plan-and-Execute não ganha a tool: o executor usa `ToolExecutionSchema` com enum fixo de tools operacionais | escopo |
| Refletor ligado só em `main()` (`src/index.ts`); `resolveChatDependencies` não cria refletor padrão | testes sem rede |

## Riscos

- **Custo**: +1 chamada de LLM por turno do `/chat`. Aceitável com modelos `:free`; em produção valeria amostrar.
- **Falso positivo**: um modelo fraco pode aprender um pedido pontual. Mitigação: prompt com exemplos, dedup da spec 009 e `forget_preference` para corrigir.
- **Segredos**: a proteção é só por prompt, sem DLP. Documentado como limitação.
