# Testando memória — FakeEmbedder, teste semântico e fire-and-forget

> A filosofia de testes sem rede e com doubles vem do projeto 01 ([03-tools-e-store-deterministico.md](../../01-padroes-de-raciocinio-e-execucao/docs/03-tools-e-store-deterministico.md)). Este doc cobre só os três problemas **novos** que memória e aprendizado trazem para os testes.
>
> **Em 30 segundos**
>
> - **Problema 1**: o modelo real baixa 23 MB → `FakeEmbedder`, em que *você* escolhe os vetores.
> - **Problema 2**: o fake não prova semântica de verdade → `*.semantic-test.ts`, rodado à parte com `npm run test:semantic`.
> - **Problema 3**: como testar que algo **não** foi esperado? → uma promise "travada" num portão.

---

## Problema 1: `npm test` não pode baixar modelo

Todo teste de `/chat` agora faz recall. Se usasse o embedder real, o `npm test` passaria a depender de internet e de alguns segundos de cold start. A solução é um **double** com vetores escolhidos à mão:

```typescript
// src/services/fake-embedder.ts
  /** Vetor unitário num único eixo: textos no mesmo eixo têm similaridade 1, em eixos diferentes 0. */
  setAxis(text: string, axis: number): this {
    return this.set(text, Array.from({ length: EMBEDDING_DIM }, (_, index) => (index === axis ? 1 : 0)));
  }
```

**Analogia**: em vez de um tradutor de verdade, você usa um dicionário que você mesmo escreveu. Não serve para provar que o tradutor é bom, mas serve perfeitamente para testar *o resto do sistema*.

Com `setAxis` você controla a similaridade:

```typescript
// src/services/chat.test.ts
  const embedder = new FakeEmbedder().setAxis("checkout primeiro", 3).setAxis("e agora?", 3);
```

"checkout primeiro" e "e agora?" não se parecem nada, mas no eixo 3 os dois têm similaridade 1. O teste não quer provar semântica. Ele quer provar que **o recall entra no prompt**.

Para limiares, `set` aceita vetores arbitrários (o fake normaliza sozinho). Assim dá para montar um fato com cosseno 0,31 e outro com 0,29 em relação à consulta:

```typescript
// src/models/memory-store.test.ts
// Testamos logo acima e logo abaixo do piso (e não exatamente 0.3): Float32 arredonda, e um
// "0.3" pode virar 0.29999998 — exatamente o tipo de teste que fica instável sem motivo.
```

### O cinto de segurança no servidor de teste

```typescript
// src/http/server.test.ts
    // Sem isso o servidor usaria o embedder real e tentaria baixar o modelo durante os testes.
    memoryStore: new SqliteMemoryStore(":memory:", new FakeEmbedder()),
```

> ✅ **Checkpoint**: fake = vetores escolhidos à mão → você controla a similaridade → teste determinístico.

---

## Problema 2: e quem prova que a busca é semântica?

O `FakeEmbedder` testa a **lógica** (dedup, piso, top-3, isolamento). Mas o requisito central da spec 009 é: "o recall acha um fato **sem palavras em comum**". Isso só um modelo de verdade prova.

A solução foi separar por **nome de arquivo**:

```json
// package.json
"test": "node --import tsx --test \"src/**/*.test.ts\"",
"test:semantic": "node --import tsx --test \"src/**/*.semantic-test.ts\"",
```

`memory-store.semantic-test.ts` não casa com `*.test.ts`, então o `npm test` nunca o pega. O teste principal:

```typescript
// src/models/memory-store.semantic-test.ts
  // Nenhuma palavra em comum com o primeiro fato: "card charges" ≈ "payment", "rush time" ≈ "peak hours".
  const recalled = await store.recall("plantonista", "Are card charges slow at rush time?");

  assert.match(recalled[0]?.fact ?? "", /checkout payment queue/);
```

| Script | Rede? | Tempo | Prova |
| --- | --- | --- | --- |
| `npm test` | não | ~30 s | a lógica do sistema inteiro |
| `npm run test:semantic` | só na 1ª vez | ~7 s (com cache) | que o modelo real entende significado |

Essa divisão virou regra na constituição (v1.2.0). A referência do curso roda o modelo real dentro do `npm test`; aqui preservamos o "`npm test` sem rede" que vem desde o projeto 01.

> ✅ **Checkpoint**: lógica → fake no `npm test`. Semântica → modelo real no `npm run test:semantic`.

---

## Problema 3: testar que algo NÃO foi esperado

Como provar que o `runChat` **não espera** o `remember`? Medir tempo é frágil. O truque é um **portão**: um `remember` que trava numa promise que só o teste abre.

```typescript
// src/services/chat.test.ts
  let releaseRemember!: () => void;
  const gate = new Promise<void>((resolve) => {
    releaseRemember = resolve;
  });
  // (...)
  memoryStore.remember = async () => {
    rememberStarted = true;
    await gate;
    rememberFinished = true;
    return { id: "memory-late", stored: true };
  };
```

E a sequência de asserções conta a história:

```typescript
// src/services/chat.test.ts
  const result = await runChat({ userId: "ana", message: "sempre priorize o checkout", reflect: false }, deps);

  assert.equal(result.answer, "react: sempre priorize o checkout");
  assert.equal(rememberFinished, false);          // ① runChat voltou com o remember travado
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(rememberStarted, true);            // ② mas ele foi disparado
  assert.equal(rememberFinished, false);
  releaseRemember();                              // ③ abrimos o portão
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(rememberFinished, true);           // ④ e ele termina sozinho, em segundo plano
```

Se alguém um dia colocar `await` antes de `scheduleLearning`, o `runChat` nunca resolveria (o portão está fechado) e o teste quebraria por timeout. É exatamente a regressão que queremos pegar.

O `setImmediate` é "espere a fila de microtasks andar": dá tempo para as promises em segundo plano progredirem.

### Fake de modelo para o refletor

O refletor só usa `withStructuredOutput(...).invoke(...)`, então o fake implementa só isso:

```typescript
// src/services/learning-reflector.test.ts
/** Modelo fake que só implementa withStructuredOutput — o único método que o refletor usa. */
function structuredModel(reply: LearningReflection | Error, seen: BaseLanguageModelInput[] = []): BaseChatModel {
```

É uma versão enxuta do `ControlledCriticModel` que o projeto já usava para testar a Reflection.

> ✅ **Checkpoint**: para provar "não esperou", trave a dependência num portão e veja a função voltar mesmo assim.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Double de embedder | [`src/services/fake-embedder.ts`](../src/services/fake-embedder.ts) | `set`, `setAxis`, fallback por hash |
| Lógica do store | [`src/models/memory-store.test.ts`](../src/models/memory-store.test.ts) | dedup, piso, top-3, isolamento, atalho sem embedder |
| Semântica real | [`src/models/memory-store.semantic-test.ts`](../src/models/memory-store.semantic-test.ts) | recall sem palavras em comum |
| Portão do fire-and-forget | [`src/services/chat.test.ts`](../src/services/chat.test.ts) | `releaseRemember`, `setImmediate` |
| Fake de modelo estruturado | [`src/services/learning-reflector.test.ts`](../src/services/learning-reflector.test.ts) | `structuredModel` |
| Scripts | [`package.json`](../package.json) | `test` × `test:semantic` |

---

⬅️ Anterior: [04 · forget_preference](./04-forget-preference.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [06 · Glossário](./06-glossario.md)**
