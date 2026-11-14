# Refletor de aprendizado — Reflection que olha para o futuro

> Este doc cobre a **spec 010** ([`010-learning-reflector`](../specs/010-learning-reflector/spec.md)), implementada em [`src/services/learning-reflector.ts`](../src/services/learning-reflector.ts). Ele **estende** o padrão Reflection do projeto 01: se ainda não leu [06-reflection.md](../../01-padroes-de-raciocinio-e-execucao/docs/06-reflection.md), leia antes. Aqui focamos só no que muda.
>
> **Em 30 segundos**
>
> - Depois que o `/chat` responde, um LLM lê a **mensagem do usuário** e pergunta: "tem algo aqui que vale lembrar para sempre?"
> - Saída estruturada: `{ hasLearning, fact }`.
> - Se sim → `memoryStore.remember(userId, fact)`.
> - Roda em **segundo plano** (fire-and-forget): a resposta não espera.
> - Nunca aprende **pedidos pontuais** nem **segredos**.

---

## O que é

Na spec 009 a memória só crescia se alguém chamasse `POST /memories`. Ninguém faz isso no meio de um plantão. O refletor de aprendizado resolve isso: **o agente aprende conversando**.

### Mesmo padrão, outro alvo

Você já conhece o `withReflection`: um segundo LLM critica a resposta e manda refazer. O refletor de aprendizado usa a **mesma mecânica** (LLM avaliando + `withStructuredOutput` + schema Zod), mas aponta para outra coisa e em outro momento:

| | `withReflection` (spec 002) | Refletor de aprendizado (spec 010) |
| --- | --- | --- |
| **Olha para** | A **resposta** do agente | A **mensagem** do usuário |
| **Quando** | **Antes** de entregar | **Depois** de entregar |
| **Pergunta** | "Esta resposta está certa?" | "Isto vale lembrar no futuro?" |
| **Saída** | `{ aproved, feedback }` | `{ hasLearning, fact }` |
| **Efeito** | Regenera a resposta | Grava na memória |
| **Bloqueia a resposta?** | Sim | Não |
| **Analogia** | Revisor de texto | Alguém tomando nota do que você gosta |

São independentes: com `reflect: true`, o turno passa pelo crítico **e** depois pelo refletor de aprendizado.

```text
POST /chat
  ├─ recall (spec 009)
  ├─ estratégia  ──(reflect: true?)──► withReflection critica e regenera   ← antes
  ├─ append no histórico (spec 008)
  ├─ responde 200 ──────────────────────────────────────────► cliente
  └─ void scheduleLearning(...)                                             ← depois, em paralelo
        └─ refletor → { hasLearning: true, fact } → remember
```

> ✅ **Checkpoint**: Reflection = um LLM avaliando. Antes, avaliava a resposta. Agora, também avalia a mensagem, para aprender.

---

## Como está sendo usado neste projeto

### O prompt: o que é "durável"

```typescript
// src/services/learning-reflector.ts
export const LEARNING_REFLECTOR_PROMPT = [
  "Você destila APRENDIZADOS DURÁVEIS da mensagem do usuário para a memória de longo prazo de um copiloto de plantão (OpsPilot).",
  'hasLearning=true SOMENTE se a mensagem contiver preferência ou fato operacional DURÁVEL (ex.: "sempre priorize o serviço checkout", "prefiro resumos curtos").',
  'hasLearning=false para pedidos PONTUAIS one-shot (ex.: "liste alertas", "abra um incidente", "resolva X agora", perguntas efêmeras de status).',
  "hasLearning=false se a mensagem contiver SEGREDOS ou dados sensíveis (senhas, tokens, API keys, credenciais).",
  "Quando hasLearning=true, fact deve ser um enunciado estável em uma frase (não copie o pedido bruto).",
  "Quando hasLearning=false, fact deve ser string vazia.",
].join(" ");
```

Três categorias, e só uma vira memória:

| Mensagem | Tipo | Aprende? |
| --- | --- | --- |
| "sempre priorize o checkout" | preferência durável | ✅ vira "O plantonista prioriza o checkout" |
| "liste os alertas agora" | pedido pontual | ❌ não serve para a próxima sessão |
| "minha senha do grafana é hunter2" | segredo | ❌ nunca grava |

Repare em "não copie o pedido bruto". O `fact` deve ser um **enunciado** ("O plantonista prioriza o checkout"), não um imperativo ("sempre priorize o checkout"). Enunciados funcionam melhor como contexto no bloco `Relevant memories:`.

> ⚠️ A proteção contra segredos é **só por prompt**. Um modelo fraco pode falhar. Num sistema real haveria também um filtro determinístico (regex de tokens, DLP) antes do `remember`.

### O schema: plano de propósito

```typescript
// src/services/learning-reflector.ts
// Schema plano (só boolean + string), como os outros structured outputs do projeto: modelos
// gratuitos via OpenRouter erram bem menos com objetos simples.
export const LearningReflectionSchema = z.object({
  hasLearning: z
    .boolean()
    .describe("true somente se houver preferência ou fato operacional DURÁVEL (nunca pedido pontual, nunca segredo)"),
  fact: z
    .string()
    .describe("Enunciado estável em 1 frase; string vazia se hasLearning=false. Nunca copie pedido pontual nem segredo."),
});
```

É a mesma lição dos schemas planos do Plan-and-Execute (projeto 01): `.describe()` é parte do prompt, e estruturas simples dão menos erro com modelos `:free`.

### Na dúvida, não aprende

```typescript
// src/services/learning-reflector.ts
export function createLLMLearningReflector(modelFactory: () => BaseChatModel): LearningReflectorFn {
  return async (userMessage) => {
    try {
      const output = await modelFactory()
        .withStructuredOutput(LearningReflectionSchema)
        .invoke([new SystemMessage(LEARNING_REFLECTOR_PROMPT), new HumanMessage(userMessage)]);
      return LearningReflectionSchema.parse(output);
    } catch {
      // Modelo fora do ar, JSON inválido, chave ausente: na dúvida, não aprende nada.
      return NO_LEARNING;
    }
  };
}
```

Esquecer de aprender algo é barato (o usuário repete depois). Aprender lixo é caro (polui todos os prompts futuros). Por isso, qualquer erro vira "nada a aprender".

> ✅ **Checkpoint**: prompt define durável × pontual × segredo; schema plano; erro = não aprende.

---

## Fire-and-forget: responder sem esperar

Este é o conceito mais importante da spec 010, e é de **JavaScript**, não de IA.

```typescript
// src/services/chat.ts
  if (deps.learningReflector) {
    // Fire-and-forget (spec 010): sem `await`, a resposta volta já, e o refletor (uma chamada
    // extra ao LLM) roda em segundo plano. Só chega aqui se o turno deu certo.
    void scheduleLearning({
      reflector: deps.learningReflector,
      memoryStore: deps.memoryStore,
      userId: request.userId,
      userMessage: request.message,
    });
  }
```

**Analogia**: você pede um café e o atendente anota seu nome para o cartão fidelidade. Ele não te faz esperar o sistema do cartão sincronizar para entregar o café. O café sai, e a anotação termina por trás.

Sem `await`, a promise de `scheduleLearning` começa a rodar, e o `runChat` segue direto para o `return`. O cliente recebe o 200 enquanto o refletor (que pode levar segundos, porque é outra chamada ao LLM) ainda trabalha.

O `void` na frente não muda o comportamento. Ele é um aviso para quem lê (e para linters): "sei que não estou esperando esta promise, é de propósito".

### A regra de ouro: quem é esquecido não pode lançar

Uma promise sem `await` que rejeita vira um **unhandled rejection**, e no Node isso pode derrubar o processo. Por isso `scheduleLearning` engole tudo:

```typescript
// src/services/learning-reflector.ts
  } catch (error) {
    // Best-effort, mas observável: registra no stderr em vez de engolir em silêncio.
    console.error("learning reflector failed:", error instanceof Error ? error.message : String(error));
  }
```

"Best-effort, mas observável": não quebra nada, mas deixa rastro no log. Engolir em silêncio esconderia um banco quebrado por semanas.

### Só turnos bem-sucedidos aprendem

O `scheduleLearning` fica **depois** do `conversationStore.append`. Se a estratégia lançar (timeout, erro), a execução nem chega lá. É coerente com a regra da spec 008 ("nunca persistir turno parcial"), e o teste `runChat does not run the learning reflector when the strategy fails` garante isso.

### Onde o refletor é ligado

```typescript
// src/index.ts
  // O refletor de aprendizado (spec 010) só é ligado aqui, no servidor de verdade. Testes de
  // /chat não o recebem e por isso nunca chamam o LLM em segundo plano.
  await startServer(port, { learningReflector: createLLMLearningReflector(createModel) });
```

`resolveChatDependencies` **não** cria um refletor padrão. Se criasse, todo teste de `/chat` dispararia uma chamada de LLM escondida.

> ✅ **Checkpoint**: fire-and-forget = disparar sem `await`. Para isso ser seguro, a função disparada nunca pode lançar.

---

## O que ficou de fora (e por quê)

| Item | Motivo |
| --- | --- |
| Evento `learning` no trace | O trace pertence à resposta, que já foi enviada quando o refletor termina. |
| Refletor na arena/bench | Esses fluxos não têm `userId` nem memória. |
| Custo | +1 chamada de LLM por turno. Com `:free` é zero; em produção, valeria amostrar. |

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Prompt e schema | [`src/services/learning-reflector.ts`](../src/services/learning-reflector.ts) | `LEARNING_REFLECTOR_PROMPT`, `LearningReflectionSchema` |
| Fallback "não aprende" | [`src/services/learning-reflector.ts`](../src/services/learning-reflector.ts) | `catch` em `createLLMLearningReflector` |
| Nunca lança | [`src/services/learning-reflector.ts`](../src/services/learning-reflector.ts) | `scheduleLearning` + `console.error` |
| Disparo sem `await` | [`src/services/chat.ts`](../src/services/chat.ts) | `void scheduleLearning(...)` depois do `append` |
| Ligação no servidor real | [`src/index.ts`](../src/index.ts) | `createLLMLearningReflector(createModel)` |
| Comparação com o crítico | [`src/agents/reflection.ts`](../src/agents/reflection.ts) | `withReflection`, `ReflectionCritiqueSchema` |
| Prova do não-bloqueio | [`src/services/chat.test.ts`](../src/services/chat.test.ts) | `runChat returns before the learning reflector's remember finishes` |

---

⬅️ Anterior: [02 · Memória semântica](./02-memoria-semantica.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [04 · forget_preference](./04-forget-preference.md)**
