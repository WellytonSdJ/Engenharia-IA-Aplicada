# Glossário — 03-memoria-e-reflexao-em-agentes-autonomos

Termos já cobertos no [glossário do projeto 01](../../01-padroes-de-raciocinio-e-execucao/docs/glossario.md) (Reflection, crítico, structured output, estratégias, Spec Kit) não são repetidos aqui. O projeto 02 não tem glossário próprio.

---

## Embeddings e busca semântica

| Termo | Definição |
| --- | --- |
| **Embedding** | Vetor de números que representa o significado de um texto. Aqui: 384 floats (`EMBEDDING_DIM`). |
| **Modelo de embeddings** | Modelo que só gera embeddings, sem conversar. Aqui: `Xenova/all-MiniLM-L6-v2` (`MODEL_ID`). |
| **Transformers.js** | Pacote `@huggingface/transformers`: roda modelos do Hugging Face em JavaScript, via ONNX. |
| **ONNX** | Formato portátil de modelo de ML; é o que permite rodar o MiniLM em Node, sem Python. |
| **Pipeline `feature-extraction`** | Tarefa do Transformers.js que devolve o vetor do texto (e não texto gerado). |
| **Mean pooling** | Média dos vetores de cada token, para ter um vetor da frase inteira (`pooling: "mean"`). |
| **Normalização L2** | Dividir o vetor pela sua norma, deixando-o com tamanho 1 (`normalize: true`). |
| **Similaridade de cosseno** | Mede se dois vetores apontam para a mesma direção: 1 = mesmo assunto, 0 = nada a ver. |
| **Produto escalar** | Soma das multiplicações posição a posição (`dot`). Com vetores normalizados, é igual ao cosseno. |
| **Busca semântica** | Achar textos por significado, mesmo sem nenhuma palavra em comum. |
| **Vector store** | Qualquer lugar que guarda vetores e responde "quais são os mais parecidos?". Aqui: SQLite + loop. |
| **Força bruta (brute-force)** | Comparar a consulta com todos os vetores do usuário, sem índice. Ok para volumes pequenos. |
| **Índice ANN** | *Approximate nearest neighbor*: índice que acelera a busca em milhões de vetores, trocando exatidão por velocidade. Não usado aqui. |
| **Cold start** | Primeira chamada que precisa carregar o modelo (download + inicialização). |
| **Lazy singleton** | Instância única criada só no primeiro uso (`getDefaultEmbedder`, `getPipeline`). |

---

## Memória do agente

| Termo | Definição |
| --- | --- |
| **Memória de curto prazo** | O fio da conversa atual: o histórico de mensagens da spec 008, por `conversationId`. |
| **Memória de longo prazo / semântica** | Fatos sobre o usuário que sobrevivem entre conversas, por `userId` (spec 009). |
| **`MemoryStore`** | Contrato com `remember`, `recall` e `forget`, sempre por `userId`. |
| **`SqliteMemoryStore`** | Implementação do `MemoryStore` com a tabela `memories` no `OPSPILOT_DB`. |
| **`remember`** | Grava um fato; devolve `{ id, stored }`. `stored: false` = deduplicado. |
| **`recall`** | Devolve até 3 fatos (`RECALL_TOP_K`) com score ≥ 0,3 (`RECALL_MIN_SCORE`), do mais parecido ao menos. |
| **`forget`** | Apaga uma memória do próprio usuário; devolve `false` se não apagou nada. |
| **Dedup semântico** | Não gravar um fato com similaridade > 0,92 (`DEDUP_THRESHOLD`) com um já existente do mesmo usuário. |
| **Top-k** | Quantos resultados no máximo o recall devolve (k = 3). |
| **Piso de similaridade (threshold)** | Score mínimo para um fato contar como relevante (0,3). |
| **Score** | A similaridade entre consulta e fato, devolvida em `RecalledMemory.score`. |
| **`Relevant memories:`** | Cabeçalho do bloco de fatos que o `/chat` coloca no topo do prompt (`formatMemoriesForPrompt`). |
| **`recalledMemories`** | Métrica da resposta do `/chat`: quantas memórias entraram no turno. |
| **BLOB** | Coluna de bytes crus do SQLite; guarda o vetor (384 × 4 = 1536 bytes). |
| **`Embedder`** | Contrato `embed(text) → Float32Array` normalizado; real (`getDefaultEmbedder`) ou fake. |
| **`EmbeddingError`** | Erro de domínio quando o modelo não carrega ou falha; vira 503 `EMBEDDING_UNAVAILABLE`. |
| **`InvalidMemoryInputError`** | Fato, consulta ou `userId` vazio chegando direto ao store; vira 400. |

---

## Aprendizado e tools

| Termo | Definição |
| --- | --- |
| **Refletor de aprendizado** | LLM que, depois do turno, lê a mensagem do usuário e decide se há um fato durável para lembrar (spec 010). |
| **`{ hasLearning, fact }`** | Saída estruturada do refletor (`LearningReflectionSchema`). |
| **Fato durável** | Preferência ou contexto que vale para sessões futuras ("prefiro resumos curtos"). |
| **Pedido pontual (one-shot)** | Pedido que só vale agora ("liste alertas"); o refletor não aprende. |
| **`LearningReflectorFn`** | Tipo da função refletora; injetável (real com LLM ou fake nos testes). |
| **`scheduleLearning`** | Chama o refletor e, se houver fato, `remember`. Nunca lança. |
| **Fire-and-forget** | Disparar uma promise sem `await`, para a resposta não esperar por ela. |
| **`void promise`** | Marca explícita de "não estou esperando esta promise, é de propósito". |
| **Unhandled rejection** | Promise rejeitada que ninguém trata; pode derrubar o processo Node. Por isso `scheduleLearning` nunca lança. |
| **Best-effort** | Tenta fazer, mas a falha não afeta o fluxo principal (só vai para o log). |
| **`forget_preference`** | Tool do ReAct que faz recall top-1 da descrição e dá `forget` nele. |
| **Closure** | Função que "lembra" variáveis do escopo onde foi criada; é como o `userId` chega à tool. |
| **`ReasoningOptions.memory`** | `{ store, userId }` passado pelo `/chat` às estratégias; liga o `forget_preference`. |
| **`AsyncLocalStorage`** | API do Node que guarda contexto por requisição assíncrona. A referência usa; nós trocamos por closure. |
| **`OPSPILOT_SYSTEM_PROMPT`** | System prompt do ReAct: formato Resumo/Achados/Próximos passos + como ler as memórias. |
| **`MEMORY_GUIDANCE`** | A instrução sobre memórias, reaproveitada no planner do Plan-and-Execute. |

---

## Testes

| Termo | Definição |
| --- | --- |
| **`FakeEmbedder`** | Double de `Embedder` em que você escolhe o vetor de cada texto (`set`, `setAxis`). |
| **`setAxis`** | Dá a um texto um vetor unitário num eixo: mesmo eixo = similaridade 1. |
| **`*.semantic-test.ts`** | Testes com o modelo real, fora do `npm test`; rodam com `npm run test:semantic`. |
| **Portão (gate)** | Promise que só o teste resolve; serve para provar que uma função **não** esperou outra. |
