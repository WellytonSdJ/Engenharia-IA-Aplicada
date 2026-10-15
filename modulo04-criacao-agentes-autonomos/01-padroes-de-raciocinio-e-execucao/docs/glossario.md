# Glossário — 01-padroes-de-raciocinio-e-execucao

Termos cobertos no [glossário do projeto anterior](../../../modulo03-mcp-na-pratica/07-api-security-auth-rate-limiting/customers-mcp/docs/glossario.md) e no [glossário do `01-multiple-mcp-tools`](../../../modulo03-mcp-na-pratica/01-multiple-mcp-tools/docs/glossario.md) (agente autônomo, tool calling loop, callbacks) não são repetidos aqui. Termos de LangGraph (`StateGraph`, `Annotation`, reducer, edges condicionais) estão no [glossário do `04-song-highlights`](../../../modulo02-integracao-apis-llms/04-song-highlights/docs/glossario.md).

---

## Spec-Driven Development

| Termo | Definição |
| --- | --- |
| **Spec-Driven Development (SDD)** | Desenvolvimento em que spec, plano e tarefas são escritos e versionados antes do código, e servem de contrato para quem implementa (pessoa ou agente). |
| **Spec Kit** | Toolkit do GitHub que instala o processo SDD num projeto: templates, scripts e comandos `speckit.*` para o agente de código. |
| **Constituição** | `.specify/memory/constitution.md` — princípios permanentes do projeto, versionados, checados em toda etapa. |
| **Constitution Check** | Seção do `plan.md` que marca PASS/violação para cada princípio da constituição, antes e depois do design. |
| **Gate** | Ponto de aprovação humana entre etapas (`approve`/`reject`) declarado no workflow do Spec Kit. |
| **`[P]` / `[USn]`** | Marcadores do `tasks.md`: tarefa paralelizável / tarefa ligada à user story *n*. |
| **`speckit-analyze`** | Comando que cruza spec, plano e tarefas em busca de inconsistências antes da implementação. |
| **`speckit-converge`** | Comando que compara o código com spec/plano/tarefas e acrescenta ao `tasks.md` o que falta construir. |

---

## Padrões de raciocínio

| Termo | Definição |
| --- | --- |
| **Estratégia de raciocínio** | Um jeito de o agente chegar à resposta (ReAct, Plan-and-Execute, reflect:*), exposto pela interface `ReasoningStrategy`. |
| **ReAct** | *Reasoning + Acting*: o modelo alterna pensamento, ação (tool call) e observação, decidindo um passo por vez. |
| **Plan-and-Execute** | Padrão com planner (escreve os passos), executor (executa um passo por vez) e replanner (revisa o plano após cada passo). |
| **Planner** | Nó que transforma o pedido numa lista estruturada de até 8 passos (`planSchema`). |
| **Executor** | Nó que executa só o próximo passo, com um `createReactAgent` de escopo reduzido. |
| **Replanner** | Nó que decide entre `continue`, `adjust` (novo plano) e `finish` (resposta final) usando `replanSchema`. |
| **Reflection** | Padrão em que um crítico (LLM) avalia a resposta e pede regeneração com feedback até aprovar ou atingir o teto. |
| **Crítico (`CriticFn`)** | Função `(answer, trace, originalInput) → { approved, feedback }`; no LLM, avalia só contra as observações do trace. |
| **Fail-safe do crítico** | Se o crítico falha ou devolve saída inválida, a resposta é tratada como aprovada. |
| **`maxReflections`** | Teto de rodadas de crítica (padrão 2). Com 0, o decorator não chama o crítico. |

---

## Contrato e observabilidade

| Termo | Definição |
| --- | --- |
| **`ReasoningStrategy`** | Interface comum: `name` + `run(input)` → `{ answer, trace, metrics }`. |
| **`TraceEvent`** | Evento tipado do raciocínio: `thought`, `action`, `observation`, `plan`, `critique` ou `answer`. |
| **Trace tipado** | Lista de `TraceEvent` devolvida junto com a resposta — dado verificável, diferente de log em texto. |
| **`llmCalls`** | Número real de chamadas ao modelo numa execução (contado, não estimado). |
| **`latencyMs`** | Tempo de parede da execução inteira, em milissegundos. |
| **`buildTraceFromMessages`** | Converte o histórico de mensagens do LangGraph (`AIMessage`/`ToolMessage`) em `TraceEvent[]`, garantindo um `answer` no fim. |
| **Decorator** | Função que recebe um objeto de uma interface e devolve outro da mesma interface com comportamento extra — aqui, `withReflection`. |

---

## LangGraph / LangChain (novo nesta unidade)

| Termo | Definição |
| --- | --- |
| **`createReactAgent`** | Agente ReAct pré-construído de `@langchain/langgraph/prebuilt`: grafo `agent ⇄ tools` cujo estado é a lista de mensagens. |
| **`recursionLimit`** | Máximo de super-steps do grafo numa execução; estourá-lo lança `GraphRecursionError`. |
| **`GraphRecursionError`** | Erro do LangGraph quando o `recursionLimit` é atingido — o sinal de "limite de iterações" do ReAct. |
| **`modelFactory`** | Função `() => ChatOpenAI` injetada nas estratégias: uma instância nova por uso e troca fácil por um double nos testes. |
| **Schema plano (flat)** | Objeto Zod com campos opcionais no lugar de `discriminatedUnion`, porque muitos modelos via OpenRouter quebram com uniões no structured output. |
| **`z.preprocess`** | Transforma o valor antes da validação — usado para trocar `undefined`/`null` por `"firing"` em `list_alerts`. |

---

## Avaliação

| Termo | Definição |
| --- | --- |
| **Arena** | CLI que roda várias estratégias sobre o mesmo pedido, cada uma com seu store, e imprime trace, métricas e resposta. |
| **Bench** | CLI com cenários fixos que mede acerto de cada estratégia conferindo o estado do store. |
| **Acerto no estado** | Critério de sucesso baseado no efeito das ações (incidentes criados/resolvidos), não no texto da resposta. |
| **Cenários C1 / C2 / C3** | `direto` (só leitura + contagem), `estruturado` (3 aberturas em ordem + 1 resolução), `dinâmico` (ler → decidir → abrir → contar). |
| **Lista de motivos (`reasons`)** | Retorno do `check` de cada cenário: vazia significa acerto; quando não vazia, já explica o que faltou — não existe uma função `diagnose` separada. |
| **`--no-replanner`** | Flag do bench que roda o Plan-and-Execute sem replanner, para medir custo × benefício do replanner. No nosso código, troca a estratégia por `plan-and-execute:no-replanner`. |
| **Finalizer** | Nó do Plan-and-Execute sem replanner: depois do último passo, uma chamada redige a resposta a partir dos resultados. |
| **Célula do bench** | Um par cenário × estratégia, executado com um store novo; é uma linha da tabela. |
| **`answerMentionsCount`** | Verifica se a resposta cita a contagem esperada (dígito isolado ou por extenso), calculada a partir do store. |
| **Catálogo de tools (`describeTools`)** | Texto com nome, argumentos e descrição de cada tool, injetado nos prompts do planner e do executor, que não recebem as tools por binding. |
| **Exaustividade com `never`** | `const _x: never = valor` — não compila se algum caso de uma união não foi tratado antes. |

---

## Ambiente de execução

| Termo | Definição |
| --- | --- |
| **Variável de ambiente** | Par `NOME=valor` entregue pelo sistema operacional ao processo quando ele nasce; no Node, fica em `process.env`. |
| **`.env` / `.env.example`** | Arquivo de texto com variáveis de ambiente (fora do git) e o modelo versionado, que lista os nomes sem os valores. |
| **`--env-file=<arq>`** | Flag do Node (≥ 20.6) que carrega o arquivo em `process.env` antes do código rodar; aborta se o arquivo não existir. |
| **`--env-file-if-exists=<arq>`** | Variante (Node ≥ 22.9) que segue em frente se o arquivo não existir. É a usada nos nossos scripts. |
| **`--import tsx`** | Registra o `tsx` como loader do Node para executar `.ts` diretamente, mantendo as flags nativas do Node disponíveis. |
| **`process.loadEnvFile()`** | Equivalente em runtime (Node ≥ 20.12) ao `--env-file`, para quando não dá para mexer na linha de comando. |

## Domínio (OpsPilot / Mercadinho)

| Termo | Definição |
| --- | --- |
| **OpsPilot** | Copiloto de plantão / incident commander do e-commerce fictício "Mercadinho"; projeto único que evolui ao longo do módulo. |
| **Alerta `firing` / `resolved`** | Alerta disparando agora / já normalizado. O seed tem 3 de cada. |
| **Incidente `open` / `resolved`** | Ocorrência aberta pelo agente / encerrada. Transição válida: `open → resolved`. |
| **sev1..sev4** | Jargão de severidade mapeado para `critical`, `high`, `medium`, `low`. |
| **`InMemoryOpsStore`** | Store em memória semeado, usado em testes, bench e arena. |
| **Seed** | Estado inicial fixo (6 serviços, incluindo `catalog` sem alertas, e 6 alertas) que torna as execuções comparáveis. |
