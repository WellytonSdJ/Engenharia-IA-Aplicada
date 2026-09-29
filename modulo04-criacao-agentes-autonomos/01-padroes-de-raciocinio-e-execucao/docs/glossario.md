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
| **`diagnose`** | Função por cenário que explica um erro (o que faltou) sem alterar o critério de acerto. |
| **`--no-replanner`** | Flag do bench que roda o Plan-and-Execute sem replanner, para medir custo × benefício do replanner. |
| **Exaustividade com `never`** | `const _x: never = valor` — não compila se algum caso de uma união não foi tratado antes. |

---

## Domínio (OpsPilot / Mercadinho)

| Termo | Definição |
| --- | --- |
| **OpsPilot** | Copiloto de plantão / incident commander do e-commerce fictício "Mercadinho"; projeto único que evolui ao longo do módulo. |
| **Alerta `firing` / `resolved`** | Alerta disparando agora / já normalizado. O seed tem 3 de cada. |
| **Incidente `open` / `resolved`** | Ocorrência aberta pelo agente / encerrada. Transição válida: `open → resolved`. |
| **sev1..sev4** | Jargão de severidade mapeado para `critical`, `high`, `medium`, `low`. |
| **`InMemoryStore` / `InMemoryOpsStore`** | Store em memória semeado (nome da referência / nome no nosso plano). |
| **Seed** | Estado inicial fixo (5 serviços, 6 alertas) que torna as execuções comparáveis. |
