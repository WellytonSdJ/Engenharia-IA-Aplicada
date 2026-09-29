# Spec-Driven Development com Spec Kit

## O que é

No módulo 03, em [`03-dev-instructions-agents`](../../../modulo03-mcp-na-pratica/03-dev-instructions-agents/docs/custom-agents-copilot.md), você viu que dá para **programar o comportamento de um agente de código** escrevendo Markdown (`.agent.md`): persona, ferramentas permitidas e fluxo de trabalho. O Spec Kit leva essa ideia um passo adiante: em vez de um agente só, ele instala **um processo inteiro** — um conjunto de comandos que conduzem o agente de código (aqui, o GitHub Copilot) por etapas com artefatos versionados entre elas.

A analogia é uma obra. Ninguém começa levantando parede: primeiro vem o **programa de necessidades** (o que a família precisa), depois a **planta** (como vai ser), depois o **cronograma** (em que ordem se faz) — e só então o **pedreiro**. Se no meio da obra alguém quiser uma suíte a mais, volta-se para a planta, não se quebra parede no improviso.

```
/speckit.constitution  → as regras permanentes da obra (vale para todas as features)
/speckit.specify       → spec.md       (O QUÊ e POR QUÊ — sem código)
     └── gate: revisar e aprovar
/speckit.plan          → plan.md + research.md + data-model.md + contracts/ + quickstart.md (COMO)
     └── gate: revisar e aprovar
/speckit.tasks         → tasks.md      (EM QUE ORDEM, tarefa por tarefa)
/speckit.implement     → código + testes, seguindo tasks.md
```

Os "gates" não são enfeite: o workflow do Spec Kit (`.specify/workflows/speckit/workflow.yml`) declara passos `type: gate` com `options: [approve, reject]` entre as etapas. A sua constituição reforça isso por escrito.

## Como está sendo usado neste projeto

### A constituição

A constituição (`.specify/memory/constitution.md`) é o documento que **toda** etapa consulta. No nosso projeto ela tem cinco princípios:

```markdown
### I. TypeScript ESM Estrito
### II. Arquitetura em Camadas
### III. Contratos e Erros Explícitos
### IV. Testes e Qualidade Não Negociáveis
### V. Integração Segura e Simples
```

E ela vira um **checklist** dentro do plano. Veja como o nosso `plan.md` checou cada princípio antes de desenhar a solução:

```markdown
## Constitution Check

- **I. TypeScript ESM Estrito**: PASS. O plano preserva TypeScript ESM, Node.js 22, `strict: true` e os scripts existentes.
- **II. Arquitetura em Camadas**: PASS. Models/stores, services/estratégias e controllers/CLI serão separados; efeitos de LLM e banco ficarão nas bordas.
- **III. Contratos e Erros Explícitos**: PASS. Zod validará ferramentas, trace, métricas, planner e flags da arena; erros de domínio serão traduzidos nas bordas.
...
```

Esse é o ganho real: decisões como "sem `dotenv`" ou "testes sem rede" não dependem de você lembrar de pedir — elas estão na constituição e o agente é obrigado a checar.

### O que o Spec Kit instalou

```
.specify/
  memory/constitution.md          → regras permanentes (versionada: 1.0.1)
  templates/*.md                  → esqueletos de spec, plan, tasks, checklist, constituição
  scripts/powershell/*.ps1        → criam a pasta da feature, resolvem caminhos, checam pré-requisitos
  workflows/speckit/workflow.yml  → o ciclo specify → plan → tasks → implement com gates
  init-options.json               → "ai": "copilot", "script": "ps", "feature_numbering": "sequential"
.github/
  copilot-instructions.md         → instruções sempre carregadas pelo Copilot (stack, comandos, fluxo)
  skills/speckit-*/SKILL.md       → um comando por skill (specify, plan, tasks, implement, ...)
```

Repare que os comandos do Spec Kit chegam ao Copilot como **Agent Skills** (`SKILL.md`) — exatamente o formato que você estudou em [`04-skills`](../../../modulo03-mcp-na-pratica/04-skills/docs/formato-skill-md.md). Não é um mecanismo novo: é um uso novo de algo que você já conhece.

### O que já produzimos (checkpoint)

```
specs/001-nucleo-raciocinio-opspilot/
  spec.md                    → 4 user stories, FR-001..FR-015, SC-001..SC-006, edge cases
  checklists/requirements.md → checklist de qualidade da spec (tudo marcado)
  plan.md                    → contexto técnico, Constitution Check, estrutura de src/
  research.md                → 5 decisões com alternativas rejeitadas
  data-model.md              → Service, Alert, Incident, TraceEvent, métricas, regras de validação
  contracts/strategy.md      → interface comum, resultado, tipos de TraceEvent
  contracts/tools.md         → contrato das tools e do store
  contracts/arena-cli.md     → flags e formato de saída da arena
  quickstart.md              → comandos de validação pós-implementação
  tasks.md                   → T001..T036 em 7 fases, com marcação [P] de paralelismo e [USn]
```

### Como ler o `tasks.md`

Cada tarefa tem um formato fixo: `- [ ] T0NN [P] [USn] descrição com caminho exato`. O `[P]` indica que ela pode rodar em paralelo com outras `[P]` da mesma fase (arquivos diferentes, sem dependência); o `[USn]` amarra a tarefa à user story da spec. As fases seguem o grafo de dependência declarado no topo:

```text
Setup -> Foundation -> US1 -> US2 -> US3 -> US4 -> Polish
```

O `/speckit.implement` vai marcando `[x]` conforme executa — e a regra de ouro, que está tanto na constituição quanto no próprio `tasks.md`, é: **nenhuma tarefa é marcada como concluída antes de o comando de validação dela passar**.

## Os outros comandos do Spec Kit

Além dos quatro principais, o Spec Kit instalou skills de apoio que você pode usar entre as etapas:

| Skill | Quando usar |
| --- | --- |
| `speckit-clarify` | Depois do `specify`, quando a spec ainda tem ambiguidade — faz perguntas e grava as respostas na spec |
| `speckit-checklist` | Gera checklists de qualidade da spec (foi ele que produziu `checklists/requirements.md`) |
| `speckit-analyze` | Depois do `tasks`, cruza spec × plan × tasks procurando inconsistências e lacunas **antes** de implementar |
| `speckit-converge` | Depois (ou no meio) do `implement`: compara o código com spec/plan/tasks e acrescenta ao `tasks.md` o que ainda falta |
| `speckit-taskstoissues` | Transforma o `tasks.md` em issues do GitHub |
| `speckit-constitution` | Cria ou emenda a constituição (com versão e data de emenda) |

> Sugestão para o próximo passo: rodar `/speckit.analyze` antes do `/speckit.implement`. O [roteiro-de-implementacao.md](./roteiro-de-implementacao.md) já lista algumas inconsistências que um analyze provavelmente apontaria (ex.: o `data-model.md` fala em `serviceId`, enquanto o seed de referência usa `service` como nome).

## Por que isso importa para agentes

O Spec Kit resolve um problema que fica gritante quando quem escreve o código é um agente: **deriva de escopo**. Um agente sem spec tende a "ajudar" demais — criar arquivos que ninguém pediu, trocar biblioteca, pular testes. Com spec + plano + tarefas versionados:

- o escopo está escrito e é revisável num diff;
- o agente tem critério objetivo de "pronto" (cada tarefa tem validação);
- mudanças de escopo precisam passar de volta pela spec ("se o escopo mudar, atualize a especificação e o plano antes de continuar").

O snapshot de referência do curso mostra o outro lado: lá, o `UNIDADE.md` registra que o `POST /chat` "estava planejado para fechar a U2, mas foi commitado junto com a persistência da U3" e que "o roteiro cita a spec como `001-nucleo-raciocinio`; o nome real ficou em inglês". Specs versionadas deixam esses desvios **visíveis** — não os impedem, mas não deixam que passem despercebidos.

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Constituição | [`.specify/memory/constitution.md`](../.specify/memory/constitution.md) | Os 5 princípios e o "Development Workflow" com os gates |
| Workflow com gates | [`.specify/workflows/speckit/workflow.yml`](../.specify/workflows/speckit/workflow.yml) | Passos `type: gate` entre specify, plan e tasks |
| Instruções sempre carregadas | [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) | Stack, convenções e o fluxo em 4 etapas |
| Comandos como skills | [`.github/skills/`](../.github/skills/) | Um `SKILL.md` por comando `speckit-*` |
| Spec | [`specs/001-nucleo-raciocinio-opspilot/spec.md`](../specs/001-nucleo-raciocinio-opspilot/spec.md) | User stories com "Independent Test" e requisitos FR-xxx |
| Constitution Check | [`specs/001-nucleo-raciocinio-opspilot/plan.md`](../specs/001-nucleo-raciocinio-opspilot/plan.md) | PASS por princípio, antes e depois do design |
| Tarefas | [`specs/001-nucleo-raciocinio-opspilot/tasks.md`](../specs/001-nucleo-raciocinio-opspilot/tasks.md) | Fases, `[P]`, `[USn]` e critérios de conclusão |
