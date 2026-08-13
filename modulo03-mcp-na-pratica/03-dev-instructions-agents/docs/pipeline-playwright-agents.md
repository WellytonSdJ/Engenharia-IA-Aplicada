# O pipeline de agentes Playwright: planner → generator → healer

Três dos quatro arquivos `.agent.md` deste projeto (`playwright-test-planner`, `playwright-test-generator`, `playwright-test-healer`) não são agentes isolados — são estágios de um mesmo fluxo de trabalho de automação de testes end-to-end, cada um consumindo o que o anterior produziu.

## 1. `playwright-test-planner` — o que testar

Explora uma aplicação web **ao vivo**, via ferramentas de navegador do servidor MCP `playwright-test` (`browser_navigate`, `browser_snapshot`, `browser_click`, etc.), com o objetivo de mapear os fluxos de usuário e desenhar cenários de teste. Workflow descrito no arquivo:

1. `planner_setup_page` — prepara a página antes de qualquer outra tool.
2. Explora a interface via snapshot (evita tirar screenshot "a menos que seja absolutamente necessário").
3. Mapeia jornadas de usuário e caminhos críticos.
4. Desenha cenários cobrindo caminho feliz, casos de borda e tratamento de erro — cada cenário com título, passos numerados, resultado esperado e estado inicial assumido (sempre "em branco/fresco").
5. Grava o plano final como Markdown via `planner_save_plan`.

**Saída**: um arquivo de plano de testes em Markdown, com seções por funcionalidade e subseções por cenário (ex: `### 1. Adding New Todos` / `#### 1.1 Add Valid Todo`), incluindo referência a um "seed file" (arquivo de setup do teste).

## 2. `playwright-test-generator` — escrever o teste

Recebe um item específico do plano gerado pelo `planner` — a própria `description` do agente no frontmatter documenta o formato esperado da chamada, com os campos `test-suite`, `test-name`, `test-file`, `seed-file` e `body` (conteúdo do cenário: passos e expectativas).

Diferença central em relação aos outros dois agentes: ele **não apenas lê o plano e escreve código** — ele executa cada passo do cenário manualmente no navegador, em tempo real, usando as tools `playwright/browser_*`, com a descrição do passo como intenção de cada chamada de tool. Só depois de rodar tudo:

1. Lê o log gerado (`generator_read_log`).
2. Grava o teste (`generator_write_test`) com base nesse log — nunca escrevendo o `.spec.ts` "de memória" sem antes ter validado o fluxo ao vivo.

**Saída**: um arquivo `.spec.ts` com um único teste, dentro de um `describe` que reflete o item do plano, comentários por passo, e código gerado a partir das melhores práticas observadas durante a execução real (não um template estático).

## 3. `playwright-test-healer` — consertar o que quebrar

Entra em cena depois que a suíte já existe e algo passou a falhar (mudança na aplicação, seletor que quebrou, timing). Workflow:

1. `test_run` — roda toda a suíte para identificar falhas.
2. `test_debug` em cada teste falho.
3. Investiga a causa (seletor mudou, problema de timing, dependência de dados, mudança real na aplicação) usando snapshot, console, network requests.
4. Edita o teste para corrigir — nunca a aplicação.
5. Reroda para validar, repetindo até passar.

Regras notáveis do arquivo: corrige um erro por vez quando há múltiplos; nunca usa `networkidle` ou outras APIs desencorajadas/depreciadas; se a causa não for o teste em si e a confiança de que o teste está correto for alta, marca com `test.fixme()` e comenta o motivo em vez de forçar uma correção artificial; e — explicitamente — **não faz perguntas ao usuário**, porque roda de forma não interativa e deve tomar a decisão mais razoável sozinha.

**Saída**: o mesmo arquivo `.spec.ts` do generator, editado até a suíte passar (ou anotado com `test.fixme()`).

## Por que separar em três agentes

Cada estágio tem um conjunto de tools e um `model` diferentes no frontmatter (`planner` e `healer` fixam `Claude Sonnet 4`; `generator` não fixa modelo), e cada um tem uma responsabilidade única e um critério de "pronto" próprio — a mesma lógica de responsabilidade única aplicada a agentes do Copilot em vez de a funções de código. Rodar os três em sequência (planejar → gerar → curar) evita um único agente genérico tentando fazer as três coisas ao mesmo tempo, com um prompt e um conjunto de tools que teriam que cobrir exploração de UI, geração de código e depuração — três habilidades com focos de atenção diferentes.

`developer.agent.md` fica fora desse pipeline: é um agente de propósito geral para desenvolvimento comum (features, bugs, refactors), sem relação com o fluxo de testes Playwright.
