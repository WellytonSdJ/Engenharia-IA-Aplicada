# Glossário

Referência rápida. Para profundidade, vá ao documento específico de cada conceito.

Termos de MCP (MCP, MCP Server, MCP Client, STDIO transport, Tool, Tool calling) já cobertos no [glossário de `05-safeguard-prompt-injection`](../../../modulo02-integracao-apis-llms/05-safeguard-prompt-injection/docs/glossario.md) e no [glossário de `01-multiple-mcp-tools`](../../01-multiple-mcp-tools/docs/glossario.md) não são repetidos aqui.

---

## Custom agents do GitHub Copilot

| Termo | Definição |
| --- | --- |
| **Custom agent** | Modo de agente do GitHub Copilot definido por um arquivo `.agent.md`, com persona, ferramentas e workflow próprios, selecionável no chat do editor. |
| **`.agent.md`** | Arquivo Markdown com frontmatter YAML + corpo em linguagem natural, salvo em `.github/agents/`, que o Copilot interpreta como definição de um custom agent. |
| **Frontmatter** | Bloco YAML entre `---` no topo do arquivo, com os metadados de configuração do agente (`description`, `name`, `tools`, `model`, `mcp-servers`). |
| **`description` (frontmatter)** | Texto exibido na UI de seleção de agente. Pode incluir um exemplo formal de como invocar o agente (visto em `playwright-test-generator.agent.md`, com placeholders de parâmetros). |
| **`tools` (frontmatter)** | Lista que restringe quais ferramentas o agente pode chamar — o mecanismo de sandboxing de cada custom agent. Pode listar categorias amplas (`'read'`, `'edit'`, `'execute'`) ou tools individuais (`playwright/browser_click`). |
| **`model` (frontmatter)** | Fixa o modelo de LLM usado por aquele agente específico, sobrescrevendo o padrão da sessão do Copilot. |
| **`mcp-servers` (frontmatter)** | Declara um servidor MCP próprio do agente diretamente no `.agent.md` (`type`, `command`, `args`, `tools` expostas) — o agente sobe/conecta a esse servidor ao ser usado. |
| **Corpo do `.agent.md`** | Texto em Markdown após o frontmatter; funciona como o system prompt do agente — persona, missão, regras do que fazer/não fazer, critérios de sucesso, workflow passo a passo. |

---

## Pipeline de teste Playwright

| Termo | Definição |
| --- | --- |
| **`playwright-test-planner`** | Agente que explora uma aplicação web ao vivo e produz um plano de testes em Markdown (cenários com passos, resultado esperado, estado inicial). |
| **`playwright-test-generator`** | Agente que executa manualmente cada passo de um cenário do plano no navegador, lê o log gerado e só então grava o arquivo `.spec.ts` correspondente. |
| **`playwright-test-healer`** | Agente que roda a suíte de testes, depura cada falha, edita o código do teste para corrigir e repete até a suíte passar (ou marca `test.fixme()`). |
| **Seed file** | Arquivo de setup referenciado no plano de testes (`**Seed:**`), usado como ponto de partida do cenário antes de gravar o teste. |
| **`generator_setup_page` / `planner_setup_page`** | Tools que preparam a página antes de qualquer outra ação de navegador — primeiro passo obrigatório de `generator` e `planner`, respectivamente. |
| **`generator_read_log` / `generator_write_test`** | Tools do `playwright-test-generator`: a primeira lê o log da execução manual do cenário; a segunda grava o `.spec.ts` com base nesse log. |
| **`test_run` / `test_debug`** | Tools do `playwright-test-healer`: a primeira roda a suíte inteira para achar falhas; a segunda depura um teste específico que falhou, pausando em pontos de erro. |
| **`test.fixme()`** | Marcação usada pelo `playwright-test-healer` para pular um teste quando há alta confiança de que o teste está correto e o problema está em outro lugar — em vez de forçar uma correção artificial. |
| **`playwright run-test-mcp-server`** | Comando (`npx playwright run-test-mcp-server`) que sobe o servidor MCP consumido por `playwright-test-planner` e `playwright-test-healer`, declarado no bloco `mcp-servers` de cada `.agent.md`. |

---

## Agente `developer`

| Termo | Definição |
| --- | --- |
| **`developer.agent.md`** | Custom agent genérico de codificação Node.js/TypeScript, independente do pipeline de testes Playwright — implementa features, corrige bugs e refatora com disciplina de testes. |
| **Success Criteria** | Seção do `developer.agent.md` que define quando uma tarefa está concluída: sem erros de tipo, testes relevantes e suíte completa passando, critério de aceite do usuário atendido. |
| **Won't do** | Seção do `developer.agent.md` listando limites explícitos do agente (não usar `eval`, não prosseguir com requisito ambíguo, não criar `types.ts` nem `index.ts` de re-export, entre outros). |
