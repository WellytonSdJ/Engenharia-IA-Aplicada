# 03 — Dev Instructions Agents

Instruções declarativas para agentes de desenvolvimento do GitHub Copilot, no formato `.agent.md`.

## Por que este projeto é diferente

Todos os subprojetos anteriores deste repositório (módulo 02 inteiro, e `01-multiple-mcp-tools` no módulo 03) são projetos de **código**: um `package.json`, um `src/`, um agente construído em runtime com LangChain/LangGraph, conectado a servidores MCP via subprocesso. Você roda `npm start` e um processo Node.js decide, em tempo real, quais tools chamar.

Aqui não há nada disso. Este projeto **não tem `package.json`, não tem `src/`, não tem nenhuma linha de código executável**. Ele contém apenas 4 arquivos Markdown com frontmatter, dentro de `.github/agents/`. Não construímos um agente — **configuramos** agentes que já existem embutidos no editor (GitHub Copilot). Cada arquivo `.agent.md` é lido pelo Copilot e vira um "modo" de agente selecionável no chat, com sua própria persona, seu próprio conjunto restrito de tools e seu próprio fluxo de trabalho — tudo isso escrito em linguagem natural (Markdown), sem nenhuma chamada a `createAgent`, `StateGraph` ou `MultiServerMCPClient`.

Em outras palavras: no módulo 02 e em `01-multiple-mcp-tools`, a "definição do agente" é código TypeScript que você escreve e executa. Aqui, a "definição do agente" é um arquivo de instrução que o Copilot interpreta — o equivalente, em espírito, a um system prompt com metadados, empacotado como artefato do repositório em vez de string dentro do código-fonte.

## O que tem nesta pasta

```
.github/agents/
  developer.agent.md                    → agente genérico de desenvolvimento Node.js/TypeScript
  playwright-test-planner.agent.md      → explora a aplicação e gera um plano de testes
  playwright-test-generator.agent.md    → transforma um item do plano em um arquivo .spec.ts
  playwright-test-healer.agent.md       → roda os testes, diagnostica falhas e corrige o código do teste
```

| Arquivo | O que faz |
| --- | --- |
| [`developer.agent.md`](./.github/agents/developer.agent.md) | Agente de codificação Node.js + TypeScript de propósito geral: implementa features, corrige bugs e refatora com disciplina de testes, seguindo princípios como imutabilidade, injeção de dependência e prompts de LLM sempre em arquivo separado. |
| [`playwright-test-planner.agent.md`](./.github/agents/playwright-test-planner.agent.md) | Explora uma aplicação web ao vivo via ferramentas de navegador (`browser_navigate`, `browser_snapshot`, etc.) e produz um plano de testes em Markdown, cobrindo fluxos felizes, casos de borda e cenários negativos. |
| [`playwright-test-generator.agent.md`](./.github/agents/playwright-test-generator.agent.md) | Recebe um cenário do plano de testes, executa cada passo manualmente no navegador para validar que funciona, e só então grava o arquivo `.spec.ts` correspondente com base no log real de execução. |
| [`playwright-test-healer.agent.md`](./.github/agents/playwright-test-healer.agent.md) | Roda a suíte de testes Playwright, depura cada teste que falha (seletores, timing, asserções), edita o código do teste para corrigir, e repete até a suíte passar — ou marca o teste como `test.fixme()` quando a causa não é o teste em si. |

Os três agentes `playwright-test-*` formam um **pipeline sequencial**: `planner` (o que testar) → `generator` (escrever o teste) → `healer` (consertar o que quebrar depois). O `developer.agent.md` é independente desse pipeline — é um agente de propósito geral para trabalho de código comum.

## Documentação de conceitos

Aprofundamento em [`docs/`](./docs/) — comece por [`docs/00-START-HERE.md`](./docs/00-START-HERE.md).
