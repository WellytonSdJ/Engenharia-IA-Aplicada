# Por onde começar

Este projeto é o terceiro do módulo 3 (MCP na Prática). Se você está chegando agora, leia nesta ordem.

---

## O que estamos vendo e por quê

> Estamos configurando agentes de IA que já vivem dentro do editor, em vez de construir um agente do zero em código.

Em `01-multiple-mcp-tools`, "definir um agente" significava escrever TypeScript: montar um `StateGraph`, conectar servidores MCP com `MultiServerMCPClient`, chamar `createAgent`. O comportamento do agente nascia de código que você escrevia, compilava (ou rodava nativamente) e executava com `npm start`.

Aqui a unidade de trabalho é outra: um arquivo Markdown com frontmatter, salvo em `.github/agents/*.agent.md`, que o **GitHub Copilot** (o assistente de IA embutido no editor) lê diretamente e transforma em um modo de agente selecionável no chat. Não existe build, não existe runtime próprio, não existe `package.json`. O "motor" que interpreta essas instruções é o próprio Copilot — o repositório só fornece a configuração declarativa: quem é o agente, quais ferramentas ele pode usar, e como ele deve se comportar.

Isso é relevante porque é uma alternativa real, e mais leve, a escrever um agente em código quando o que se precisa é **especializar o comportamento do assistente de codificação** para tarefas repetitivas do dia a dia (gerar testes, curar testes quebrados, seguir um padrão de código específico do time) — sem precisar de infraestrutura de agente, servidor MCP ou orquestração própria. A troca é: menos controle e observabilidade (não há como plugar callbacks como em `01-multiple-mcp-tools`), mas zero código para manter.

```
01-multiple-mcp-tools: agente = código TypeScript rodado por você (createAgent, StateGraph, MCP client)
03-dev-instructions-agents (aqui): agente = arquivo .agent.md interpretado pelo Copilot no editor
```

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [custom-agents-copilot.md](./custom-agents-copilot.md) | O formato do arquivo `.agent.md` em si: o que cada campo do frontmatter faz, e como o corpo em Markdown vira o "system prompt" do agente. Base para entender os outros 3 documentos. |
| 2 | [pipeline-playwright-agents.md](./pipeline-playwright-agents.md) | Como os três agentes de teste (`planner`, `generator`, `healer`) se encaixam em um fluxo sequencial, cada um lendo a saída do anterior. |
| 3 | [glossario.md](./glossario.md) | Referência rápida de todos os termos novos. Consulte quando encontrar algo que não reconhece. |

---

## Mapa dos arquivos

```
.github/agents/developer.agent.md                    → agente genérico Node.js/TypeScript (implementa, corrige, refatora)
.github/agents/playwright-test-planner.agent.md       → explora a app viva e escreve um plano de testes (.md)
.github/agents/playwright-test-generator.agent.md     → executa cada passo do plano no navegador e grava o .spec.ts
.github/agents/playwright-test-healer.agent.md        → roda a suíte, depura falhas e edita os testes até passarem
```

Não há `src/`, `package.json`, `docker-compose.yaml` ou qualquer artefato de execução — a pasta `.github/agents/` é o projeto inteiro.

---

## Como "rodar" isto

Não há comando `npm start`. Esses arquivos são consumidos pelo GitHub Copilot dentro do VS Code (ou de outro editor com suporte a custom agents): ao colocar `.agent.md` em `.github/agents/`, cada um passa a aparecer como uma opção de agente no seletor de modo do chat do Copilot. Selecionar `playwright-test-planner`, por exemplo, faz o Copilot assumir a persona e as restrições de ferramentas descritas naquele arquivo para a conversa corrente.
