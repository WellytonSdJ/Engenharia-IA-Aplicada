# Documentação — Dev Instructions Agents

Documentação de estudo do projeto `03-dev-instructions-agents`, terceiro do módulo 3 (MCP na Prática).

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura ordenada, o que estamos vendo e por quê, mapa dos arquivos |
| [custom-agents-copilot.md](./custom-agents-copilot.md) | O formato `.agent.md` do GitHub Copilot: frontmatter, campos (`description`, `name`, `tools`, `model`, `mcp-servers`), corpo em linguagem natural |
| [pipeline-playwright-agents.md](./pipeline-playwright-agents.md) | Como os três agentes `playwright-test-planner` → `playwright-test-generator` → `playwright-test-healer` colaboram para produzir e manter uma suíte de testes |
| [glossario.md](./glossario.md) | Todos os termos novos deste projeto — referência rápida |

---

## Contexto do projeto

Quatro arquivos `.agent.md` em `.github/agents/`, sem nenhum código executável. Definem agentes de desenvolvimento embutidos no GitHub Copilot: um agente genérico (`developer`) e um pipeline de três agentes especializados em testes Playwright (`planner` → `generator` → `healer`).
