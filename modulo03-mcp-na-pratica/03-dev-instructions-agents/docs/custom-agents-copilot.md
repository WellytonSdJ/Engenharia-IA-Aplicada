# Custom agents do GitHub Copilot (`.agent.md`)

## O formato

Um custom agent do Copilot é um arquivo Markdown com extensão `.agent.md`, salvo em `.github/agents/`. Ele tem duas partes:

1. **Frontmatter YAML** (entre `---`): metadados que configuram o agente — quem ele é e o que ele pode tocar.
2. **Corpo em Markdown**: instruções em linguagem natural que funcionam como o system prompt do agente — persona, missão, regras, formato de saída esperado.

Não há compilação nem execução própria: o Copilot lê o arquivo, aplica o frontmatter como configuração de sessão, e usa o corpo como contexto de instrução para o modelo por trás do chat.

## Campos do frontmatter observados nos 4 arquivos

| Campo | Presente em | O que faz |
| --- | --- | --- |
| `description` | todos | Descrição do agente exibida na UI de seleção. Em `playwright-test-generator.agent.md`, a descrição inclui um exemplo de invocação completo (com placeholders `<test-suite>`, `<test-name>`, `<test-file>`, `<seed-file>`, `<body>`), servindo tanto de documentação quanto de guia de formato para quem chama o agente. |
| `name` | os 3 agentes Playwright | Identificador curto do agente (ex: `playwright-test-planner`). `developer.agent.md` não define `name` — usa o nome do arquivo. |
| `tools` | todos | Lista (ou array YAML) restringindo quais ferramentas o agente pode invocar. Varia por agente: `developer.agent.md` usa categorias amplas (`'vscode'`, `'execute'`, `'read'`, `'edit'`, `'search'`, `'web'`, `'agent'`, `'context7/*'`, `'todo'`); os agentes Playwright listam tools individuais e específicas (ex: `playwright/browser_click`, `playwright-test/test_run`). Restringir as tools é o mecanismo de "sandboxing" do agente — cada um só pode fazer o que sua lista permite. |
| `model` | `playwright-test-healer`, `playwright-test-planner` | Fixa o modelo usado por aquele agente (`Claude Sonnet 4` nos dois). `developer.agent.md` e `playwright-test-generator.agent.md` não fixam modelo — usam o padrão da sessão do Copilot. |
| `mcp-servers` | `playwright-test-healer`, `playwright-test-planner` | Declara um servidor MCP próprio do agente, com `type: stdio`, `command`, `args` e a lista de `tools` expostas (aqui, `"*"` — todas). Ambos sobem o mesmo servidor: `npx playwright run-test-mcp-server`. Isso é o ponto de contato entre este projeto e o resto do módulo (MCP): o agente do Copilot conecta a um servidor MCP exatamente como o `MultiServerMCPClient` de `01-multiple-mcp-tools`, só que a configuração fica declarada no `.agent.md` em vez de em código TypeScript. |

Note a diferença entre os dois grupos de tools Playwright: `playwright-test-planner` e `playwright-test-healer` usam o prefixo `playwright-test/` (tools do servidor MCP `mcp-servers.playwright-test` declarado no próprio arquivo), enquanto `playwright-test-generator` usa o prefixo `playwright/` sem declarar `mcp-servers` — presumindo que esse servidor já está disponível no ambiente do Copilot por outra via.

## O corpo como instrução comportamental

O corpo de cada arquivo segue a mesma lógica de um system prompt bem escrito para um agente autônomo:

- **Missão/persona em 1-2 frases** ("You are the Playwright Test Healer, an expert test automation engineer...").
- **Workflow numerado** — passos que o agente deve seguir em ordem (visto nos 3 agentes Playwright: setup → execução → leitura de log/resultado → gravação/correção).
- **Regras negativas explícitas** — o que o agente não deve fazer. Exemplos reais: `developer.agent.md` lista "Won't do" (não introduzir `eval`, não prosseguir com requisitos ambíguos, não criar `types.ts`, não criar `index.ts` de re-export); `playwright-test-healer.agent.md` instrui "Do not ask user questions, you are not interactive tool" e "Never wait for networkidle or use other discouraged or deprecated apis".
- **Critérios de sucesso/qualidade** — `developer.agent.md` define "Success Criteria" (tipos sem erro, testes relevantes passando, suíte completa passando, critério de aceite do usuário atendido); `playwright-test-planner.agent.md` define "Quality Standards" para o plano gerado.

Essa estrutura (persona + regras do que fazer/não fazer + critério de sucesso + workflow passo a passo) é o padrão comum aos 4 arquivos, ainda que cada um adapte os títulos das seções ao seu domínio.

## Por que isso importa no contexto do módulo

O módulo inteiro é sobre como dar a um LLM acesso a ferramentas e contexto externos. `.agent.md` é mais um mecanismo para isso — mas operando em uma camada diferente dos outros projetos: em vez de um agente que você programa e roda como processo próprio (`01-multiple-mcp-tools`), é uma configuração que **personaliza o agente de codificação que já roda dentro do editor**, incluindo a possibilidade de esse agente também se conectar a servidores MCP (visto em `mcp-servers` dos agentes Playwright).
