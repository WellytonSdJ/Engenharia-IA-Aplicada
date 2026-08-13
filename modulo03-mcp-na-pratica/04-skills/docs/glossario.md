# Glossário

Referência rápida. Para profundidade, vá ao documento específico de cada conceito.

Termos de MCP (MCP, MCP Server, MCP Client, STDIO transport, Tool, Tool calling, Agente) já cobertos no [glossário de `05-safeguard-prompt-injection`](../../../modulo02-integracao-apis-llms/05-safeguard-prompt-injection/docs/glossario.md) e no [glossário de `01-multiple-mcp-tools`](../../01-multiple-mcp-tools/docs/glossario.md) não são repetidos aqui.

---

## Formato Skill

| Termo | Definição |
| --- | --- |
| **Skill (Agent Skill)** | Pacote de instruções em Markdown, com um `SKILL.md` na raiz, que um agente carrega sob demanda como contexto adicional — em vez de expor uma função chamável, como faz uma tool de MCP. |
| **`SKILL.md`** | Arquivo principal de uma skill: frontmatter YAML (`name`, `description`) seguido de corpo em Markdown com instruções, exemplos e tabelas de referência. |
| **Frontmatter da skill** | Bloco YAML entre `---` no topo do `SKILL.md`, com `name` (identificador) e `description` (texto usado pelo agente para decidir se a skill é relevante para o pedido atual). |
| **`description` (skill)** | Campo do frontmatter escrito para casar com gatilhos de linguagem natural do usuário (nomes de tarefas, frases típicas) — é a base da descoberta de skills, sem handshake de protocolo. |
| **Carregamento sob demanda / progressive disclosure** | Padrão em que o `SKILL.md` traz uma versão resumida de cada tópico, e só aponta para arquivos de `references/` mais longos quando o caso de uso exige detalhe adicional (ex: seção "When to Load Reference Documentation" da skill `neo4j-cypher-guide`). |
| **Arquivo de referência (`reference.md` / `references/*.md`)** | Arquivo Markdown auxiliar de uma skill, com conteúdo mais extenso que o `SKILL.md` (tabelas completas, casos avançados), carregado apenas quando necessário. Não há um formato ou local fixo — cada skill organiza do seu jeito (`ffmpeg` usa um único `reference.md` na raiz; `neo4j-cypher-guide` usa uma subpasta `references/` com múltiplos arquivos). |

---

## Gerenciamento de skills

| Termo | Definição |
| --- | --- |
| **Skills CLI (`npx skills`)** | Ferramenta de linha de comando para descobrir, instalar e atualizar skills a partir de repositórios GitHub — o "gerenciador de pacotes" do ecossistema de agent skills. |
| **`npx skills find [query]`** | Comando que busca skills publicadas por palavra-chave. |
| **`npx skills add <owner/repo@skill>`** | Comando que instala uma skill específica de um repositório GitHub na pasta local de skills. |
| **`npx skills check` / `npx skills update`** | Comandos para verificar e aplicar atualizações das skills já instaladas. |
| **[skills.sh](https://skills.sh/)** | Catálogo/marketplace web onde skills publicadas podem ser navegadas antes de instalar. |
| **`skills-lock.json`** | Lockfile do gerenciador de skills: para cada skill instalada, registra `source` (repositório de origem), `sourceType` e `computedHash` (hash do conteúdo, usado para detectar divergência/atualização) — o análogo de um `package-lock.json` para skills. |
| **`refs.txt`** | Arquivo de anotações de pesquisa neste subprojeto, com links usados ao localizar/escolher as skills instaladas (não é lido pelo agente em tempo de execução — é material de estudo). |

---

## Skills específicas deste subprojeto

| Termo | Definição |
| --- | --- |
| **`ffmpeg` (skill)** | Skill com receitas de comando FFmpeg/ffprobe para conversão, corte, compressão e otimização de vídeo/áudio, incluindo presets específicos para publicação em YouTube, Twitter/X, LinkedIn e para uso em projetos Remotion. |
| **`find-skills` (skill)** | Meta-skill: instrui o agente a reconhecer pedidos que poderiam ser resolvidos por uma skill existente e a buscá-la/instalá-la via Skills CLI, em vez de tentar resolver a tarefa do zero. |
| **`neo4j-cypher-guide` (skill)** | Skill com guia de sintaxe Cypher moderna para o banco Neo4j: lista de funções removidas/depreciadas (ex: `id()` substituído por `elementId()`), padrões de subquery (`CALL`, `COUNT{}`, `COLLECT{}`) e Quantified Path Patterns (QPP) para travessias de grafo eficientes. |
| **QPP (Quantified Path Pattern)** | Sintaxe do Cypher moderno para casar caminhos de tamanho variável com filtragem inline durante a travessia (ex: `(a)((n WHERE n.active)-[]->(m)){1,5}(b)`), mais eficiente que o padrão antigo de relacionamento de comprimento variável (`-[*1..5]->`) porque poda caminhos durante a expansão em vez de depois. |
| **`elementId()`** | Função Cypher moderna que substitui a função removida `id()`; retorna uma string (não um inteiro) identificando um nó ou relacionamento. |
| **Remotion** | Framework citado como contexto de uso da skill `ffmpeg` (renderização de vídeo programática); os presets de FFmpeg da skill preparam assets de entrada/saída compatíveis com pipelines Remotion. |
