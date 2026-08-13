# 04 — Skills

Estudo do formato **Agent Skills** — uma forma de empacotar capacidades para agentes de IA que é alternativa (e complementar) ao MCP. Este subprojeto, ao contrário dos demais do módulo, **não tem código executável**: é uma coleção de skills instaladas via CLI, cujo conteúdo em si é o material de estudo.

## O que é uma "Skill"

Uma Skill é um pacote de **instruções em linguagem natural e contexto** (arquivos Markdown, principalmente um `SKILL.md`) que o próprio agente decide carregar em tempo de execução, com base numa descrição curta declarada no arquivo. Diferente de uma tool de MCP, uma skill não expõe uma função que o agente *chama* com parâmetros estruturados — ela é texto que o agente *lê* e passa a seguir como instrução, geralmente combinado com ferramentas genéricas que o agente já possui (rodar comandos de shell, ler/escrever arquivos).

### Skills vs. MCP/tools

| | MCP / Tools | Skills |
| --- | --- | --- |
| O que expõe | Funções com schema de entrada/saída, chamadas via protocolo (JSON-RPC sobre STDIO/HTTP) | Instruções em Markdown + arquivos de referência, carregados como contexto |
| Quem executa a ação | O servidor MCP, fora do processo do agente | O próprio agente, usando as ferramentas genéricas que já tem (shell, leitura/escrita de arquivo) |
| Como é descoberta | Handshake do protocolo lista as tools disponíveis | Frontmatter (`name` + `description`) de cada `SKILL.md`; o agente decide carregar com base na descrição bater com o pedido do usuário |
| Necessita servidor/processo rodando? | Sim (o servidor MCP) | Não — são apenas arquivos no sistema de arquivos do agente |
| Empacotamento/distribuição | Pacote npm que roda como servidor MCP | Pacote de arquivos instalado por um gerenciador de skills (ex: `npx skills`), versionado por um lockfile |

Este subprojeto documenta o lado "Skills" desse contraste — o mesmo contraste que o [README do módulo 03](../README.md) descreve como "outra forma de empacotar capacidades para agentes, em contraste com tools/MCP".

## Skills encontradas neste subprojeto

Todas instaladas em `.agents/skills/`, uma pasta por skill, cada uma com um `SKILL.md` na raiz e (quando necessário) arquivos de referência auxiliares.

| Skill | Fonte (`skills-lock.json`) | O que faz |
| --- | --- | --- |
| [`ffmpeg`](./.agents/skills/ffmpeg/SKILL.md) | `digitalsamba/claude-code-video-toolkit` | Receitas de linha de comando FFmpeg/ffprobe para conversão, corte, compressão, velocidade e otimização de vídeo/áudio para múltiplas plataformas (YouTube, Twitter/X, LinkedIn, web) e para projetos Remotion |
| [`find-skills`](./.agents/skills/find-skills/SKILL.md) | `vercel-labs/skills` | Ensina o agente a buscar e instalar outras skills usando a Skills CLI (`npx skills find` / `npx skills add`) quando o usuário pede uma capacidade que talvez já exista como skill publicada |
| [`neo4j-cypher-guide`](./.agents/skills/neo4j-cypher-guide/SKILL.md) | `tomasonjo/blogs` | Guia de sintaxe moderna de Cypher (Neo4j) para geração de queries de leitura: funções removidas/depreciadas, subqueries (`CALL`, `COUNT{}`, `COLLECT{}`), e Quantified Path Patterns (QPP) para travessias eficientes |

Os dois arquivos na raiz também fazem parte do material:

- [`refs.txt`](./refs.txt) — anotações de referência com links para `skills.sh` (o "marketplace" de skills) e para os repositórios de origem das skills instaladas.
- [`skills-lock.json`](./skills-lock.json) — lockfile do gerenciador de skills: registra, para cada skill instalada, o repositório GitHub de origem (`source`), o tipo de fonte (`sourceType`) e um hash do conteúdo (`computedHash`) — o mesmo papel que um `package-lock.json` cumpre para dependências npm, mas aplicado a pacotes de skills.

## Documentação de conceitos

Aprofundamento em [`docs/`](./docs/):

| Documento | Conteúdo |
| --- | --- |
| [00-START-HERE.md](./docs/00-START-HERE.md) | Trilha de leitura, o que estamos vendo e por quê, mapa dos arquivos |
| [formato-skill-md.md](./docs/formato-skill-md.md) | Estrutura de um `SKILL.md`: frontmatter, corpo, arquivos de `references/`, e como o agente decide carregar cada um |
| [skills-vs-mcp-e-gerenciamento.md](./docs/skills-vs-mcp-e-gerenciamento.md) | Skills como alternativa ao MCP, e como elas são instaladas/versionadas (Skills CLI, `skills-lock.json`, `skills.sh`) |
| [glossario.md](./docs/glossario.md) | Termos novos deste subprojeto |
