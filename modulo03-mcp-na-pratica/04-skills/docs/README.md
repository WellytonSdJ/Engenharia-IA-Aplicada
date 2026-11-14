# Documentação — Skills

Documentação de estudo do subprojeto `04-skills`, o quarto do módulo 3 (MCP na Prática). Diferente dos outros subprojetos do módulo, este não tem código — é uma coleção de skills instaladas, estudadas como material em si.

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura ordenada, o que estamos vendo e por quê, mapa dos arquivos |
| [01-formato-skill-md.md](./01-formato-skill-md.md) | Estrutura de um `SKILL.md` (frontmatter + corpo), arquivos de `references/`, e como o agente decide carregar cada um |
| [02-skills-vs-mcp-e-gerenciamento.md](./02-skills-vs-mcp-e-gerenciamento.md) | Skills como alternativa ao MCP para empacotar capacidades, e como são instaladas/versionadas (Skills CLI, lockfile) |
| [03-glossario.md](./03-glossario.md) | Todos os termos novos deste subprojeto — referência rápida |

---

## Contexto do projeto

Três skills instaladas em `.agents/skills/`, cada uma resolvendo um domínio bem diferente:

- **`ffmpeg`** — receitas de linha de comando para processamento de vídeo/áudio
- **`find-skills`** — meta-skill que ensina o agente a descobrir e instalar outras skills
- **`neo4j-cypher-guide`** — guia de sintaxe moderna de Cypher para geração de queries Neo4j

Mais os arquivos `refs.txt` (anotações de origem) e `skills-lock.json` (lockfile do gerenciador de skills).

---

**➡️ Próximo passo: [00 · Por onde começar](./00-START-HERE.md)**
