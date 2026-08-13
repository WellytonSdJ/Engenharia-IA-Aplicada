# Por onde começar

Este projeto é o quarto do módulo 3 (MCP na Prática). Se você está chegando agora, leia nesta ordem.

---

## O que estamos vendo e por quê

> Estamos vendo como estender um agente **sem escrever código nem subir um servidor** — só arquivos de texto que o próprio agente decide ler.

Nos subprojetos anteriores deste módulo, estender o que um agente sabe fazer significava conectar um **servidor MCP**: um processo separado, falando um protocolo (JSON-RPC sobre STDIO), expondo tools com schema de entrada/saída que o agente chama via tool calling. Esse é o modelo que `01-multiple-mcp-tools` usa com os servidores `filesystem` e `mongodb-mcp-server`.

**Skills** resolvem o mesmo problema — "dar ao agente uma capacidade que ele não tinha" — de outro jeito: um pacote de arquivos Markdown (`SKILL.md` +, opcionalmente, arquivos de referência auxiliares) que descreve, em linguagem natural, como fazer algo. Não há servidor, não há protocolo, não há schema de parâmetros. O agente lê a descrição da skill (`name` + `description` no frontmatter), decide se ela é relevante para o pedido do usuário, carrega o conteúdo como contexto adicional e segue as instruções usando as ferramentas genéricas que já tem — normalmente shell e leitura/escrita de arquivos.

```
MCP (01, 05...):  processo servidor + protocolo + tools com schema → agente CHAMA uma função
Skills (aqui):    arquivo Markdown com instruções → agente LÊ e SEGUE as instruções
```

Este subprojeto não tem código para rodar. O "produto" é a própria pasta `.agents/skills/`, com três skills reais instaladas por um gerenciador de skills (evidenciado pelo `skills-lock.json`), cada uma cobrindo um domínio de conhecimento diferente: processamento de vídeo (FFmpeg), descoberta de outras skills, e geração de queries Cypher modernas para Neo4j.

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [formato-skill-md.md](./formato-skill-md.md) | A peça central: como um `SKILL.md` é estruturado (frontmatter + corpo), e como o agente decide quando carregar a skill inteira ou só um arquivo de `references/`. Usa as 3 skills instaladas como exemplo real. |
| 2 | [skills-vs-mcp-e-gerenciamento.md](./skills-vs-mcp-e-gerenciamento.md) | Compara skills com MCP/tools ponto a ponto, e explica como as skills chegaram nesta pasta — a Skills CLI, o `skills-lock.json`, o `refs.txt` e o site `skills.sh`. |
| 3 | [glossario.md](./glossario.md) | Referência rápida de todos os termos novos. Consulte quando encontrar algo que não reconhece. |

---

## Mapa dos arquivos

```
.agents/skills/ffmpeg/SKILL.md                          → skill de processamento de vídeo/áudio com FFmpeg
.agents/skills/ffmpeg/reference.md                       → tabelas de referência (filtros, codecs, CRF) carregadas sob demanda

.agents/skills/find-skills/SKILL.md                       → meta-skill: como buscar/instalar outras skills via Skills CLI
                                                            (não tem arquivos de references/ — instruções cabem no próprio SKILL.md)

.agents/skills/neo4j-cypher-guide/SKILL.md                → skill de geração de Cypher moderno para Neo4j
.agents/skills/neo4j-cypher-guide/references/
  deprecated-syntax.md                                    → lista completa de sintaxe removida/depreciada
  qpp.md                                                   → Quantified Path Patterns: sintaxe e padrões de performance
  subqueries.md                                            → CALL/COUNT{}/COLLECT{} subqueries e boas práticas de ordenação

refs.txt                                                   → anotações de origem (links para skills.sh e repositórios)
skills-lock.json                                            → lockfile: origem (repo GitHub) e hash de cada skill instalada

video.mp4, video_bw.mp4                                    → não fazem parte do material de estudo (ignorados)
```

---

## Como "usar" este subprojeto

Não há `npm install` nem `npm start` — não é um projeto de código. O jeito de estudar é ler os `SKILL.md` e seus arquivos de referência diretamente, comparando a estrutura entre as três skills (uma delas, `ffmpeg`, tem um arquivo `reference.md` solto na raiz da skill; `neo4j-cypher-guide` organiza suas referências numa subpasta `references/`; `find-skills` não precisa de nenhum arquivo auxiliar). Essa variação de organização é ela mesma parte do que este subprojeto documenta — o formato Skill não impõe uma estrutura rígida de arquivos além do `SKILL.md` na raiz.
