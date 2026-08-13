# O formato SKILL.md

Cada skill instalada neste subprojeto segue a mesma estrutura de arquivo: um `SKILL.md` na raiz da pasta da skill, com um **frontmatter YAML** seguido de um **corpo em Markdown**. É esse par — metadados curtos + instruções longas — que faz o formato funcionar.

## Frontmatter: o "contrato de descoberta"

As três skills abrem com o mesmo formato de frontmatter, só `name` e `description`:

```yaml
---
name: ffmpeg
description: Video and audio processing with FFmpeg. Use for format conversion, resizing, compression, audio extraction, and preparing assets for Remotion. Triggers include converting GIF to MP4, resizing video, extracting audio, compressing files, or any media transformation task.
---
```

```yaml
---
name: find-skills
description: Helps users discover and install agent skills when they ask questions like "how do I do X", "find a skill for X", "is there a skill that can...", or express interest in extending capabilities. This skill should be used when the user is looking for functionality that might exist as an installable skill.
---
```

```yaml
---
name: neo4j-cypher-guide
description: Comprehensive guide for writing modern Neo4j Cypher read queries. Essential for text2cypher MCP tools and LLMs generating Cypher queries. Covers removed/deprecated syntax, modern replacements, CALL subqueries for reads, COLLECT patterns, sorting best practices, and Quantified Path Patterns (QPP) for efficient graph traversal.
---
```

O campo que faz o trabalho pesado é `description`. Ela não é uma frase de efeito — é escrita para **casar com os gatilhos que o agente vai ver na conversa**: nomes de tarefas ("converting GIF to MP4"), frases que o usuário digitaria ("find a skill for X", "is there a skill that can..."), e contexto de uso ("Essential for text2cypher MCP tools"). É só com base nesse texto — sem executar nada, sem handshake de protocolo — que o agente decide se aquela skill é relevante para o pedido atual e carrega o `SKILL.md` inteiro no contexto.

Isso contrasta com a descoberta de tools MCP, onde o cliente lista as tools disponíveis perguntando ao servidor (uma chamada real do protocolo) e recebe de volta um schema JSON de cada uma. Aqui a "listagem" é estática: os `SKILL.md` já existem no sistema de arquivos do agente, e a decisão de carregar é uma leitura de texto, não uma chamada de rede.

## Corpo: instruções, não uma API

Depois do frontmatter, o corpo do `SKILL.md` é português (ou inglês) corrido — comandos de exemplo, tabelas de referência rápida, checklists, árvores de decisão. Não há um schema de "parâmetros de entrada". A skill `ffmpeg` traz literalmente comandos de shell prontos para copiar (`ffmpeg -i input.gif ...`); a skill `neo4j-cypher-guide` traz queries Cypher de exemplo com comentários "// WRONG" / "// CORRECT"; a skill `find-skills` traz uma sequência de passos ("Step 1: Understand What They Need" → "Step 4: Offer to Install") com os comandos exatos da Skills CLI a rodar.

O agente que carrega a skill não está recebendo uma função para chamar — está recebendo know-how para aplicar com as ferramentas genéricas que já tem (executar comandos de shell, ler/escrever arquivos, responder em texto).

## Arquivos de referência: carregamento em dois níveis

Duas das três skills têm arquivos auxiliares além do `SKILL.md`, mas cada uma resolve isso de um jeito:

- **`ffmpeg/reference.md`** — um único arquivo solto na raiz da skill, com tabelas de filtros, codecs, CRF e containers. O `SKILL.md` principal não menciona explicitamente quando carregá-lo (ele funciona como material de apoio geral, consultado quando o comando pronto do `SKILL.md` não é suficiente).

- **`neo4j-cypher-guide/references/`** — uma subpasta com três arquivos (`deprecated-syntax.md`, `qpp.md`, `subqueries.md`), e o próprio `SKILL.md` documenta explicitamente **quando carregar cada um**, numa seção dedicada:

  ```markdown
  ## When to Load Reference Documentation

  ### references/deprecated-syntax.md
  - Migrating queries from older Neo4j versions
  - Encountering syntax errors with legacy queries
  - Need complete list of removed/deprecated features

  ### references/subqueries.md
  - Working with CALL subqueries for reads
  - Using COLLECT or COUNT subqueries
  ...

  ### references/qpp.md
  - Optimizing variable-length path queries
  - Need early filtering during traversal
  - Working with paths longer than 3-4 hops
  ```

Isso mostra o padrão de **carregamento em dois níveis** (progressive disclosure): o `SKILL.md` sozinho já é suficiente para casos comuns (contém uma versão resumida de cada tópico — ex: já traz exemplos de QPP e de subqueries inline), e só busca o arquivo de `references/` correspondente quando o caso é mais específico (migração de sintaxe legada completa, otimização de travessias longas, etc.). Isso evita carregar todo o conteúdo de referência no contexto do agente o tempo todo — só o `SKILL.md` (mais curto) é lido para a decisão inicial, e os arquivos de referência (mais longos e detalhados) entram só quando o próprio guia aponta para eles.

A skill `find-skills` não tem pasta de referências — todo o conteúdo (categorias de skills, exemplos de comandos, dicas de busca) cabe no próprio `SKILL.md`, sem necessidade de um segundo nível.

## Feedback e proveniência dentro do próprio arquivo

A skill `ffmpeg` termina com uma seção "Feedback & Contributions" que instrui o próprio agente a atualizar o arquivo (`.claude/skills/ffmpeg/SKILL.md`) se o usuário disser "improve this skill", e aponta para o repositório de origem (`github.com/digitalsamba/claude-code-video-toolkit`) para abrir um PR. Isso reforça que uma skill é só um arquivo de texto sob controle de versão — editável tanto pelo agente quanto por humanos, e distribuída como qualquer outro artefato versionado.
