# Skills vs. MCP, e como as skills chegam nesta pasta

## Duas formas de empacotar capacidades

O restante do módulo 3 usa MCP para estender o que um agente consegue fazer: `01-multiple-mcp-tools` conecta os servidores `@modelcontextprotocol/server-filesystem` e `mongodb-mcp-server`, cada um um processo separado falando um protocolo (JSON-RPC sobre STDIO), expondo **tools** com schema de entrada/saída que o agente descobre via handshake e chama via tool calling.

Skills resolvem o mesmo problema geral — "o agente precisa saber/fazer algo que não sabe por padrão" — sem nenhuma dessas peças:

| Aspecto | MCP / Tools | Skills |
| --- | --- | --- |
| Unidade de empacotamento | Servidor (processo) expondo tools com schema JSON | Pasta com `SKILL.md` (+ arquivos de referência opcionais) |
| Descoberta | Handshake do protocolo lista tools em tempo de execução | Frontmatter (`name`, `description`) lido estaticamente; o agente casa a `description` com o pedido do usuário |
| Execução da capacidade | O servidor MCP executa a ação e devolve um resultado estruturado | O próprio agente executa, seguindo as instruções da skill, usando ferramentas genéricas (shell, arquivos) que ele já possui |
| Precisa de processo rodando? | Sim — o servidor MCP precisa estar de pé (mesmo que via STDIO lazy) | Não — é leitura de arquivo |
| Tipagem de entrada/saída | Schema explícito (JSON Schema / Zod, conforme o adapter) | Nenhuma — linguagem natural, comandos de exemplo |
| Onde mora o "como fazer" | No código do servidor MCP | No texto do `SKILL.md`, junto com o modelo que vai interpretá-lo |

Nenhuma das skills instaladas aqui (`ffmpeg`, `find-skills`, `neo4j-cypher-guide`) sobe um processo ou define uma função chamável. `ffmpeg`, por exemplo, não *é* um wrapper de FFmpeg — é um guia de comandos que o agente executa via sua própria ferramenta de shell, exatamente como executaria qualquer outro comando que soubesse de cor. A skill contribui o conhecimento ("quais flags usar para compatibilidade com Remotion", "como calcular o fator de velocidade para `atempo`"), não a execução.

Isso também explica por que uma skill pode coexistir com tools reais no mesmo agente: a descrição da skill `neo4j-cypher-guide` menciona explicitamente ser "Essential for **text2cypher MCP tools**" — ou seja, o cenário pretendido é um agente que tem uma tool MCP capaz de *rodar* Cypher contra um banco Neo4j, e usa esta skill só para saber *escrever* a query corretamente antes de chamar a tool. Skills e MCP não são mutuamente exclusivos — uma resolve "que sintaxe/comando usar", a outra resolve "como executar contra um sistema externo real".

## Como as skills chegaram nesta pasta

A presença de `skills-lock.json` mostra que estas três skills não foram escritas à mão neste repositório — foram **instaladas** por um gerenciador de skills, o mesmo papel que `package-lock.json` cumpre para dependências npm:

```json
{
  "version": 1,
  "skills": {
    "ffmpeg": {
      "source": "digitalsamba/claude-code-video-toolkit",
      "sourceType": "github",
      "computedHash": "51fa05bc18bb..."
    },
    "find-skills": {
      "source": "vercel-labs/skills",
      "sourceType": "github",
      "computedHash": "6412eb4eb3b9..."
    },
    "neo4j-cypher-guide": {
      "source": "tomasonjo/blogs",
      "sourceType": "github",
      "computedHash": "2ff7242c1f42..."
    }
  }
}
```

Cada entrada registra: a skill (`name`), o repositório GitHub de onde ela veio (`source`), o tipo de fonte (`sourceType: "github"`), e um hash do conteúdo (`computedHash`) — usado presumivelmente para detectar se a skill instalada localmente diverge da versão no repositório de origem (ex: para decidir se há update disponível).

A skill `find-skills` documenta a ferramenta de linha de comando responsável por esse fluxo: a **Skills CLI** (`npx skills`), descrita como "o gerenciador de pacotes para o ecossistema aberto de agent skills". Os comandos centrais, segundo o próprio `SKILL.md`:

```bash
npx skills find [query]     # busca skills por palavra-chave (ex: "react performance")
npx skills add <owner/repo@skill>   # instala uma skill a partir do GitHub
npx skills check            # verifica se há updates disponíveis
npx skills update           # atualiza todas as skills instaladas
npx skills init my-skill    # cria uma skill própria do zero
```

O site citado como catálogo é [skills.sh](https://skills.sh/), e é exatamente o que aparece em `refs.txt`:

```
vercel skills
    http://skills.sh/
    https://skills.sh/toolshell/skills/agent-browser
    https://skills.sh/digitalsamba/claude-code-video-toolkit/ffmpeg
```

Ou seja: `refs.txt` são anotações de pesquisa/navegação (prováveis links visitados ao escolher quais skills instalar), e `skills-lock.json` é o resultado final — o registro determinístico de proveniência das três skills que acabaram instaladas em `.agents/skills/`. A referência a `supabase.com/blog/postgres-best-practices-for-ai-agents` no mesmo arquivo sugere que o mesmo tipo de "guia de boas práticas para geração de queries" existe para outros bancos de dados além do Neo4j — o mesmo padrão que `neo4j-cypher-guide` aplica a Cypher.
