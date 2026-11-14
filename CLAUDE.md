# Engenharia de IA Aplicada — instruções para IAs

Leia antes de qualquer tarefa neste repositório. Vale para qualquer IA (Claude, Copilot, etc.).

## O que é este repositório

Uma **wiki de estudo e revisão** da pós em Engenharia de IA Aplicada. Cada módulo do curso tem uma pasta, e cada unidade estudada vira um **projeto com código + `docs/`**.

O objetivo é que a wiki **funcione sozinha**: quem a lê não precisa abrir o material do curso para entender os conceitos. O material base é a **fonte** de onde o conteúdo vem. A wiki não é um espelho dele, nem um índice que aponta para ele.

## Para quem escrevemos

- **Devs** (nível pleno): sabem programar e conhecem TypeScript/Node, mas estão vendo o conceito de IA pela primeira vez.
- **Leitura amigável para TDAH:** o texto precisa prender a atenção e ser fácil de retomar no meio.
  - Resumo no topo ("Em 30 segundos") antes de qualquer detalhe.
  - Uma ideia por seção, com parágrafos curtos (até ~4 linhas). Prefira tabelas, listas e diagramas ASCII a blocos de texto.
  - Toda ideia abstrata ganha uma **analogia** do mundo real e um **exemplo de código** curto e real.
  - Use marcadores visuais com moderação e sempre com o mesmo significado: ✅ checkpoint, ⚠️ pegadinha, 💡 dica, 🧪 exercício.
  - Tom de professor gente boa: direto, com humor leve, sem jargão corporativo.

## Material base (a fonte)

- **Caminho:** `C:\Users\Colaborador\Documents\projects\ESTUDO\POS_ENGENHARIA_IA\engenharia-de-software-com-ia-aplicada`
- **Sempre** rode `git pull` nesse repositório antes de consultá-lo, para garantir que está atualizado. Se o pull falhar, avise o usuário antes de seguir.
- Ele é **somente leitura** para nós: nunca commite, nunca faça push e nunca altere arquivos lá (há arquivos não rastreados, como `windows/`, que não são nossos).
- Cada unidade do material costuma ter `UNIDADE.md` (o que é novo + desvios entre roteiro e implementação), `specs/NNN-slug/` (spec, plan, research, contracts) e, às vezes, `preview.md` (material visual da aula).

### Mapa de módulos (base → wiki)

| Material base | Wiki |
| --- | --- |
| `modulo01-fundamentos-de-ia-e-llms-para-programadores` | `modulo01-fundamentos-de-ia-e-llms-para-programadores` |
| `modulo02-integracao-apis-llms` | `modulo02-integracao-apis-llms` |
| `modulo03-mcp-na-pratica` | `modulo03-mcp-na-pratica` |
| `modulo04-criacao-de-agentes-autonomos-novo` | `modulo04-criacao-agentes-autonomos` |
| `modulo05+` | ainda não iniciados |

### Módulo 04: numeração deslocada DE PROPÓSITO

A unidade **U1 (Arquitetura de Agentes de Código)** do material foi **pulada intencionalmente**. Por isso:

- **Pastas:** a unidade base N vira o projeto wiki **N−1** (U2 → `01-…`, U5 → `04-…`, U9 → `08-…`).
- **Specs:** a wiki tem uma spec extra (`003-fix-replanner-output`), então hoje a **spec base N vira a spec wiki N+1** (ex.: base `008-semantic-memory` → wiki `009-semantic-memory`). Confira sempre que nascer um projeto novo.
- **Não renumere**, não "corrija" o deslocamento e não crie conteúdo da U1, a menos que o usuário peça.

## Padrão da wiki

### Estrutura de cada projeto

```
NN-nome-do-projeto/
├── README.md            # visão geral do projeto (código)
├── specs/NNN-slug/      # specs do Spec Kit (módulo 04+)
└── docs/
    ├── README.md        # índice: tabela de docs + contexto
    ├── 00-START-HERE.md # trilha de leitura, mapa do código, como rodar
    ├── 01-<conceito>.md # um doc por conceito NOVO, numerado na ordem de leitura
    ├── 02-<conceito>.md
    └── NN-glossario.md  # sempre o último número; só termos novos
```

O `README.md` de cada módulo lista os projetos e os docs. O `README.md` da raiz mostra a árvore de todos os módulos.

### Ordem de leitura e navegação

- **Docs numerados:** o prefixo `NN-` é a posição do doc na trilha do `00-START-HERE.md`. A tabela "Trilha de leitura" segue exatamente essa numeração.
- **Rodapé obrigatório** no fim de **todo** doc (README, START-HERE, conceitos, glossário):

  ```markdown
  ---

  ⬅️ Anterior: [01 · Título](./01-x.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [03 · Título](./03-y.md)**
  ```

  - Cadeia: `README.md` → `00-START-HERE.md` → `01-…` → … → glossário.
  - O glossário aponta para o `00-START-HERE.md` do **próximo projeto**. O último projeto existente aponta para o roteiro no `README.md` do módulo.
  - Ao criar um projeto novo, atualize o rodapé do glossário do projeto anterior.
- Inserir um doc no meio da trilha **renumera** os seguintes. Corrija todos os links (docs, READMEs do módulo e da raiz, `GUIA-DE-TERMOS.md`) e os rodapés vizinhos.

### Esqueleto de um doc de conceito

```markdown
# <Conceito> — <subtítulo que diz para que serve>

> Este doc cobre a spec NNN (link para ../specs/...). Antes, leia <doc pré-requisito> (link).
>
> **Em 30 segundos**
> - 3 a 5 bullets com a ideia inteira

## O que é            → explicação do zero + analogia + tabela comparativa se ajudar
## Como está no projeto → trechos REAIS do código, com o path em comentário
## ⚠️ Pegadinhas        → o erro mais comum e como evitar
## 🧪 Exercício         → o que o material deixou como exercício/não implementado (se houver)
> ✅ **Checkpoint**: uma frase que fixa a ideia (repita ao longo do doc)
## Referências no projeto → tabela Conceito | Arquivo | O que observar
```

### Wiki que se referencia sozinha

- Links **sempre relativos e internos**: entre docs, para o código e as specs do próprio projeto, para docs de projetos anteriores (`../../01-…/docs/04-react.md`) e para o glossário.
- Conceito já explicado num projeto anterior **não é re-explicado**: linke e cubra só o que mudou, avisando isso no topo do doc.
- Todo doc novo entra no `docs/README.md` do projeto, na trilha do `00-START-HERE.md` e no `README.md` do módulo.

### Nunca cite o material original

- Nada de links, caminhos ou nomes do repositório do curso (`POS_ENGENHARIA_IA`, `engenharia-de-software-com-ia-aplicada`, `modulo04-…-novo`), nem de hashes de commit, "snapshot" ou URLs do autor do curso.
- Não escreva "como o professor fez no repositório X". Reescreva o conceito com nossas palavras, ancorado no **nosso** código e nas **nossas** specs.
- Quando o material divergir do roteiro da aula (algo planejado e não implementado), cubra o **conceito** mesmo assim, numa seção 🧪 Exercício ou "o desenho ideal × o que foi feito".

## Arquivos locais (nunca commitar)

Estes arquivos ficam em `.git/info/exclude` e nunca entram em commit:

- `GUIA-DE-TERMOS.md` (raiz): índice pessoal de termos.
- `INVENTARIO-COBERTURA.md` (raiz de cada módulo): inventário de cobertura **base × wiki**, para acompanhamento de quem mantém a wiki.

## Skills relacionadas

- `validar-cobertura-estudo`: compara uma unidade ou módulo do material base com a wiki e atualiza o `INVENTARIO-COBERTURA.md`.
- `dev-generate-doc-to-study`: gera o `docs/` de um projeto seguindo este padrão.
- `git-commit`: mensagens de commit, sempre em inglês.

## Fluxo para uma unidade nova

1. `git pull` no material base.
2. `validar-cobertura-estudo` → inventário da unidade (o que ensinar).
3. Criar o projeto wiki (no módulo 04, cópia do projeto anterior + specs novas).
4. `dev-generate-doc-to-study` usando o inventário como checklist.
5. `validar-cobertura-estudo` de novo → confirmar que os ❌ viraram ✅.
6. Commit (sem os arquivos locais) só quando o usuário pedir.
