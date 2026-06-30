# RAG com Grafos

## O que é RAG

RAG (Retrieval-Augmented Generation) é a técnica de buscar informações relevantes antes de gerar uma resposta, enriquecendo o contexto do modelo.

A ideia central é simples: o LLM não precisa "saber" os dados — ele precisa receber os dados certos no prompt e raciocinar sobre eles.

```
[sem RAG]  pergunta → LLM → resposta (baseada em treinamento)
[com RAG]  pergunta → buscar dados → LLM(dados + pergunta) → resposta (baseada em fatos reais)
```

O tipo de busca varia conforme o banco de dados:

| Tipo de banco | Busca feita | Melhor para |
| --- | --- | --- |
| Vetorial (pgvector, Pinecone) | Similaridade semântica | Documentos, textos, FAQs |
| Relacional (Postgres, MySQL) | SQL gerado por LLM | Dados estruturados com tabelas |
| **Grafo (Neo4j)** | **Cypher gerado por LLM** | **Relacionamentos entre entidades** |

---

## Por que grafo para dados de vendas

O domínio deste projeto — alunos comprando cursos — é naturalmente um grafo:

```
(Student)-[:PURCHASED {status, amount, paymentMethod, paymentDate}]->(Course)
(Student)-[:PROGRESS {progress: 0-100}]->(Course)
```

Perguntas sobre esse domínio envolvem relacionamentos:

- "Quais cursos são comprados juntos com frequência?" → encontrar padrões de co-compra entre estudantes
- "Quais alunos compraram mas nunca começaram?" → cruzar PURCHASED com ausência de PROGRESS
- "Compare receita entre cursos com alta e baixa conclusão" → relacionar PURCHASED.amount com PROGRESS.progress

Em SQL, essas queries exigiriam múltiplos JOINs e subconsultas. No Neo4j, os relacionamentos são cidadãos de primeira classe — você os navega diretamente.

---

## Neo4j e Cypher

**Neo4j** é um banco de dados orientado a grafos. Ao invés de tabelas com colunas, ele armazena nós (entidades) e relacionamentos (arestas com propriedades).

**Cypher** é a linguagem de query do Neo4j. A sintaxe espelha a estrutura do grafo visualmente:

```cypher
// "Encontre alunos que compraram cursos e têm 100% de progresso"
MATCH (s:Student)-[:PURCHASED {status: "paid"}]->(c:Course)
MATCH (s)-[:PROGRESS {progress: 100}]->(c)
RETURN s.name AS aluno, c.name AS curso
```

O padrão `(nó)-[:RELACIONAMENTO]->(nó)` reflete diretamente como os dados estão armazenados.

---

## O schema do banco neste projeto

```
Nós:
  Student { id: UUID, name, email, phone }
  Course  { name, url }

Relacionamentos:
  (Student)-[:PURCHASED { status, paymentMethod, paymentDate, amount }]->(Course)
  (Student)-[:PROGRESS  { progress: 0-100 }]->(Course)
```

**Regras de negócio importantes** (injetadas nos prompts via `salesContext.ts`):

- Alunos só têm `PROGRESS` em cursos que compraram com `status = "paid"`
- Receita: sempre filtrar por `status = "paid"` e excluir `status = "refunded"`
- Um aluno pode ter comprado e pedido reembolso — o `PROGRESS` pode existir mesmo com `status = "refunded"`

---

## Como o schema é injetado nos prompts

A diferença fundamental de RAG com grafos para RAG com vetores é que aqui não buscamos documentos — buscamos o **schema do banco** para guiar a geração de código.

```typescript
// neo4jService.ts
async getSchema(): Promise<string> {
  const graph = await this.getGraph()
  return graph.schema // schema como string descritiva
}

// cypherGeneratorNode.ts
const schema = await neo4jService.getSchema()
// schema é injetado no prompt → LLM sabe quais nós, relacionamentos e propriedades existem
```

O LLM nunca "chuta" a estrutura do banco — ele a recebe dinamicamente a cada requisição. Se o schema mudar, os prompts se atualizam automaticamente.

---

## RAG clássico vs. RAG com grafos

| Aspecto | RAG com vetor | RAG com grafo (este projeto) |
| --- | --- | --- |
| **O que é buscado** | Chunks de texto similares | Schema do banco + resultados de query |
| **Como é buscado** | Embedding + similaridade cosseno | Cypher gerado e executado pelo LLM |
| **Formato da resposta intermediária** | Texto relevante | Registros estruturados (linhas/colunas) |
| **Falha possível** | Não encontrar chunk relevante | Query Cypher inválida ou sem resultados |
| **Mitigação** | Ajustar threshold de similaridade | Validar + autocorrigir a query |

---

## Referências no projeto

| Conceito | Arquivo |
| --- | --- |
| Schema e conexão Neo4j | [src/services/neo4jService.ts](../src/services/neo4jService.ts) |
| Seed de dados fictícios | [data/seedHelpers.ts](../data/seedHelpers.ts) |
| Regras de negócio nos prompts | [src/prompts/v1/salesContext.ts](../src/prompts/v1/salesContext.ts) |
| Injeção de schema no gerador | [src/graph/nodes/cypherGeneratorNode.ts](../src/graph/nodes/cypherGeneratorNode.ts) |
| Docker com Neo4j | [docker-compose.yaml](../docker-compose.yaml) |
