# Text-to-Cypher — O Núcleo do Sistema

## O problema central

O usuário faz uma pergunta em português. O banco de dados só entende Cypher. O LLM precisa fazer a ponte — mas sem guardrails, o LLM gera código inválido, usa propriedades que não existem, ou retorna objetos em vez de valores planos.

Este documento explica como os três nós centrais resolvem esse problema:
`cypherGenerator` → `cypherExecutor` → `cypherCorrection`

---

## cypherGenerator — como o LLM gera Cypher

### Ingredientes do prompt

O gerador não recebe apenas a pergunta. Recebe um prompt composto de quatro partes:

**1. Schema real do banco (buscado em tempo real)**

```typescript
// neo4jService.ts
const schema = await graph.schema
// Resultado: string com nós, relacionamentos e propriedades do banco
```

O schema é injetado no prompt a cada requisição. Se o banco mudar (novo nó, nova propriedade), o gerador se adapta automaticamente.

**2. Contexto de negócio (`salesContext.ts`)**

Regras que o LLM não consegue inferir do schema:

```
- Alunos só têm PROGRESS em cursos com PURCHASED.status = "paid"
- Receita: filtrar por status = "paid", excluir status = "refunded"
- Um par aluno-curso tem no máximo uma compra e um registro de progresso
- Sempre use CASE para contagem condicional
```

**3. Restrições de sintaxe Cypher**

Problemas específicos com o modelo usado foram mitigados com regras explícitas:

```
- Use elementId() em vez de id()
- Nunca use COUNT{} — use COUNT() com WITH
- Use EXISTS{} para verificações de existência
- Sempre use alias AS em todos os campos retornados
- Retorne valores planos — nunca objetos ou arrays aninhados
- Máximo de 3 hops de relacionamento
- Retorne plain text — sem markdown, sem backticks
```

**4. Exemplos de queries (few-shot)**

O prompt inclui 5 exemplos com pergunta → Cypher:

```
"Liste todos os cursos"
→ MATCH (c:Course) RETURN c.name AS course

"Quais alunos compraram o curso X?"
→ MATCH (s:Student)-[:PURCHASED {status: "paid"}]->(c:Course {name: "X"})
   RETURN s.name AS student, s.email AS email

"Distribuição de receita por curso"
→ MATCH (s:Student)-[p:PURCHASED {status: "paid"}]->(c:Course)
   RETURN c.name AS course,
          SUM(p.amount) AS totalRevenue,
          COUNT(s) AS studentCount
   ORDER BY totalRevenue DESC
```

### Saída validada por Zod

```typescript
// queryAnalyzer.ts
const CypherQuerySchema = z.object({
  query: z.string(), // Cypher pura, sem markdown
})
```

O LLM não pode retornar nada fora desse schema — a estrutura é garantida antes de continuar.

---

## cypherExecutor — validar antes de executar

### Por que validar com EXPLAIN?

Executar uma query inválida diretamente no Neo4j pode ser lento e retornar erros de baixa qualidade para o LLM corrigir. O `EXPLAIN` verifica a sintaxe e o plano de execução sem tocar nos dados:

```typescript
// neo4jService.ts
async validateQuery(query: string): Promise<string | null> {
  const session = driver.session()
  await session.run(`EXPLAIN ${query}`)
  return null // null = sem erro
}
```

Se o `EXPLAIN` passar, a query é executada normalmente. Se falhar, o erro vai direto para o `cypherCorrectionNode`.

### Tratamento de resultados vazios

Resultados vazios não são erros — são dados válidos. O executor não dispara a correção quando `dbResults` é vazio. O `analyticalResponseNode` trata esse caso com uma resposta explicativa.

---

## cypherCorrection — o loop de autocorreção

### Por que autocorrigir?

LLMs cometem erros previsíveis ao gerar Cypher:

- Usar `COUNT{}` (sintaxe inválida) em vez de `COUNT()` com `WITH`
- Referenciar variáveis fora do escopo após `WITH`
- `ORDER BY` em campo que pode ser NULL sem `NULLS LAST`
- Usar propriedades que não existem no schema

Em vez de retornar um erro para o usuário, o sistema tenta corrigir automaticamente.

### Como o loop funciona

```
cypherExecutor detecta erro
    ↓
(correctionAttempts < maxCorrectionAttempts=1)?
    ↓ sim
cypherCorrection recebe: query original + erro + pergunta original
    ↓
LLM gera correctedQuery + explanation
    ↓
cypherExecutor tenta executar a query corrigida
    ↓
(ainda com erro?)
    ↓ sim, mas correctionAttempts >= 1
analyticalResponse recebe o erro e gera resposta explicativa
```

O limite de uma tentativa de correção (`maxCorrectionAttempts: 1` em `config.ts`) evita loops infinitos. Se a correção falhar, o sistema abandona e responde ao usuário com uma explicação do problema.

### O prompt de correção

O `cypherCorrectionNode` injeta o contexto completo para depuração:

```
Query que falhou:
  MATCH (s:Student) RETURN COUNT{(s)-[:PURCHASED]->()} AS total

Erro retornado:
  SyntaxException: Invalid input '{': expected ...

Pergunta original:
  Quantas compras cada aluno fez?

Regras de correção:
  - COUNT{} não é suportado → use WITH + COUNT()
  - Agregações devem ser separadas com WITH
```

O LLM retorna `correctedQuery` (a versão corrigida) e `explanation` (o que foi mudado e por quê).

---

## O ciclo completo em código

```typescript
// graph.ts — edges condicionais após cypherExecutor
const routeAfterExecutor = (state: GraphState) => {
  if (state.needsCorrection && (state.correctionAttempts ?? 0) < config.maxCorrectionAttempts)
    return 'correction'
  if (state.isMultiStep && (state.currentStep ?? 0) < (state.subQuestions?.length ?? 0))
    return 'generator'
  return 'response'
}
```

A lógica de roteamento vive em um único lugar — as funções de edge — e não nos nós. Os nós apenas setam flags no state (`needsCorrection`, `isMultiStep`); quem decide para onde ir é o grafo.

---

## Referências no projeto

| Conceito | Arquivo |
| --- | --- |
| Geração de Cypher | [src/graph/nodes/cypherGeneratorNode.ts](../src/graph/nodes/cypherGeneratorNode.ts) |
| Validação e execução | [src/graph/nodes/cypherExecutorNode.ts](../src/graph/nodes/cypherExecutorNode.ts) |
| Autocorreção | [src/graph/nodes/cypherCorrectionNode.ts](../src/graph/nodes/cypherCorrectionNode.ts) |
| Prompts de geração | [src/prompts/v1/cypherGenerator.ts](../src/prompts/v1/cypherGenerator.ts) |
| Prompts de correção | [src/prompts/v1/cypherCorrection.ts](../src/prompts/v1/cypherCorrection.ts) |
| Contexto de negócio | [src/prompts/v1/salesContext.ts](../src/prompts/v1/salesContext.ts) |
| Serviço Neo4j | [src/services/neo4jService.ts](../src/services/neo4jService.ts) |
