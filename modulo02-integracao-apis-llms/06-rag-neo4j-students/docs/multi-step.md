# Multi-Step — Decomposição de Perguntas Complexas

## O problema

Algumas perguntas não podem ser respondidas com uma única query Cypher:

> "Compare a receita entre cursos com alta e baixa taxa de conclusão"

Para responder isso, você precisa:
1. Calcular a taxa de conclusão por curso
2. Calcular a receita dos cursos com alta conclusão (>70%, por exemplo)
3. Calcular a receita dos cursos com baixa conclusão

Tentar gerar uma query que faça tudo isso de uma vez é uma fonte confiável de erros — o LLM precisa coordenar múltiplas agregações, condicionais e junções em uma query só. A estratégia multi-step resolve isso dividindo o problema.

---

## O que é o queryPlanner

O `queryPlannerNode` é o primeiro nó após `extractQuestion`. Ele faz uma chamada LLM para classificar a pergunta:

```typescript
// queryAnalyzer.ts
const QueryAnalysisSchema = z.object({
  complexity: z.enum(['simple', 'complex']),
  requiresDecomposition: z.boolean(),
  subQuestions: z.array(z.string()), // geradas na mesma língua da pergunta
  reasoning: z.string(),             // explicação da decisão
})
```

### Critérios de classificação

**Simples** (uma query resolve):
- Recuperação direta de uma entidade (`"Liste todos os cursos"`)
- Filtragem básica (`"Quais alunos compraram o curso X?"`)
- Agregação única (`"Qual é a receita total?"`)

**Complexa** (precisa de decomposição):
- Comparação entre grupos (`"Compare receita entre cursos com alta e baixa conclusão"`)
- Múltiplas agregações dependentes (`"Média de receita dos top 3 cursos vs. os demais"`)
- Análise relacional multi-etapa (`"Quais alunos progrediram mais rápido após a segunda compra?"`)

O limit é `maxSubQuestions: 3` (configurável em `config.ts`). Cada sub-pergunta deve ser respondível de forma independente.

---

## Como o estado multi-step evolui

```typescript
// Após queryPlannerNode (se complexo):
{
  isMultiStep: true,
  subQuestions: [
    "Qual é a taxa média de conclusão por curso?",
    "Qual é a receita total dos cursos com taxa de conclusão > 70%?",
    "Qual é a receita total dos cursos com taxa de conclusão <= 70%?",
  ],
  currentStep: 0,
  subQueries: [],   // preenchido progressivamente
  subResults: [],   // preenchido progressivamente
}
```

```typescript
// Após step 0 (cypherGenerator + cypherExecutor):
{
  currentStep: 1,
  subQueries: ["MATCH (s:Student)-[:PROGRESS]->(c:Course) ..."],
  subResults: [[{ course: "IA Aplicada", avgProgress: 78 }, ...]],
}

// Após step 1:
{
  currentStep: 2,
  subQueries: [..., "MATCH ... WHERE progress > 70 ..."],
  subResults: [[...], [{ totalRevenue: 45000 }]],
}

// Após step 2 (último step):
{
  currentStep: 3, // >= subQuestions.length → vai para analyticalResponse
  subQueries: [..., ..., "MATCH ... WHERE progress <= 70 ..."],
  subResults: [[...], [...], [{ totalRevenue: 12000 }]],
}
```

---

## Roteamento condicional multi-step

```typescript
// graph.ts — edge após cypherExecutor
if (state.isMultiStep && state.currentStep < state.subQuestions.length)
  return 'generator' // volta para gerar a próxima sub-query
```

O `cypherGeneratorNode` detecta que está em modo multi-step pelo `currentStep`:

```typescript
// cypherGeneratorNode.ts
const isSubQuestion = state.isMultiStep && state.currentStep !== undefined

const question = isSubQuestion
  ? state.subQuestions![state.currentStep!]  // sub-pergunta atual
  : state.question!                           // pergunta completa (caso simples)
```

---

## Síntese dos resultados

Quando todos os steps terminam, o `analyticalResponseNode` recebe o contexto completo:

```typescript
// analyticalResponseNode.ts — síntese multi-step
if (state.isMultiStep && state.subResults && state.subResults.length > 0) {
  const stepsData = state.subQuestions!.map((q, i) => ({
    step: i + 1,
    question: q,
    query: state.subQueries?.[i],
    results: state.subResults![i],
  }))

  // Envia para um prompt de síntese que gera uma narrativa coerente
  return generateMultiStepSynthesis(stepsData, state.question!)
}
```

O prompt de síntese recebe cada step com sua pergunta, query e resultados, e pede ao LLM que produza uma análise que integre os dados de todos os steps em uma resposta única.

---

## Por que não uma query só?

Além da complexidade técnica, há um motivo prático: o LLM erra mais em queries complexas. Uma query que precisa calcular médias condicionais, filtrar por percentil e agregar receita ao mesmo tempo tem mais superfície de erro do que três queries simples. A decomposição reduz a chance de falha e, quando um step falha, a correção tem um escopo bem menor para trabalhar.

---

## Exemplo completo

**Pergunta**: "Quais cursos têm mais alunos que compraram mas nunca começaram?"

**queryPlanner decide**: complexo, decompor em:
1. "Quantos alunos compraram cada curso?"
2. "Quantos alunos têm progresso zero em cada curso?"
3. "Qual a proporção de alunos sem progresso por curso, ordenada do maior para o menor?"

**Execução**:
- Step 0: query conta compradores por curso → `subResults[0]`
- Step 1: query conta alunos com `progress = 0` → `subResults[1]`
- Step 2: query calcula proporção e ordena → `subResults[2]`

**analyticalResponse**: sintetiza os três resultados em:
> "O curso *Fundamentos de Python* tem a maior proporção de alunos inativos: 42% dos 31 compradores nunca iniciaram o conteúdo. Em segundo lugar, *Design para Devs* com 38%. Os cursos mais concluídos foram..."

---

## Referências no projeto

| Conceito | Arquivo |
| --- | --- |
| Análise de complexidade | [src/graph/nodes/queryPlannerNode.ts](../src/graph/nodes/queryPlannerNode.ts) |
| Schema do queryAnalyzer | [src/prompts/v1/queryAnalyzer.ts](../src/prompts/v1/queryAnalyzer.ts) |
| Geração por step | [src/graph/nodes/cypherGeneratorNode.ts](../src/graph/nodes/cypherGeneratorNode.ts) |
| Síntese multi-step | [src/graph/nodes/analyticalResponseNode.ts](../src/graph/nodes/analyticalResponseNode.ts) |
| Prompt de síntese | [src/prompts/v1/analyticalResponse.ts](../src/prompts/v1/analyticalResponse.ts) |
| Limite de sub-perguntas | [src/config.ts](../src/config.ts) |
