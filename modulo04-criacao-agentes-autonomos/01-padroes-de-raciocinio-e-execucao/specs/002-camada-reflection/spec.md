# Feature Specification: Camada de Reflection

**Feature Branch**: `002-camada-reflection`

**Created**: 2026-10-05

**Status**: Draft

**Input**: User description: "Camada reflection: withreflection(strategy, opts) decora qualquer ReasoningStrategy: executa a base; um crítico (mesmo modelo, saída estruturada {aproved, feedback}) avalia a resposta contra as observações do trace; se reprovar, regenera com o feedback no contexto; para em approved ou maxReflections(default: 2). Evento 'critique' no trace; métricas somam as chamadas extras. Arena: reflect:react e reflect:plan-and-execute"

## Escopo

Adicionar reflection como uma capacidade que pode envolver qualquer estratégia de raciocínio existente, sem substituir sua execução base. O crítico avalia a resposta usando as observações registradas durante a execução; respostas reprovadas podem ser regeneradas com o feedback do crítico até aprovação ou até o limite configurado. A arena deve permitir comparar as variantes com reflection de ReAct e Plan-and-Execute.

Fora de escopo: alterar as estratégias base, mudar as ferramentas operacionais, criar um modelo de crítica separado ou acrescentar controles de reflection além do limite máximo de ciclos.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Reforçar uma estratégia com reflection (Priority: P1)

Como pessoa desenvolvedora do OpsPilot, quero envolver qualquer estratégia de raciocínio existente em uma camada de reflection, para que uma resposta possa ser revisada contra as observações disponíveis e corrigida antes de ser apresentada.

**Why this priority**: A capacidade de envolver estratégias existentes é o núcleo da feature e habilita o restante das jornadas sem duplicar o fluxo de raciocínio.

**Independent Test**: Envolver uma estratégia controlada por respostas determinísticas, reprovar sua primeira resposta e aprovar a seguinte; verificar que a resposta final corresponde à tentativa aprovada e que o feedback chegou ao contexto da regeneração.

**Acceptance Scenarios**:

1. **Given** uma estratégia que retorna uma resposta e um trace com observações, **When** ela é envolvida pela camada de reflection, **Then** a estratégia base é executada e sua resposta é avaliada pelo crítico.
2. **Given** que o crítico aprova uma resposta, **When** a avaliação termina, **Then** essa resposta é retornada sem uma regeneração adicional.
3. **Given** que o crítico reprova uma resposta, **When** ainda há ciclos de reflection disponíveis, **Then** a estratégia gera uma nova resposta com o feedback recebido no contexto.
4. **Given** qualquer estratégia válida, **When** ela é envolvida, **Then** seu comportamento base permanece utilizável sem exigir mudanças na implementação da estratégia.

### User Story 2 - Comparar variantes refletidas na arena (Priority: P1)

Como pessoa desenvolvedora do OpsPilot, quero selecionar `reflect:react` e `reflect:plan-and-execute` na arena, para comparar as estratégias base com suas variantes que revisam e regeneram respostas.

**Why this priority**: A arena é a superfície de comparação existente e deve tornar a nova capacidade observável e acessível.

**Independent Test**: Executar a arena selecionando cada identificador refletido e confirmar que cada variante roda sua estratégia base com reflection, produzindo resultado, trace e métricas independentes.

**Acceptance Scenarios**:

1. **Given** `reflect:react` selecionado, **When** a arena executa a seleção, **Then** ReAct é executado com a camada de reflection e o resultado é identificado como a variante refletida.
2. **Given** `reflect:plan-and-execute` selecionado, **When** a arena executa a seleção, **Then** Plan-and-Execute é executado com a camada de reflection e o resultado é identificado como a variante refletida.
3. **Given** as duas variantes selecionadas, **When** a arena termina, **Then** seus traces e métricas são apresentados separadamente, sem misturar eventos de uma execução com a outra.

### User Story 3 - Inspecionar críticas e custo adicional (Priority: P2)

Como pessoa desenvolvedora ou operadora, quero ver as decisões do crítico no trace e as chamadas extras nas métricas, para entender por que uma resposta foi regenerada e qual custo a revisão acrescentou.

**Why this priority**: A visibilidade permite auditar a decisão de aprovação e comparar o custo da reflexão com a execução base.

**Independent Test**: Usar uma sequência conhecida de reprovações e aprovações e verificar a ordem dos eventos de crítica, o feedback registrado e a contagem total de chamadas.

**Acceptance Scenarios**:

1. **Given** uma crítica concluída, **When** o trace é inspecionado, **Then** contém um evento `critique` com a decisão de aprovação e o feedback correspondente.
2. **Given** que a execução inclui críticas ou regenerações, **When** as métricas são lidas, **Then** contabilizam todas as chamadas da estratégia base e as chamadas extras do crítico e das regenerações.
3. **Given** uma execução refletida, **When** sua latência é medida, **Then** ela inclui o tempo gasto na execução base, nas críticas e nas regenerações.

### Edge Cases

- O crítico reprova todas as respostas até o limite; a execução termina no limite e retorna a resposta mais recente, preservando no trace que ela não foi aprovada.
- `maxReflections` é zero; a resposta base é retornada sem crítica nem regeneração.
- O crítico aprova na primeira avaliação; nenhuma regeneração ocorre.
- A estratégia base não produz observações no trace; a crítica ainda recebe a resposta e as observações existentes, mesmo que vazias.
- Uma crítica ou regeneração falha ou retorna conteúdo que não respeita o contrato esperado; a execução reporta a falha claramente, sem convertê-la em uma aprovação ou resposta bem-sucedida.
- O feedback de uma crítica precisa permanecer associado à tentativa que o originou e não deve ser aplicado a uma variante ou execução diferente na arena.
- A estratégia termina sem uma resposta utilizável; a camada não deve inventar uma resposta aprovada.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: O sistema MUST disponibilizar `withReflection(strategy, opts)` como decorador de qualquer estratégia compatível com o contrato comum de estratégia de raciocínio.
- **FR-002**: O sistema MUST executar a estratégia base para obter a resposta e o trace que serão avaliados.
- **FR-003**: O sistema MUST avaliar cada resposta com um crítico que use o mesmo modelo configurado para a estratégia envolvida.
- **FR-004**: O crítico MUST avaliar a resposta em relação às observações disponíveis no trace da tentativa e MUST produzir uma saída estruturada com o campo booleano `aproved` e o campo textual `feedback`.
- **FR-005**: Quando `aproved` for verdadeiro, o sistema MUST encerrar a reflexão e retornar a resposta avaliada sem regenerá-la.
- **FR-006**: Quando `aproved` for falso e ainda houver ciclos disponíveis, o sistema MUST regenerar a resposta usando o feedback da crítica no contexto da estratégia.
- **FR-007**: A opção `maxReflections` MUST limitar o número de ciclos de regeneração e MUST ter valor padrão 2; o valor zero MUST permitir a execução base sem ciclos de crítica ou regeneração.
- **FR-008**: Cada avaliação do crítico MUST acrescentar ao trace um evento `critique` que registre a decisão e o feedback em ordem com os demais eventos da execução.
- **FR-009**: O resultado MUST preservar os eventos da estratégia base e incluir os eventos de reflexão na ordem real em que ocorreram.
- **FR-010**: As métricas MUST contabilizar as chamadas da estratégia base e todas as chamadas extras feitas para críticas e regenerações, sem omitir nem contar duas vezes uma chamada.
- **FR-011**: A latência reportada MUST incluir a execução base e o tempo gasto em todas as críticas e regenerações.
- **FR-012**: Quando o limite de regenerações for atingido sem aprovação, o sistema MUST retornar a resposta mais recente e deixar explícito no trace que a última crítica a reprovou.
- **FR-013**: A arena MUST disponibilizar os identificadores `reflect:react` e `reflect:plan-and-execute` para executar as variantes refletidas das estratégias correspondentes.
- **FR-014**: A arena MUST manter resultados, traces e métricas isolados para cada estratégia selecionada, incluindo as variantes refletidas.
- **FR-015**: Falhas do crítico, de validação de sua saída ou de regeneração MUST ser comunicadas como falhas da execução; não MUST ser tratadas como aprovação nem ocultadas por uma resposta aparentemente bem-sucedida.

### Key Entities

- **Estratégia refletida**: Uma estratégia existente executada com revisão e ciclos de regeneração, mantendo o resultado e o trace da estratégia envolvida.
- **Crítica**: Avaliação estruturada de uma resposta à luz das observações disponíveis, contendo decisão de aprovação e feedback.
- **Ciclo de reflection**: Uma crítica seguida, quando reprovada e dentro do limite, por uma tentativa de regeneração contextualizada pelo feedback.
- **Métricas da execução refletida**: Contagem total de chamadas e latência acumulada da execução base, das críticas e das regenerações.
- **Variante da arena**: Uma seleção identificável da arena que associa uma estratégia base a reflection sem misturar seus dados de execução.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Em 100% dos casos de aprovação, a resposta retornada é a resposta aprovada e nenhuma regeneração posterior é executada.
- **SC-002**: Com `maxReflections` em seu padrão, uma execução reprovada continuamente realiza no máximo duas regenerações após a execução base e termina com a última resposta e sua rejeição registradas.
- **SC-003**: Para qualquer sequência controlada de críticas e regenerações, o trace contém um evento de crítica por avaliação, na ordem ocorrida, e as métricas contabilizam cada chamada extra exatamente uma vez.
- **SC-004**: A arena consegue selecionar e executar separadamente ambas as variantes solicitadas, identificando corretamente sua estratégia base e mantendo traces e métricas isolados.
- **SC-005**: Um valor de limite zero resulta em uma execução base sem chamadas do crítico ou regenerações.

## Assumptions

- `maxReflections` conta regenerações permitidas depois da resposta base. Cada tentativa, incluindo a resposta base, recebe uma avaliação, portanto o limite padrão permite no máximo duas regenerações e até três avaliações do crítico.
- A chave `aproved` é mantida exatamente como especificada no contrato solicitado.
- Ao esgotar o limite sem aprovação, a resposta mais recente é retornada como melhor resultado disponível, com a reprovação final explícita no trace; o chamador não deve interpretá-la como aprovada.
- As métricas existentes de contagem de chamadas e latência permanecem compatíveis, e a reflexão acrescenta nelas o custo de todas as operações extras.
- O feedback é utilizado apenas na regeneração imediatamente subsequente à crítica que o produziu.
- Os identificadores da arena representam variantes distintas; selecionar uma estratégia base sem prefixo continua executando-a sem reflection.
