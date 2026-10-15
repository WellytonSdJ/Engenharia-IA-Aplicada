# Feature Specification: Robust Plan-and-Execute Replanning

**Feature Branch**: `003-fix-replanner-output`

**Created**: 2026-10-12

**Status**: Implemented

**Input**: User description: "Ao executar a arena com a estratégia Plan-and-Execute, uma solicitação para abrir três incidentes sev2 para os alertas checkout, payments e catalog, nessa ordem, e resolver o primeiro falha com `A continue decision cannot include replacement steps or an answer.` Corrija isso e explique o que houve."

## User Scenarios & Testing

### User Story 1 - Continuar uma execução planejada (Priority: P1)

Como operador, quero que o agente continue a execução do plano quando o replanner escolher explicitamente `continue`, para que solicitações com múltiplas ações não falhem por uma resposta estruturada ambígua do modelo.

**Why this priority**: A falha interrompe uma operação solicitada que pode conter várias ações sequenciais e impede que o operador receba o resultado da execução.

**Independent Test**: Executar a estratégia Plan-and-Execute com respostas controladas do modelo que incluam uma decisão de continuar e verificar que o plano restante é preservado e executado em ordem.

**Acceptance Scenarios**:

1. **Given** um plano com passos restantes e uma decisão válida de continuar, **When** o replanner é consultado, **Then** o plano restante continua sem ser substituído e a execução prossegue.
2. **Given** uma resposta do replanner que combina `continue` com campos de outra decisão, **When** a resposta é validada, **Then** o agente segue um contrato de saída inequívoco e não descarta conteúdo contraditório silenciosamente.
3. **Given** que o plano não tem passos restantes, **When** o replanner responde `continue`, **Then** a execução não entra em ciclo nem encerra como sucesso sem uma resposta final.
4. **Given** uma sequência que cria incidentes para vários alertas e depois resolve o primeiro, **When** Plan-and-Execute conclui a solicitação, **Then** as ações são executadas na ordem solicitada e o agente retorna uma resposta final.

### Edge Cases

- O modelo retorna campos `steps` ou `answer` incompatíveis com a decisão selecionada.
- O modelo retorna uma decisão `continue` quando não há etapas restantes.
- A etapa executada é uma ação com efeito persistente e a resposta do replanner não pode provocar a repetição dessa etapa.
- A sequência planejada atinge o limite de oito etapas acumuladas.

## Requirements

### Functional Requirements

- **FR-001**: O agente MUST definir saídas do replanner que tornem explícita a combinação permitida de decisão e conteúdo para `continue`, `adjust` e `finish`.
- **FR-002**: Uma decisão `continue` MUST preservar as etapas restantes do plano e não repetir etapas já concluídas.
- **FR-003**: O agente MUST rejeitar ou tratar explicitamente respostas contraditórias do replanner; MUST NOT ignorar silenciosamente etapas ou respostas incompatíveis com a decisão declarada.
- **FR-004**: O agente MUST preservar a ordem das ações solicitadas, os resultados das etapas concluídas e o limite atual de oito etapas acumuladas.
- **FR-005**: O agente MUST retornar uma resposta final após concluir uma solicitação com múltiplas ações, incluindo operações de criação e resolução de incidentes.

### Key Entities

- **Decisão de replanning**: Escolha explícita entre continuar o plano restante, substituí-lo ou concluir com uma resposta.
- **Plano de execução**: Sequência ordenada de etapas ainda não concluídas e etapas concluídas com seus resultados.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A execução de um cenário controlado com três criações de incidente e uma resolução conclui as quatro ações na ordem pedida e retorna uma resposta final.
- **SC-002**: Em todos os testes de contrato do replanner, combinações válidas de decisão e conteúdo seguem a ação declarada; combinações inválidas não são aceitas como sucesso.
- **SC-003**: A execução continua respeitando o limite de oito etapas acumuladas e não reexecuta etapas concluídas após uma decisão `continue`.

## Assumptions

- A decisão explícita do replanner continua sendo a autoridade para o fluxo; conteúdo conflitante não deve ser escolhido por heurística.
- O comportamento existente de validação, limite de etapas, persistência e ordenação das ferramentas deve ser preservado.
- A solicitação do usuário requer uma correção no agente e uma explicação da causa; não requer mudança no formato dos comandos da arena.
