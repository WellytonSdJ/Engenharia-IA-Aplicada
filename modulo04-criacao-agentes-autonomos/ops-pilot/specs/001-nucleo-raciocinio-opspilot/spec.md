# Feature Specification: Núcleo de raciocínio do OpsPilot

**Feature Branch**: `001-nucleo-raciocinio-opspilot`

**Created**: 2026-09-28

**Status**: Draft

**Input**: User description: "Núcleo de raciocínio do OpsPilot com uma interface comum de estratégias, fábrica única de modelos, ferramentas mock de incidentes, estratégias ReAct e Plan-and-Execute, limites de iteração, métricas, arena e testes determinísticos."

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Executar uma estratégia de raciocínio observável (Priority: P1)

Como pessoa operadora do OpsPilot, quero enviar uma solicitação operacional para uma estratégia de raciocínio e receber uma resposta acompanhada de um trace tipado e métricas, para entender o que foi feito e avaliar o custo da execução.

**Why this priority**: O contrato comum é a base para comparar estratégias, testar o agente e integrar a arena.

**Independent Test**: Executar uma estratégia com um input determinístico e verificar que o resultado contém uma resposta, uma sequência ordenada de eventos válidos e métricas não negativas.

**Acceptance Scenarios**:

1. **Given** um input válido e uma estratégia disponível, **When** a estratégia é executada, **Then** o resultado contém `answer`, `trace` e `metrics`.
2. **Given** um trace produzido por uma estratégia, **When** seus eventos são inspecionados, **Then** cada evento possui um tipo entre `thought`, `action`, `observation`, `plan`, `critique` e `answer`.
3. **Given** um evento do tipo `action`, **When** seus dados são inspecionados, **Then** ele identifica a ferramenta utilizada e contém os argumentos validados dessa chamada.
4. **Given** uma execução concluída, **When** suas métricas são lidas, **Then** `llmCalls` e `latencyMs` são números inteiros ou decimais não negativos e representam a execução realizada.

### User Story 2 - Investigar alertas e incidentes com ferramentas controladas (Priority: P1)

Como pessoa operadora, quero consultar alertas e abrir ou resolver incidentes usando ferramentas com contratos claros, para que as estratégias possam agir sobre um estado operacional previsível.

**Why this priority**: Sem ferramentas confiáveis, as estratégias não conseguem produzir ações úteis nem demonstrar o fluxo de incidentes.

**Independent Test**: Inicializar o store com o seed padrão, listar alertas por status, abrir um incidente e resolvê-lo, verificando as mudanças no estado e os erros para entradas inválidas.

**Acceptance Scenarios**:

1. **Given** o seed padrão, **When** alertas são listados por `firing` ou `resolved`, **Then** a resposta contém somente alertas daquele status e o conjunto inicial possui 3 de cada status.
2. **Given** um serviço existente e uma severidade válida, **When** um incidente é aberto com título, serviço e severidade, **Then** um incidente persistido é criado e seu identificador é retornado.
3. **Given** um incidente aberto existente, **When** ele é resolvido pelo identificador, **Then** seu estado passa a resolvido e a operação é refletida no store.
4. **Given** argumentos ausentes, inválidos ou um identificador inexistente, **When** uma ferramenta é chamada, **Then** ela falha com um erro de domínio ou validação traduzível na borda, sem corromper o estado.

### User Story 3 - Comparar ReAct e Plan-and-Execute (Priority: P1)

Como pessoa desenvolvedora do OpsPilot, quero executar mais de uma estratégia sobre o mesmo input, para comparar seus traces, respostas, chamadas de LLM e latência em condições equivalentes.

**Why this priority**: A comparação é o objetivo operacional da arena e permite avaliar as duas formas de raciocínio.

**Independent Test**: Executar a arena com as duas estratégias sobre um input simples e verificar que cada resultado é identificado, possui trace e métricas, e respeita o limite configurado.

**Acceptance Scenarios**:

1. **Given** um input e uma lista de estratégias, **When** a arena é executada, **Then** cada estratégia roda sobre o mesmo input e produz uma saída identificada separadamente.
2. **Given** as flags `--strategies` e `--max-iterations`, **When** a arena é iniciada, **Then** elas selecionam as estratégias e limitam a execução conforme os valores fornecidos.
3. **Given** uma execução da arena, **When** a saída é impressa, **Then** ela inclui traces e métricas de cada estratégia sem misturar seus eventos.

### User Story 4 - Validar o núcleo sem rede (Priority: P2)

Como pessoa desenvolvedora, quero testar o store, as ferramentas e a formatação de traces sem chamadas de rede, para obter feedback rápido e reprodutível durante o desenvolvimento.

**Why this priority**: Testes determinísticos reduzem a dependência de credenciais, disponibilidade do provedor e variações de resposta do modelo.

**Independent Test**: Executar a suíte focada no store e na formatação de trace em ambiente sem credenciais ou acesso de rede e obter resultados repetíveis.

**Acceptance Scenarios**:

1. **Given** o ambiente sem chave de provedor, **When** os testes determinísticos são executados, **Then** eles não tentam acessar a rede e passam usando o store controlado.
2. **Given** eventos válidos e inválidos, **When** o formatador de trace é executado, **Then** os eventos válidos são formatados em ordem e entradas inválidas são rejeitadas de forma previsível.

### Edge Cases

- O input é vazio, excede o tamanho suportado ou não pode ser interpretado pela estratégia.
- A chave ou o modelo do provedor não estão configurados para uma execução que exige LLM.
- Uma ferramenta recebe argumentos inválidos, uma severidade desconhecida, um status não suportado ou um identificador inexistente.
- A estratégia atinge o limite de iterações antes de produzir uma resposta final.
- O planner produz zero passos, mais de oito passos ou uma lista com passos inválidos.
- O executor falha em um passo e o replanner precisa decidir se interrompe, registra a crítica ou continua com o restante.
- Duas execuções tentam alterar o mesmo incidente; o estado final deve permanecer válido e consistente.
- O seed é executado mais de uma vez; a operação deve ser segura para desenvolvimento e não criar duplicatas inesperadas.
- O provedor de LLM demora, falha ou retorna uma resposta incompatível com o formato esperado.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: O sistema MUST oferecer uma interface comum de estratégia de raciocínio com nome e operação `run(input)` que retorne resposta, trace e métricas.
- **FR-002**: O sistema MUST validar eventos de trace e aceitar somente os tipos `thought`, `action`, `observation`, `plan`, `critique` e `answer`.
- **FR-003**: Eventos `action` MUST carregar o nome da ferramenta e seus argumentos; eventos dos demais tipos MUST carregar dados suficientes para serem formatados e auditados.
- **FR-004**: O sistema MUST reportar as métricas `llmCalls` e `latencyMs` para cada execução.
- **FR-005**: O sistema MUST disponibilizar uma fábrica única de modelos que leia `OPENROUTER_API_KEY` e `OPENROUTER_MODEL`, use a base URL compatível com OpenRouter e configure temperatura zero.
- **FR-006**: O sistema MUST disponibilizar as ferramentas `list_alerts(status)`, `open_incident(title, service, severity)` e `resolve_incident(id)` com schemas Zod para entrada e saída.
- **FR-007**: O sistema MUST fornecer um store operacional pré-populado com 5 serviços e 6 alertas variados, sendo 3 `firing` e 3 `resolved`.
- **FR-008**: O sistema MUST fornecer um script explícito para executar o seed inicial e permitir verificar o estado criado.
- **FR-009**: O sistema MUST manter uma fronteira de persistência compatível com MySQL e Sequelize, sem impedir o uso do store em memória para mocks, arena e testes determinísticos.
- **FR-010**: A estratégia ReAct MUST usar o agente ReAct pré-construído do LangGraph, conectar as ferramentas disponíveis e capturar o trace completo da execução.
- **FR-011**: A estratégia Plan-and-Execute MUST executar um planner que produza uma lista estruturada de passos, um executor que processe um passo por vez com as ferramentas e um replanner que revise os passos restantes após cada execução.
- **FR-012**: A estratégia Plan-and-Execute MUST aceitar no máximo 8 passos por plano e encerrar quando não houver passos restantes.
- **FR-013**: Toda estratégia MUST respeitar o limite de iterações configurado e contabilizar as chamadas de LLM realizadas.
- **FR-014**: A arena MUST executar uma ou mais estratégias sobre o mesmo input, imprimir os traces e as métricas e aceitar as opções `--strategies` e `--max-iterations`.
- **FR-015**: O sistema MUST incluir testes determinísticos, sem rede, para o store e para a formatação do trace.

### Key Entities

- **ReasoningStrategy**: contrato comum que identifica uma estratégia e executa um input operacional.
- **TraceEvent**: evento tipado que representa uma etapa observável do raciocínio ou da execução de uma ferramenta.
- **ReasoningMetrics**: métricas de chamadas de LLM e latência de uma execução.
- **Service**: serviço monitorado pelo OpsPilot.
- **Alert**: sinal operacional associado a um serviço, com status `firing` ou `resolved`.
- **Incident**: ocorrência aberta ou resolvida a partir de uma ação operacional.
- **ReasoningStrategyRun**: resultado comparável produzido por uma estratégia para um mesmo input.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Uma execução válida sempre retorna resposta, trace ordenado e as duas métricas obrigatórias, ou um erro estruturado sem estado parcialmente corrompido.
- **SC-002**: O seed inicial produz exatamente 5 serviços e 6 alertas, distribuídos em 3 alertas `firing` e 3 `resolved`.
- **SC-003**: 100% dos testes do store e da formatação de trace passam sem rede e produzem o mesmo resultado em execuções repetidas.
- **SC-004**: Nenhuma execução Plan-and-Execute ultrapassa 8 passos planejados ou o limite de iterações configurado.
- **SC-005**: A arena consegue apresentar resultados separados de pelo menos duas estratégias para o mesmo input, incluindo trace e métricas por estratégia.
- **SC-006**: Entradas inválidas das ferramentas são rejeitadas antes de alterar o estado operacional.

## Assumptions

- A implementação desta feature será feita em TypeScript ESM no runtime Node.js 22 LTS, usando as dependências já declaradas no projeto.
- O store em memória é a fonte controlada para mocks, seed, arena e testes sem rede; MySQL com Sequelize é a fronteira de persistência para o estado durável quando essa integração for ativada.
- `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` são obrigatórios somente para execuções que chamam o provedor de LLM; testes determinísticos não dependem deles.
- `thought` no trace representa um resumo observável e limitado do raciocínio da estratégia, não a exposição irrestrita de conteúdo interno do modelo.
- A severidade aceita para incidentes será definida no contrato da feature e compartilhada entre ferramentas, store e testes.
- O input da arena é um texto operacional comum para todas as estratégias; a seleção de estratégias usa nomes estáveis e documentados.
- O script de seed deve ser idempotente ou detectar dados já semeados para evitar duplicatas durante o desenvolvimento.
- A execução do seed ocorrerá somente na etapa de implementação, após a especificação, o plano e as tarefas serem aprovados.
