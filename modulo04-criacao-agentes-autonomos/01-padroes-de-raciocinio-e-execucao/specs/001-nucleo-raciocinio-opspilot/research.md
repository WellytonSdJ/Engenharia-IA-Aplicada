# Research: Núcleo de raciocínio do OpsPilot

## Decision 1: Fábrica OpenRouter com ChatOpenAI

- **Decision**: Centralizar a criação do modelo em `src/agents/model.ts`, usando `ChatOpenAI` com `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `temperature: 0` e `configuration.baseURL = "https://openrouter.ai/api/v1"`.
- **Rationale**: `@langchain/openai` é a dependência instalada e OpenRouter expõe uma API compatível com OpenAI. Uma única fábrica evita configurações divergentes entre ReAct, planner, executor e replanner.
- **Alternatives considered**: Instanciar modelos em cada estratégia foi rejeitado por duplicar configuração e dificultar testes; adicionar outro SDK foi rejeitado por contrariar a stack existente.

## Decision 2: ReAct com agente pré-construído

- **Decision**: Usar `createReactAgent` de `@langchain/langgraph/prebuilt`, conectar as tools estruturadas e transformar mensagens/updates do grafo em `TraceEvent`.
- **Rationale**: A versão instalada `@langchain/langgraph` `0.2.74` exporta o agente pré-construído e suporta `stream`/`invoke` com estado de mensagens e tool calling.
- **Alternatives considered**: Implementar manualmente o loop ReAct foi rejeitado porque duplicaria lógica do LangGraph e aumentaria o risco de divergência no trace.

## Decision 3: Plan-and-Execute como StateGraph com estado `done`

- **Decision**: Construir um `StateGraph` com nós `planner`, `executor` e `replanner`, estado `input`, `plan`, `done` e `answer`, e arestas condicionais. O executor seleciona uma ferramenta disponível, executa somente o próximo passo e acrescenta `[step, result]` a `done`. O replanner escolhe explicitamente `continue`, `adjust` ou `finish`.
- **Rationale**: `StateGraph` é exportado pela versão instalada e representa diretamente o ciclo com estado compartilhado. O máximo de oito passos é aplicado a `done.length + plan.length`, mesmo depois de ajustes, além do limite de iterações da arena.
- **Alternatives considered**: Um loop imperativo foi rejeitado para manter a estratégia compatível com a observabilidade e o modelo de grafo do LangGraph.

## Decision 4: Arquivo JSON persistente para o store operacional

- **Decision**: Usar `src/data/ops-store.json` como fonte de dados única do runtime, com leitura/escrita validadas e atualizações atômicas. Manter `InMemoryOpsStore` apenas como adapter leve de teste; testes de persistência usam arquivos temporários injetados.
- **Rationale**: O agente deve ler o seed atual e persistir incidentes novos sem serviço externo, enquanto testes ficam independentes do JSON compartilhado.
- **Alternatives considered**: Usar MySQL/Sequelize foi adiado, pois não é necessário para o estágio atual; manter apenas a memória foi rejeitado porque novos itens deixariam de persistir entre execuções.

## Decision 5: Métricas e trace

- **Decision**: Medir `latencyMs` com relógio monotônico disponível no runtime e incrementar `llmCalls` em cada invocação de modelo feita pela estratégia. O trace será montado a partir de eventos de estratégia e tool calls, com resumo controlado para `thought`.
- **Rationale**: As métricas precisam ser comparáveis entre estratégias e o trace deve ser auditável sem expor raciocínio interno irrestrito do modelo.
- **Alternatives considered**: Inferir chamadas e latência a partir de texto final foi rejeitado por ser impreciso.
