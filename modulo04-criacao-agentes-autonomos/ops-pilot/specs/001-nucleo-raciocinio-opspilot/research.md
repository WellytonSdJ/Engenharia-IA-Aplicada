# Research: Núcleo de raciocínio do OpsPilot

## Decision 1: Fábrica OpenRouter com ChatOpenAI

- **Decision**: Centralizar a criação do modelo em `src/agents/model.ts`, usando `ChatOpenAI` com `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `temperature: 0` e `configuration.baseURL = "https://openrouter.ai/api/v1"`.
- **Rationale**: `@langchain/openai` é a dependência instalada e OpenRouter expõe uma API compatível com OpenAI. Uma única fábrica evita configurações divergentes entre ReAct, planner, executor e replanner.
- **Alternatives considered**: Instanciar modelos em cada estratégia foi rejeitado por duplicar configuração e dificultar testes; adicionar outro SDK foi rejeitado por contrariar a stack existente.

## Decision 2: ReAct com agente pré-construído

- **Decision**: Usar `createReactAgent` de `@langchain/langgraph/prebuilt`, conectar as tools estruturadas e transformar mensagens/updates do grafo em `TraceEvent`.
- **Rationale**: A versão instalada `@langchain/langgraph` `0.2.74` exporta o agente pré-construído e suporta `stream`/`invoke` com estado de mensagens e tool calling.
- **Alternatives considered**: Implementar manualmente o loop ReAct foi rejeitado porque duplicaria lógica do LangGraph e aumentaria o risco de divergência no trace.

## Decision 3: Plan-and-Execute como StateGraph

- **Decision**: Construir um `StateGraph` com nós `planner`, `executor` e `replanner`, estado tipado e arestas condicionais. O executor processa somente o próximo passo; o replanner decide entre continuar e finalizar.
- **Rationale**: `StateGraph` é exportado pela versão instalada e representa diretamente o ciclo com estado compartilhado. O limite de 8 passos e o limite de iterações serão guards do estado, não apenas instruções de prompt.
- **Alternatives considered**: Um loop imperativo foi rejeitado para manter a estratégia compatível com a observabilidade e o modelo de grafo do LangGraph.

## Decision 4: Store in-memory e adapter Sequelize

- **Decision**: Definir um contrato de store no domínio, implementar `InMemoryOpsStore` para mocks/arena/testes e um adapter Sequelize/MySQL para persistência durável. As tools recebem o store por injeção.
- **Rationale**: Os testes exigem determinismo e ausência de rede, enquanto a feature exige fronteira MySQL com Sequelize. A injeção mantém essas necessidades separadas sem duplicar contratos de negócio.
- **Alternatives considered**: Usar apenas MySQL nos testes foi rejeitado por exigir infraestrutura externa; usar apenas memória foi rejeitado por não cumprir a fronteira de persistência solicitada.

## Decision 5: Métricas e trace

- **Decision**: Medir `latencyMs` com relógio monotônico disponível no runtime e incrementar `llmCalls` em cada invocação de modelo feita pela estratégia. O trace será montado a partir de eventos de estratégia e tool calls, com resumo controlado para `thought`.
- **Rationale**: As métricas precisam ser comparáveis entre estratégias e o trace deve ser auditável sem expor raciocínio interno irrestrito do modelo.
- **Alternatives considered**: Inferir chamadas e latência a partir de texto final foi rejeitado por ser impreciso.
