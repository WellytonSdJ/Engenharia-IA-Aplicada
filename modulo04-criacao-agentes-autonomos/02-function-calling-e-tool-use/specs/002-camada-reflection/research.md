# Research: Camada de Reflection

## Decisão: Decorar o contrato de estratégia existente

- **Decision**: Implementar `withReflection(strategy, opts)` como uma `ReasoningStrategy` que delega execuções à estratégia recebida e mantém o nome da variante `reflect:<strategy.name>`.
- **Rationale**: `ReasoningStrategy` já padroniza `run(input, options)` e resultados com resposta, trace e métricas; a composição evita duplicar fluxos de ReAct e Plan-and-Execute.
- **Alternatives considered**: Adicionar lógica de crítica separadamente em cada estratégia base (duplicaria o ciclo e tornaria extensões inconsistentes); modificar a interface comum (desnecessário para anexar feedback a um input textual).

## Decisão: Structured output com o modelo já injetado

- **Decision**: Usar `BaseChatModel.withStructuredOutput` com schema Zod e passar à estratégia e ao crítico a mesma instância resolvida a partir de `ReasoningOptions.model` ou da factory existente.
- **Rationale**: Plan-and-Execute já usa `withStructuredOutput` para planner, executor e replanner; essa é a prática local e mantém a crítica determinística sob modelo injetado.
- **Alternatives considered**: Criar um segundo modelo independente (viola o requisito e pode causar divergência de configuração); analisar JSON de texto manualmente (duplica validação e é menos confiável).
- **Evidence**: `src/agents/plan-and-execute.ts` usa `options.model.withStructuredOutput(schema).invoke(...)`; `src/agents/model.ts` oferece `resolveModel` para a factory única e injeção.

## Decisão: Avaliar pedido e observações, sem usar outros eventos como evidência

- **Decision**: Extrair somente conteúdo de eventos `observation` do trace para o bloco de evidências e passar também o pedido original e a resposta candidata.
- **Rationale**: Corresponde ao exemplo fornecido e torna verificável o limite de evidência do crítico; pensamentos, planos e ações não são resultados observados.
- **Alternatives considered**: Enviar o trace inteiro ao crítico (permite avaliar por eventos que não confirmam fatos); enviar apenas a resposta (não permite confrontar a execução com evidências).

## Decisão: Feedback como contexto de regeneração

- **Decision**: Acrescentar o feedback da crítica reprovadora a uma nova entrada textual, com o pedido original claramente preservado; manter o pedido original para futuras avaliações.
- **Rationale**: A interface genérica não expõe um canal separado de revisão e aceita apenas uma string, portanto o bloco contextual funciona com todas as estratégias sem alterar o contrato.
- **Alternatives considered**: Adicionar campo `feedback` a `ReasoningOptions` e modificar cada estratégia (mais invasivo e demanda atualizar contratos); substituir o pedido original pelo feedback (perde o objetivo operacional).

## Decisão: Limite zero e contagem

- **Decision**: `maxReflections` é um inteiro não negativo que conta regenerações, default 2. Com zero, retorna-se o resultado base sem crítica; com valor N, cada resposta produzida (base e até N regenerações) é avaliada.
- **Rationale**: Preserva a semântica de “parar em approved ou maxReflections” e os critérios da especificação; a resposta final após reprovação continua auditável.
- **Alternatives considered**: Contar toda crítica como uma reflexão (torna a resposta base sem avaliação ou limita regenerações a N-1); permitir somente tentativas aprovadas como finais (descarta uma resposta real em limite esgotado e muda o contrato da estratégia).

## Decisão: Contabilizar custo incremental

- **Decision**: Somar `llmCalls` e `latencyMs` reportados por cada execução de estratégia e adicionar uma chamada e a duração medida para cada invocação do crítico.
- **Rationale**: Métricas do núcleo já são não negativas e cada estratégia registra suas próprias chamadas; acrescentar apenas chamadas de crítica no wrapper evita contagem duplicada.
- **Alternatives considered**: Substituir as métricas pelo último resultado (omite tentativas anteriores); estimar chamadas pelo número de mensagens do trace (trace não representa necessariamente uma chamada por evento).
