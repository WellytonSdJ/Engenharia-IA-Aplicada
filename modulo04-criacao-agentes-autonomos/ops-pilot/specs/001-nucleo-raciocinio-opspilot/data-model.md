# Data Model: Núcleo de raciocínio do OpsPilot

## Service

- `id`: identificador estável.
- `name`: nome único do serviço.
- `team`: equipe responsável.
- `description`: descrição operacional.

## Alert

- `id`: identificador estável.
- `serviceId`: referência a `Service`.
- `title`: descrição curta do alerta.
- `severity`: severidade operacional.
- `status`: `firing` ou `resolved`.
- `createdAt`: instante de criação.
- `resolvedAt`: instante opcional de resolução.

**Seed**: exatamente 6 alertas, 3 `firing` e 3 `resolved`, distribuídos entre os 5 serviços.

## Incident

- `id`: identificador estável.
- `title`: título obrigatório.
- `serviceId`: referência a `Service`.
- `severity`: severidade validada.
- `status`: `open` ou `resolved`.
- `createdAt`: instante de abertura.
- `resolvedAt`: instante opcional de resolução.

**Transitions**: `open -> resolved`; uma resolução repetida ou um identificador inexistente gera erro de domínio sem mutação parcial.

## TraceEvent

- `type`: `thought | action | observation | plan | critique | answer`.
- `content`: resumo textual ou estrutura serializável do evento.
- `tool`: obrigatório quando `type = action`.
- `args`: argumentos validados quando `type = action`.
- `timestamp`: instante relativo ou absoluto para ordenação.

## ReasoningMetrics

- `llmCalls`: número não negativo de chamadas ao modelo.
- `latencyMs`: duração não negativa da execução.
- `iterations`: número de ciclos usados, quando necessário para diagnóstico.

## ReasoningResult

- `answer`: resposta final da estratégia.
- `trace`: lista ordenada de `TraceEvent`.
- `metrics`: `ReasoningMetrics`.

## Validation Rules

- Schemas Zod devem validar entradas e saídas nas fronteiras.
- `list_alerts` aceita apenas status suportado.
- `open_incident` exige título, serviço existente e severidade suportada.
- `resolve_incident` exige identificador de incidente aberto.
- Um plano não pode conter mais de 8 passos.
- Toda execução deve terminar com `answer` ou erro estruturado, respeitando o limite de iterações.
