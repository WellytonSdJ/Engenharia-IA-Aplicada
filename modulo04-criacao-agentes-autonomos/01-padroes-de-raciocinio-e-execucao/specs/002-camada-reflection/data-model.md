# Data Model: Camada de Reflection

## ReflectionOptions

- `maxReflections`: inteiro não negativo; default 2. Conta regenerações após a resposta produzida pela execução inicial.

## Critique

- `aproved`: booleano obrigatório que indica se a resposta candidata está sustentada.
- `feedback`: texto obrigatório. Se `aproved` for falso, contém correções específicas e acionáveis; se verdadeiro, pode conter uma justificativa breve.
- **Evidence**: pedido original, conteúdos dos eventos `observation` da tentativa atual e a resposta candidata. Outros tipos de evento e conhecimento externo não fazem parte das evidências autorizadas.

## ReflectionAttempt

- `result`: `ReasoningResult` produzido pela estratégia para uma tentativa.
- `feedback`: feedback da crítica anterior, ausente na tentativa inicial.
- `critique`: decisão estruturada para esta resposta.
- **Transitions**: executar tentativa inicial → (limite 0: terminar) ou criticar → (aprovada: terminar) ou (reprovada e regenerações abaixo do limite: executar próxima tentativa com feedback) ou (reprovada e limite esgotado: terminar com a resposta mais recente).

## ReflectedReasoningResult

- `answer`: resposta da tentativa aprovada ou, no limite, a resposta mais recente reprovada.
- `trace`: traces das tentativas na ordem real, intercalados com eventos `critique`; cada evento usa `type: "critique"` e registra `aproved` e `feedback` em `content`.
- `metrics.llmCalls`: soma das chamadas informadas por todas as tentativas mais uma por crítica.
- `metrics.latencyMs`: soma da latência de todas as tentativas e do tempo medido nas críticas.
- `metrics.iterations`: se disponível nos resultados base, soma as iterações das execuções ocorridas.

## ArenaVariant

- `name`: `reflect:react` ou `reflect:plan-and-execute`.
- `strategy`: estratégia base correspondente envolvida pelo decorador.
- Cada variant executa e retorna trace/métricas separados através do contrato `ReasoningStrategy`.
