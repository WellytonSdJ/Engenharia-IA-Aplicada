# Tools Contract

## Store Boundary

As ferramentas recebem um store com operações para listar alertas, criar incidentes e resolver incidentes. No runtime, o store persiste em `src/data/ops-store.json`; testes podem injetar `InMemoryOpsStore` ou um arquivo temporário isolado.

## Tools

### `list_alerts`

- Input: `{ status: "firing" | "resolved" }`.
- Output: lista de alertas com serviço, título, severidade e status.
- Side effect: nenhum.

### `open_incident`

- Input: `{ title: string; service: string; severity: Severity }`.
- Output: incidente criado com identificador e status `open`.
- Side effect: cria um incidente no store.
- Errors: serviço inexistente é rejeitado sem alterar o estado; a mensagem lista os serviços conhecidos (`Known services: ...`) para o agente se corrigir. A descrição da tool mapeia `sev1=critical`, `sev2=high`, `sev3=medium`, `sev4=low`.

### `list_incidents`

- Input: `{ status: "open" | "resolved" | "all" }`, default `open`.
- Output: lista de incidentes com serviço, título, severidade, estado e datas.
- Side effect: nenhum.

### `resolve_incident`

- Input: `{ id: string }`.
- Output: incidente atualizado com status `resolved`.
- Side effect: altera somente um incidente aberto.

Todas as entradas e saídas são validadas com Zod. Falhas de validação e domínio não devem deixar mutações parciais.
