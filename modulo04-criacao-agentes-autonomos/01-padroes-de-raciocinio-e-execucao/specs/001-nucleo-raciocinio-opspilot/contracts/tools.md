# Tools Contract

## Store Boundary

As ferramentas recebem um store com operações para listar alertas, criar incidentes e resolver incidentes. O contrato é implementado por `InMemoryOpsStore` e por um adapter Sequelize/MySQL.

## Tools

### `list_alerts`

- Input: `{ status: "firing" | "resolved" }`.
- Output: lista de alertas com serviço, título, severidade e status.
- Side effect: nenhum.

### `open_incident`

- Input: `{ title: string; service: string; severity: Severity }`.
- Output: incidente criado com identificador e status `open`.
- Side effect: cria um incidente no store.

### `resolve_incident`

- Input: `{ id: string }`.
- Output: incidente atualizado com status `resolved`.
- Side effect: altera somente um incidente aberto.

Todas as entradas e saídas são validadas com Zod. Falhas de validação e domínio não devem deixar mutações parciais.
