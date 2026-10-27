# Tools Contract (revisão desta feature)

## Store Boundary

As ferramentas recebem um `OpsStore` injetado (`createOpsTools(store)`). Em produção, `defaultOpsStore` passa a ser um `SqliteOpsStore` (arquivo em `OPSPILOT_DB`, default `./data/opspilot.db`); testes e bench/arena continuam injetando `InMemoryOpsStore` ou um `SqliteOpsStore(":memory:")` conforme o cenário.

Toda `description` abaixo segue as regras de design acordadas em `research.md`: diz quando usar a tool, mapeia vocabulário ambíguo do usuário para o valor de domínio aceito, usa `.describe()` em cada campo do schema Zod, explicita os valores de enum por extenso, não promete efeito colateral que a tool não garante, e antecipa erros de domínio previsíveis quando isso ajuda o modelo a se recuperar.

## Tools

### `list_alerts`

- **Quando usar**: para inspecionar a saúde atual dos serviços ou ver o que está tocando o plantão agora — antes de decidir se abre um incidente.
- Input: `{ status?: "firing" | "resolved" | "all" }` (default `"firing"`). Campo `status` com `.describe()`: "Filtra alertas por status; 'firing' = ainda ativos, 'resolved' = já resolvidos, 'all' = todos. Default: firing.".
- Output: lista de alertas com serviço, título, severidade e status.
- Side effect: nenhum.

### `open_incident`

- **Quando usar**: quando o usuário pede explicitamente para abrir/registrar um incidente para um serviço — não para apenas relatar um alerta já existente sem pedido de abertura.
- Input: `{ title: string; service: string; severity: "low" | "medium" | "high" | "critical" }`.
  - `title`: "Título curto e descritivo do incidente, não vazio.".
  - `service`: "Nome exato do serviço afetado (ex.: 'checkout', 'payments'); use o nome como aparece na lista de serviços conhecida, não um nome aproximado.".
  - `severity`: "Severidade do incidente: critical, high, medium ou low. Se o usuário falar em sevN (plantão/ITIL), mapeie sev1=critical, sev2=high, sev3=medium, sev4=low antes de preencher este campo.".
- Output: incidente criado com identificador e status `open`.
- Side effect: cria um incidente no store.
- Errors: serviço inexistente é rejeitado sem alterar o estado; a mensagem lista os serviços conhecidos (`Known services: ...`) para o modelo tentar de novo com o nome certo.

### `list_incidents`

- **Quando usar**: para ver o que já está em andamento ou já foi resolvido antes de abrir um incidente novo (evita duplicar) ou para responder "o que está aberto agora".
- Input: `{ status?: "open" | "resolved" | "all" }` (default `"open"`). Campo `status` com `.describe()`: "Filtra incidentes por status; 'open' = ainda em andamento, 'resolved' = já encerrados, 'all' = todos. Default: open.".
- Output: lista de incidentes com serviço, título, severidade, status e datas (`createdAt`, `resolvedAt`, `summary` quando houver).
- Side effect: nenhum.

### `resolve_incident`

- **Quando usar**: quando o usuário confirma que um incidente específico (já com `id` conhecido, normalmente obtido de `open_incident` ou `list_incidents`) foi mitigado/encerrado.
- Input: `{ id: string }`. Campo `id` com `.describe()`: "Identificador exato do incidente a resolver, no formato retornado por open_incident/list_incidents (ex.: 'incident-...').".
- Output: incidente atualizado com status `resolved`.
- Side effect: altera somente um incidente aberto; chamar de novo sobre o mesmo id depois de resolvido é rejeitado, não sobrescreve.
- Errors: id inexistente ou já resolvido é rejeitado sem alterar o estado.

### `consultar_runbook` (novo)

- **Quando usar**: quando o usuário pede os passos de investigação/mitigação recomendados para um serviço (ex.: "qual o runbook de payments?", "como eu investigo isso?") — não serve para listar alertas ou incidentes.
- Input: `{ service: string }`. Campo `service` com `.describe()`: "Nome exato do serviço para buscar o runbook (ex.: 'checkout', 'payments', 'identity'); use o nome como aparece na lista de serviços conhecida.".
- Output: `{ service: string; title: string; content: string } | { service: string; found: false }` — quando o serviço existe mas não tem runbook cadastrado, a tool retorna essa forma explícita de "não encontrado" em vez de lançar erro.
- Side effect: nenhum.
- Errors: serviço inexistente (não apenas "sem runbook") é rejeitado com a mesma mensagem padrão de serviços conhecidos usada por `open_incident`.

Todas as entradas e saídas são validadas com Zod. Falhas de validação e domínio não devem deixar mutações parciais.
