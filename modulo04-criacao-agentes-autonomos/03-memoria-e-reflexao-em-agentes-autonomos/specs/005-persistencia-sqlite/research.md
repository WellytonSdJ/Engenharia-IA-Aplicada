# Phase 0 Research: Persistência Real de Operações

Não há `NEEDS CLARIFICATION` pendente no Technical Context — as decisões abaixo já estavam fechadas com o usuário antes do plano (ver conversa de `/speckit.specify`) ou são diretamente derivadas da constitution já emendada. Este documento registra o porquê de cada uma, para consulta futura.

## Decisão: usar `node:sqlite` (`DatabaseSync`) nativo, sem driver externo

- **Decision**: Persistência real via o módulo nativo `node:sqlite`, API síncrona (`DatabaseSync`).
- **Rationale**: Já é suportado pelo Node 22 LTS (estável desde a 22.5, sem flag experimental a partir da 23.4; o ambiente local roda Node 24). Elimina a necessidade de qualquer dependência de produção para acesso a banco de dados — alinhado à constitution (Princípio V: "use as dependências existentes antes de adicionar"). A API síncrona é adequada aqui: todo acesso de store já é local ao processo, sem round-trip de rede, e simplifica o código (sem `await` artificial em cada query).
- **Alternatives considered**:
  - `better-sqlite3` (pacote externo síncrono popular): rejeitado por adicionar uma dependência nativa compilada que o `node:sqlite` já torna redundante.
  - Manter Sequelize, só trocando o dialect para `sqlite3`: rejeitado porque a constitution emendada já exclui ORM e bancos tradicionais; manter o Sequelize manteria a complexidade de configuração (dialect, migrations) sem necessidade.
  - Driver assíncrono via `node:sqlite` com wrappers `Promise`: descartado — a interface `OpsStore` já é `async`/`Promise`-based (para ser substituível por qualquer backend), então os métodos do novo store continuam `async`, apenas internamente chamando a API síncrona do driver.

## Decisão: caminho do arquivo via `OPSPILOT_DB`, com `:memory:` em testes e bench/arena

- **Decision**: `OPSPILOT_DB` controla o caminho do arquivo; default `./data/opspilot.db` quando ausente; testes e bench/arena usam `:memory:` explicitamente (não dependem da env var).
- **Rationale**: Segue o padrão já estabelecido no projeto para configuração via ambiente nativo do Node (`--env-file-if-exists`), sem introduzir `dotenv` (Princípio V). `:memory:` em testes garante isolamento total entre execuções (User Story 3) sem precisar limpar arquivos entre testes.
- **Alternatives considered**: arquivo temporário por teste (`tmp/test-<random>.db`) — rejeitado por exigir limpeza de arquivos e ser mais lento que `:memory:` sem benefício, já que `node:sqlite` suporta `:memory:` nativamente.

## Decisão: DDL idempotente no construtor, `CHECK` para campos de domínio fechado

- **Decision**: O construtor do `SqliteOpsStore` executa `CREATE TABLE IF NOT EXISTS` para as 4 tabelas, com `CHECK` nas colunas que espelham enums do domínio (`severity`, `status` de alerts e de incidents).
- **Rationale**: Atende FR-007 e FR-012 da spec — inicialização automática sem passo manual, e nenhuma gravação fora do domínio conhecido é aceita mesmo que uma camada anterior falhe. `IF NOT EXISTS` torna seguro recriar o store (um novo processo, ou um teste novo com o mesmo arquivo) sem apagar dados existentes.
- **Alternatives considered**: sistema de migrations versionadas (ex.: numeração de scripts `0001_init.sql`) — descartado por desproporcional ao escopo (schema pequeno e estável); se o schema evoluir no futuro, migrations podem ser revisitadas em uma spec própria.

## Decisão: `Runbook` como tabela e conceito de domínio novo, 1 runbook por serviço no máximo

- **Decision**: Nova tabela `runbooks` com chave estrangeira para `services` (relação 0..1), schema Zod `RunbookSchema` em `domain.ts`, método novo na interface `OpsStore` para buscar runbook por serviço.
- **Rationale**: A spec (User Story 2) exige diferenciar "serviço sem runbook" de "serviço inexistente" — isso só é possível modelando o runbook como relação opcional (0 ou 1) para um serviço que precisa existir.
- **Alternatives considered**: embutir o conteúdo do runbook como coluna extra em `services` — rejeitado porque misturaria um conceito opcional e de conteúdo potencialmente longo dentro da tabela de serviços, e dificultaria tratar "sem runbook" de forma explícita no contrato de dados.

## Decisão: `InMemoryOpsStore` permanece para testes e bench/arena; `JsonOpsStore`/`SequelizeOpsStore` são removidos

- **Decision**: `InMemoryOpsStore` (já existente) não migra para SQLite; continua sendo instanciado por teste/execução. `JsonOpsStore` e `SequelizeOpsStore`, junto com o arquivo `src/data/ops-store.json` e as dependências `mysql`, `mysql2`, `sequelize`, `@types/sequelize`, são removidos.
- **Rationale**: Decisão já confirmada com o usuário no `/speckit.specify`. `InMemoryOpsStore` é o store determinístico usado pelo bench/arena (docs/tools-e-store-deterministico.md) — reescrevê-lo para SQLite não traria benefício e arriscaria a reprodutibilidade dos cenários C1/C2/C3 do bench. Os outros dois stores deixam de ter razão de existir quando SQLite é a única persistência real.
- **Alternatives considered**: manter `JsonOpsStore` como fallback — rejeitado pelo usuário explicitamente ("SQLite é o único store real").

## Decisão: descrições das tools revisadas com um conjunto fixo de regras de design

- **Decision**: Toda tool em `src/services/tools.ts` (existentes e a nova `consultar_runbook`) segue o mesmo conjunto de regras ao escrever `description` e schema Zod:
  1. A descrição diz **quando** usar a tool (gatilho de intenção), não só o que ela faz.
  2. Toda ambiguidade de vocabulário do usuário final (ex.: "sev2") é mapeada explicitamente para o valor de domínio aceito.
  3. Todo campo do schema Zod tem `.describe()` com uma frase que um modelo consegue usar para preencher o argumento corretamente.
  4. Campos de enum são explicitados por extenso na descrição (não só "ver schema").
  5. A descrição não promete nada que a tool não garante (sem overclaim de efeitos colaterais).
  6. Erros de domínio previsíveis (recurso não encontrado, transição inválida) são antecipados na descrição quando ajudam o modelo a se recuperar sem tentar de novo às cegas.
- **Rationale**: `open_incident` já aplica parcialmente a regra 2 (mapeamento sev1..sev4); a spec pede estender esse padrão a todas as tools e aplicar `.describe()` por campo, hoje ausente em todo o arquivo.
- **Alternatives considered**: deixar a descrição mínima e confiar só no nome dos campos — rejeitado porque é exatamente a lacuna que already causou a "dívida" conhecida em `open_incident` (comentário no código atual) e que a spec pede para sanar.
