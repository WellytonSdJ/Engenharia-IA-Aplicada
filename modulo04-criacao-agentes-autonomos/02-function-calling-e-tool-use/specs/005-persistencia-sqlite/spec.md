# Feature Specification: Persistência Real de Operações

**Feature Branch**: `005-persistencia-sqlite`

**Created**: 2026-10-14

**Status**: Implemented

**Input**: User description: "Persistencia real de operações: SQLiteOpsStore implementa a interface OpsStore existente via node:sqlite (DatabaseSync); caminho em OPSPILOT_DB (default ./data/opspilot.db); ':memory:' nos testes. 4 tabelas - services, alerts, incidents, runbooks - espelhando os tipos atuais do domínio (incidents ganha resolved_at e summary, anuláveis; DDL idempotente no construtor; CHECK em todo campo de domínio fechado (severity e status). Seed idempotente = cenário Mercadinho do mock (6 serviços, 6 alertas: 3 firing, 3 resolved; runbooks de checkout/payments/auth). Prepared statements em toda query; sem SQL concatenado. Tools novas: list_incidents (já existe, revisar descrição) e consultar_runbook(service) - descrições pelas 6 regras. Composição injeta o SqliteOpsStore; mock in-memory fica para testes e para bench. data/ no .gitignore. Revisar descrições das tools existentes pelas 6 regras. Testes ':memory:' seed, abrir/listar/resolver, filtros, CHECKS; testes das tools existentes passam a rodar sobre ':memory:'. Remover a persistência tradicional (MySQL/Sequelize) e o store em arquivo JSON — SQLite é a única persistência real do projeto."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Operações do copiloto sobrevivem a reinícios (Priority: P1)

Um responsável de plantão abre um incidente pelo copiloto, encerra a sessão (ou a API reinicia), e ao voltar consulta a lista de incidentes: o incidente aberto antes continua lá, com o mesmo estado.

**Why this priority**: É o motivo central da mudança — hoje o estado real de incidentes não sobrevive de forma confiável a um reinício do processo em produção. Sem isso, o copiloto não é confiável para uso real de plantão.

**Independent Test**: Com o processo da API rodando, abrir um incidente via tool, reiniciar o processo, listar incidentes novamente e confirmar que o incidente aberto aparece com os mesmos dados.

**Acceptance Scenarios**:

1. **Given** a API foi reiniciada após um incidente ser aberto, **When** o usuário lista incidentes abertos, **Then** o incidente criado antes do reinício aparece com título, serviço e severidade preservados.
2. **Given** um incidente foi resolvido antes do reinício, **When** o usuário lista incidentes com filtro "resolved" ou "all" após o reinício, **Then** o incidente aparece como resolvido, com a data de resolução preservada.

---

### User Story 2 - Consultar o runbook de um serviço durante um incidente (Priority: P1)

Durante uma investigação, o responsável de plantão pede ao copiloto o runbook de um serviço (por exemplo, "payments") para saber os passos recomendados de mitigação.

**Why this priority**: É uma capacidade nova que falta hoje — o agente não tem como responder "o que fazer" além de abrir/listar incidentes e alertas; o runbook fecha a lacuna entre "constatar o problema" e "agir sobre ele".

**Independent Test**: Pedir ao copiloto o runbook de um serviço com runbook cadastrado (checkout, payments ou identity) e confirmar que a resposta traz o conteúdo esperado; pedir o runbook de um serviço sem runbook cadastrado e confirmar uma resposta clara de "não encontrado".

**Acceptance Scenarios**:

1. **Given** o serviço "payments" tem um runbook cadastrado, **When** o usuário pergunta "qual o runbook de payments?", **Then** o copiloto retorna os passos do runbook de payments.
2. **Given** o serviço "notifications" não tem runbook cadastrado, **When** o usuário pergunta pelo runbook desse serviço, **Then** o copiloto informa que não há runbook para esse serviço, sem erro não tratado.
3. **Given** o usuário pede o runbook de um serviço inexistente, **When** a consulta é feita, **Then** o copiloto informa que o serviço não existe.

---

### User Story 3 - Ambiente de testes e de avaliação não depende de infraestrutura externa (Priority: P2)

Um desenvolvedor roda a suíte de testes e os cenários de bench/arena localmente, sem precisar instalar ou configurar um servidor de banco de dados.

**Why this priority**: Garante que a mudança de persistência não aumente o atrito de desenvolvimento nem introduza infraestrutura externa (ex.: servidor MySQL) que hoje não é necessária para rodar o projeto.

**Independent Test**: Rodar `npm test` e os cenários de bench em uma máquina limpa, sem qualquer banco de dados instalado, e confirmar que tudo passa usando armazenamento efêmero.

**Acceptance Scenarios**:

1. **Given** nenhum banco de dados externo está disponível na máquina, **When** o desenvolvedor roda a suíte de testes, **Then** todos os testes relacionados a persistência de operações passam normalmente.
2. **Given** um cenário de bench é executado duas vezes em sequência, **When** cada execução começa, **Then** o estado inicial do cenário é o mesmo nas duas execuções (reprodutibilidade), sem resíduo da execução anterior.

---

### User Story 4 - Carga inicial de dados é previsível e repetível (Priority: P2)

Um operador prepara o ambiente de produção e executa a carga inicial de dados (seed) do copiloto. Rodar a carga novamente (por engano ou por manutenção) não duplica nem corrompe os dados existentes além do estado inicial esperado.

**Why this priority**: Evita que um reseed acidental quebre o ambiente de produção com serviços, alertas ou runbooks duplicados.

**Independent Test**: Rodar o comando de seed duas vezes seguidas sobre o mesmo armazenamento e confirmar que o resultado final é idêntico ao de uma única execução.

**Acceptance Scenarios**:

1. **Given** o seed já foi executado uma vez, **When** o seed é executado novamente sobre o mesmo armazenamento, **Then** a quantidade de serviços, alertas e runbooks permanece a mesma da primeira execução, sem duplicatas.

### Edge Cases

- O que acontece se o usuário tentar abrir um incidente para um serviço que não existe? O sistema deve rejeitar a operação e informar os serviços válidos conhecidos.
- O que acontece se o usuário tentar resolver um incidente que já está resolvido, ou que não existe? O sistema deve rejeitar a operação com uma mensagem clara, sem alterar dados.
- O que acontece se um valor de severidade ou status fora do conjunto permitido chegar até a camada de armazenamento (por um caminho que não passou pela validação de entrada)? O armazenamento deve rejeitar a gravação, nunca persistir um valor fora do domínio conhecido.
- O que acontece se dois processos tentarem escrever no mesmo arquivo de armazenamento ao mesmo tempo (por exemplo, dois workers da API)? A operação de resolver um incidente não pode permitir que o mesmo incidente seja resolvido duas vezes de forma inconsistente.
- O que acontece se o diretório onde o arquivo de dados deveria ficar não existir ainda na primeira execução? O sistema deve conseguir inicializar o armazenamento e os dados de qualquer forma, sem exigir um passo manual de preparação.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema DEVE persistir serviços, alertas, incidentes e runbooks de forma que os dados sobrevivam ao reinício do processo da aplicação.
- **FR-002**: O sistema DEVE continuar oferecendo, sem alteração de contrato observável para quem consome as tools, as operações já existentes: listar alertas por status, abrir incidente, resolver incidente e listar incidentes por status.
- **FR-003**: O sistema DEVE oferecer uma nova capacidade de consulta de runbook por serviço, retornando o conteúdo do runbook quando ele existir para o serviço informado.
- **FR-004**: O sistema DEVE informar de forma clara quando um runbook não existe para um serviço válido, e de forma clara quando o próprio serviço informado não existe — tratando os dois casos como respostas distintas e compreensíveis.
- **FR-005**: O sistema DEVE rejeitar a abertura de incidentes para serviços inexistentes, informando ao solicitante os serviços válidos conhecidos.
- **FR-006**: O sistema DEVE rejeitar a resolução de um incidente que já esteja resolvido ou que não exista, sem alterar nenhum dado nesse caso.
- **FR-007**: O sistema DEVE impedir que qualquer severidade ou status fora dos valores de domínio conhecidos seja gravado no armazenamento, mesmo que a validação de entrada de uma camada anterior falhe ou seja contornada.
- **FR-008**: O sistema DEVE garantir que a operação de resolver um incidente seja consistente mesmo sob tentativas concorrentes de resolver o mesmo incidente — apenas uma delas deve ter efeito, as demais devem ser rejeitadas como "já resolvido".
- **FR-009**: O sistema DEVE permitir a execução de testes automatizados e dos cenários de avaliação (bench/arena) usando um armazenamento efêmero e isolado por execução, sem exigir infraestrutura externa de banco de dados.
- **FR-010**: O sistema DEVE oferecer um procedimento de carga inicial de dados (seed) que, executado múltiplas vezes sobre o mesmo armazenamento, resulte sempre no mesmo conjunto de dados iniciais (sem duplicação).
- **FR-011**: A carga inicial de dados DEVE reproduzir o cenário de referência do projeto: seis serviços (api, checkout, payments, identity, notifications, catalog), seis alertas (três em curso e três resolvidos, distribuídos entre os serviços como hoje), e runbooks cadastrados para os serviços checkout, payments e identity.
- **FR-012**: O sistema DEVE inicializar sozinho a estrutura de armazenamento necessária (sem exigir um passo manual de setup) na primeira vez que for executado em um ambiente novo, incluindo a criação de diretórios ausentes.
- **FR-013**: O local onde os dados reais de operação são armazenados DEVE ser configurável por ambiente, com um local padrão sensato quando não configurado.
- **FR-014**: Os arquivos de dados reais de operação NÃO DEVEM ser versionados no controle de versão do projeto.
- **FR-015**: O sistema NÃO DEVE mais depender de um banco de dados tradicional que exija um servidor externo (ex.: MySQL) nem de um arquivo de dados em formato JSON como forma de persistência real — a persistência passa a ser self-contained no processo da aplicação.
- **FR-016**: Cada capacidade exposta ao agente (listar alertas, abrir incidente, resolver incidente, listar incidentes, consultar runbook) DEVE ter uma descrição clara de quando deve ser usada e qual é o significado de cada parâmetro aceito, incluindo os valores possíveis quando o parâmetro aceita um conjunto fechado de opções.

### Key Entities *(include if feature involves data)*

- **Serviço**: Representa um serviço de produção monitorado (nome, equipe responsável, descrição). Já existe no domínio; não muda nesta feature.
- **Alerta**: Representa um alerta de monitoramento associado a um serviço, com severidade e status (em curso ou resolvido). Já existe no domínio; não muda nesta feature.
- **Incidente**: Representa um incidente operacional aberto para um serviço, com severidade e status (aberto ou resolvido). Passa a admitir, opcionalmente, um resumo de resolução e a data/hora em que foi resolvido.
- **Runbook**: Novo conceito. Representa o conjunto de passos recomendados de investigação/mitigação para um serviço específico. Cada serviço tem no máximo um runbook; nem todo serviço precisa ter um.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Um incidente aberto antes de um reinício da aplicação continua visível, com os mesmos dados, em 100% das verificações feitas após o reinício.
- **SC-002**: A suíte de testes automatizados do projeto roda do início ao fim em uma máquina sem qualquer banco de dados externo instalado, com a mesma taxa de sucesso de hoje.
- **SC-003**: Uma pergunta sobre o runbook de um serviço com runbook cadastrado recebe os passos corretos em 100% das consultas feitas durante a avaliação do agente (bench/arena).
- **SC-004**: Executar a carga inicial de dados duas vezes em sequência sobre o mesmo armazenamento produz exatamente a mesma contagem de serviços, alertas e runbooks nas duas vezes.
- **SC-005**: Nenhum arquivo de dados reais de operação aparece no histórico de versionamento do projeto após a mudança.

## Assumptions

- O cenário de referência ("Mercadinho") usado hoje em memória — seis serviços, seis alertas (três em curso, três resolvidos) — continua sendo o conjunto de dados inicial padrão; a única adição é um runbook para cada um dos serviços checkout, payments e identity.
- Serviços que não têm runbook cadastrado (api, notifications, catalog) são um caso esperado e não um erro — a consulta deve diferenciar "serviço sem runbook" de "serviço inexistente".
- O armazenamento efêmero usado em testes e em cenários de avaliação (bench/arena) é isolado por execução: uma execução não deve ver dados deixados por outra.
- A migração de persistência não altera o comportamento observável das capacidades já existentes (listar alertas, abrir incidente, resolver incidente, listar incidentes) do ponto de vista de quem as usa — apenas passa a ser duradoura entre reinícios.
- "Resumo de resolução" do incidente é um campo opcional que pode ficar vazio; nenhuma capacidade existente hoje exige preenchê-lo, então seu preenchimento fica fora do escopo funcional desta feature (apenas a estrutura de dados já prevê o campo para uso futuro).
