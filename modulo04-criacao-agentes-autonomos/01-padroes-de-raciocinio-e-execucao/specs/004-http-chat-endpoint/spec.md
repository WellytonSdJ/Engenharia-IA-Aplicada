# Feature Specification: Endpoint HTTP de Chat

**Feature Branch**: `004-http-chat-endpoint`

**Created**: 2026-10-12

**Status**: Draft

**Input**: User description: "POST /chat em src/http/server.ts (ou padrão do express): body { message, strategy?, reflect } validado com zod; default react. 200 {answer, trace, metrics}; 400 body inválido (issues do zod); 422 stratégia desconhecida; timeout 60s -> 504. Registry em src/agents/index.ts (nome -> estratégia; reflect aplica withReflection). Teste de integração com estratégia fake determinística, sem rede"

## Escopo

Expor as estratégias de raciocínio do OpsPilot por uma interface HTTP: um cliente envia uma mensagem, escolhe opcionalmente a estratégia e se deseja reflection, e recebe a resposta com o trace e as métricas da execução. A escolha da estratégia passa a vir de um registro único de nomes, que também sabe aplicar reflection sobre a estratégia escolhida. Entradas inválidas, estratégias desconhecidas e execuções que excedem o tempo limite recebem respostas de erro distintas e previsíveis.

Fora de escopo: autenticação, rate limiting, streaming de resposta, histórico ou sessão de conversa entre requisições, cancelamento cooperativo da execução após o timeout, configuração de `maxIterations`/`maxReflections` pelo cliente e mudanças no comportamento das estratégias, da arena ou do bench.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Conversar com o agente por HTTP (Priority: P1)

Como cliente do OpsPilot, quero enviar uma mensagem por HTTP e receber a resposta do agente com o trace e as métricas, para usar o agente fora da linha de comando.

**Why this priority**: É o caminho principal da feature; sem ele não há endpoint utilizável.

**Independent Test**: Subir o servidor com uma estratégia fake determinística registrada, enviar uma mensagem válida e verificar status 200 com `answer`, `trace` e `metrics` iguais aos produzidos pela estratégia fake, sem acesso à rede.

**Acceptance Scenarios**:

1. **Given** o servidor ativo, **When** o cliente envia `{ message }` sem `strategy`, **Then** a execução usa a estratégia `react` e a resposta é 200 com `{ answer, trace, metrics }`.
2. **Given** o servidor ativo, **When** o cliente envia `{ message, strategy: "plan-and-execute" }`, **Then** a execução usa a estratégia nomeada e a resposta é 200.
3. **Given** uma execução concluída, **When** o cliente lê o corpo da resposta, **Then** `trace` e `metrics` seguem o mesmo contrato do resultado de raciocínio já usado pela arena.

### User Story 2 - Pedir reflection na requisição (Priority: P1)

Como cliente, quero indicar `reflect: true` para que a estratégia escolhida rode com a camada de reflection, para comparar ou obter respostas revisadas sem precisar de um nome de estratégia separado.

**Why this priority**: Reflection é uma capacidade já existente e o pedido a inclui explicitamente no contrato do corpo.

**Independent Test**: Com uma estratégia fake e um crítico determinístico, enviar `reflect: true` e verificar que o trace contém eventos `critique`; enviar `reflect: false` e verificar que não contém.

**Acceptance Scenarios**:

1. **Given** `reflect: true` e uma estratégia válida, **When** a requisição é processada, **Then** a estratégia roda envolvida por reflection e o trace inclui os eventos de crítica.
2. **Given** `reflect: false` ou ausente, **When** a requisição é processada, **Then** a estratégia roda sem reflection.

### User Story 3 - Receber erros claros e distintos (Priority: P1)

Como cliente, quero que cada tipo de falha tenha um status e um corpo próprios, para distinguir erro meu (corpo inválido, estratégia inexistente) de demora do agente.

**Why this priority**: Contratos de erro explícitos são exigência da constituição (princípio III) e tornam o endpoint testável.

**Independent Test**: Enviar corpos inválidos, uma estratégia desconhecida e acionar uma estratégia fake que nunca termina com timeout reduzido no teste; verificar 400, 422 e 504, respectivamente.

**Acceptance Scenarios**:

1. **Given** um corpo sem `message`, com `message` vazio ou com campos de tipo errado, **When** a requisição chega, **Then** a resposta é 400 e o corpo lista as issues de validação (caminho do campo e mensagem).
2. **Given** `strategy` com um nome que não está no registro, **When** a requisição chega, **Then** a resposta é 422 e o corpo informa o nome recebido e os nomes disponíveis.
3. **Given** uma execução que não termina dentro do limite de 60 segundos, **When** o limite é atingido, **Then** a resposta é 504 e nenhuma resposta 200 é enviada depois para a mesma requisição.

### User Story 4 - Resolver estratégias por um registro único (Priority: P2)

Como pessoa desenvolvedora do OpsPilot, quero um registro central que mapeie nomes para estratégias e aplique reflection quando pedido, para que HTTP e futuras interfaces resolvam estratégias da mesma forma.

**Why this priority**: Reduz duplicação, mas o endpoint funciona mesmo que outras interfaces ainda não adotem o registro.

**Independent Test**: Consultar o registro com um nome conhecido, com `reflect` ligado e desligado, e com um nome desconhecido; verificar estratégia base, estratégia refletida e erro de estratégia desconhecida.

**Acceptance Scenarios**:

1. **Given** o nome `react` e `reflect` falso, **When** o registro resolve, **Then** retorna a estratégia ReAct sem reflection.
2. **Given** um nome conhecido e `reflect` verdadeiro, **When** o registro resolve, **Then** retorna essa estratégia envolvida por reflection.
3. **Given** um nome desconhecido, **When** o registro resolve, **Then** sinaliza um erro de domínio de estratégia desconhecida que contém os nomes disponíveis.

### Edge Cases

- Corpo ausente, que não é JSON ou não é um objeto: resposta 400, sem executar nenhuma estratégia.
- Campos extras no corpo: são ignorados e não causam erro.
- `message` composto só de espaços: tratado como vazio e responde 400.
- `strategy` presente, mas vazio: erro de validação (400), não estratégia desconhecida (422).
- Corpo inválido **e** estratégia desconhecida ao mesmo tempo: a validação vem primeiro, então a resposta é 400.
- `strategy` com nome de variante refletida (por exemplo, `reflect:react`): não é um nome base do registro, então responde 422; reflection é pedida apenas por `reflect`.
- A estratégia falha com erro inesperado (por exemplo, limite de iterações ou falha do modelo): resposta 500 com mensagem genérica, sem stack trace, secrets ou configuração interna.
- A estratégia termina depois do timeout: o resultado tardio é descartado e não altera a resposta 504 já enviada.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: O sistema MUST aceitar `POST /chat` com corpo JSON contendo `message` (texto obrigatório, não vazio após remover espaços), `strategy` (texto opcional, não vazio quando presente) e `reflect` (booleano opcional, padrão `false`).
- **FR-002**: O corpo MUST ser validado por um schema Zod antes de qualquer execução; quando a validação falhar, a resposta MUST ser 400 com as issues do Zod (pelo menos caminho e mensagem de cada issue).
- **FR-003**: Quando `strategy` estiver ausente, o sistema MUST usar `react`.
- **FR-004**: O sistema MUST resolver a estratégia por um registro único que mapeia nomes para estratégias e, quando `reflect` for verdadeiro, MUST retornar a estratégia envolvida por `withReflection`.
- **FR-005**: O registro MUST conter, no mínimo, `react` e `plan-and-execute`.
- **FR-006**: Quando o nome não existir no registro, o sistema MUST responder 422 com o nome recebido e a lista de nomes disponíveis.
- **FR-007**: Em caso de sucesso, o sistema MUST responder 200 com `{ answer, trace, metrics }`, conforme o contrato de resultado de raciocínio existente.
- **FR-008**: Cada execução MUST ter um limite de 60 segundos; ao ser excedido, a resposta MUST ser 504 com uma mensagem de timeout, e o resultado tardio MUST ser descartado.
- **FR-009**: O limite de tempo e o registro de estratégias MUST poder ser substituídos na montagem do servidor, para que os testes usem uma estratégia fake e um timeout curto sem alterar o comportamento padrão.
- **FR-010**: Erros não previstos durante a execução MUST resultar em 500 com mensagem genérica, sem expor stack trace, secrets ou detalhes de configuração.
- **FR-011**: Os erros de estratégia desconhecida e de timeout MUST ser modelados como erros de domínio e traduzidos para status HTTP somente na borda HTTP.
- **FR-012**: O sistema MUST ter testes de integração que exercitem o endpoint por HTTP real em uma porta local efêmera, usando uma estratégia fake determinística, sem chamadas de rede externas, cobrindo 200 (padrão e estratégia nomeada), `reflect`, 400, 422, 504 e 500.

### Key Entities

- **Requisição de chat**: mensagem do usuário, nome opcional da estratégia e indicação de reflection.
- **Registro de estratégias**: mapeamento de nomes para estratégias de raciocínio, com a capacidade de aplicar reflection na resolução.
- **Resposta de chat**: resposta final, trace de eventos e métricas da execução.
- **Erro de chat**: status e corpo que distinguem entrada inválida, estratégia desconhecida, timeout e falha interna.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: 100% das requisições válidas cuja execução termina dentro do limite recebem 200 com resposta, trace e métricas completos.
- **SC-002**: 100% dos corpos inválidos recebem 400 com pelo menos uma issue identificando o campo com problema, sem que nenhuma estratégia seja executada.
- **SC-003**: 100% das estratégias desconhecidas recebem 422 com a lista de nomes disponíveis.
- **SC-004**: Nenhuma requisição espera mais que 60 segundos (mais uma pequena margem de processamento) por uma resposta; ao exceder o limite, ela recebe 504.
- **SC-005**: A suíte de testes de integração roda sem acesso à rede externa e produz o mesmo resultado em execuções repetidas.

## Assumptions

- O pedido escreve `reflect` sem `?`, mas ele é tratado como opcional com padrão `false`, para que `{ message }` sozinho seja uma requisição válida, coerente com o padrão `react` de `strategy`.
- O servidor segue o padrão já usado no projeto: Express 5, com o roteamento em `src/http/server.ts`, a coordenação em um controller e o registro em `src/agents/index.ts`. A fronteira em camadas da constituição (princípio II) é mantida.
- O registro contém apenas nomes base. A arena continua com os identificadores `reflect:*` que já tem; migrá-la para o novo registro é opcional e não faz parte desta feature.
- O modelo e o store usados em produção são os mesmos que as estratégias já resolvem por padrão; o endpoint não recebe modelo, store, `maxIterations` ou `maxReflections` do cliente.
- O timeout encerra a requisição HTTP, mas não garante interromper a execução interna da estratégia; o cancelamento cooperativo fica fora de escopo.
- O formato de erro para 422, 504 e 500 é um objeto JSON com `error` (código curto) e `message`; o 400 acrescenta `issues`.
- `npm run dev` continua sendo o ponto de entrada e passa a iniciar o servidor HTTP por `src/index.ts`, sem script novo (decidido no [plan.md](./plan.md)).
