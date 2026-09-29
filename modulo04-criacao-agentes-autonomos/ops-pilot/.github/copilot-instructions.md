# opsPilot

Copiloto de plantão para gerenciamento de alertas e incidentes de produção. A API é um agente LangChain/LangGraph conectado ao OpenRouter.

## Stack

- Node.js 22 LTS; TypeScript ESM com `strict: true`.
- Zod valida entradas e saídas nas fronteiras HTML e CLI.
- Express e MySQL.
- Testes com `node:test` via `tsx`.
- Use o carregamento de variáveis de ambiente nativo do Node.js; não use `dotenv`.

## Comandos

- `npm run dev`
- `npm run arena`
- `npm run bench`
- `npm test` ou `npm run test`
- `npm run typecheck`

## Convenções

- Organize a aplicação em MVC: Model, Service e Controller.
- Valide toda entrada externa com Zod.
- Modele erros de domínio com classes e traduza-os na borda.
- Crie ou atualize testes junto com cada mudança de lógica.
- Mantenha `npm run typecheck` e `npm test` verdes.
- Prefira funções puras e isole efeitos colaterais nas bordas.
- Nunca leia ou versione `.env`, chaves ou outros secrets.

## Fluxo

- Siga obrigatoriamente as quatro etapas do Spec Kit, nesta ordem: `/speckit.specify -> /speckit.plan -> /speckit.tasks -> /speckit.implement`.
- **1. Specify (`/speckit.specify`)**: transforme a necessidade do usuário em uma especificação clara e verificável. Registre o problema, o objetivo, o escopo, os usuários ou sistemas afetados, as restrições, os critérios de aceitação e os casos que ficam explicitamente fora do escopo. Nesta etapa, não implemente código; esclareça ambiguidades e produza uma especificação versionada que sirva como contrato para as etapas seguintes.
- Após `specify`, revise e aprove a especificação antes de iniciar o planejamento; se for rejeitada, corrija-a e repita a revisão.
- **2. Plan (`/speckit.plan`)**: com a especificação definida, analise o código existente e descreva a solução técnica. Identifique os módulos e arquivos envolvidos, as interfaces e dependências, as decisões de arquitetura, as estratégias de tratamento de erros, os testes necessários e os riscos conhecidos. O plano deve explicar como os critérios de aceitação serão atendidos.
- Após `plan`, revise e aprove o plano antes de gerar as tarefas; se for rejeitado, corrija-o e repita a revisão.
- **3. Tasks (`/speckit.tasks`)**: decomponha o plano em tarefas pequenas, concretas e executáveis, mantendo a ordem das dependências. Cada tarefa deve indicar o que será alterado, em qual parte do projeto, qual resultado é esperado e como será validada. Inclua tarefas de testes, documentação e verificação de `npm run typecheck` e `npm test` quando forem afetadas pela mudança. Não pule diretamente do plano para a implementação.
- **4. Implement (`/speckit.implement`)**: execute as tarefas na ordem definida, fazendo a menor alteração necessária e respeitando a especificação e o plano. Valide cada mudança relevante com os testes apropriados e, ao final, execute `npm run typecheck` e `npm test`. Se a implementação revelar uma necessidade fora da especificação, pare, atualize a especificação e o plano pelas etapas correspondentes antes de continuar; não altere o escopo silenciosamente.
- Versione as especificações junto com o projeto.
