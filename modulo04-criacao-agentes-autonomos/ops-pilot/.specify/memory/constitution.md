# opsPilot Constitution

## Core Principles

### I. TypeScript ESM Estrito

O projeto deve usar Node.js 22 LTS, TypeScript com `strict: true` e módulos ESM. Novos módulos devem ter contratos explícitos, evitar `any` desnecessário e preservar a compatibilidade com a configuração e os scripts definidos em `package.json`.

### II. Arquitetura em Camadas

Organize a aplicação em Model, Service e Controller. Controllers devem coordenar entradas e respostas, Services devem concentrar regras de negócio e Models devem representar o acesso e os contratos dos dados. Prefira funções puras e mantenha efeitos colaterais nas bordas da aplicação.

### III. Contratos e Erros Explícitos

Toda entrada e saída externa, incluindo as fronteiras HTTP, HTML e CLI, deve ser validada com Zod. Erros de domínio devem ser modelados com classes próprias e traduzidos para respostas adequadas somente na borda correspondente. Os contratos entre o agente LangChain/LangGraph, a API e os serviços devem permanecer claros e verificáveis.

### IV. Testes e Qualidade Não Negociáveis

Toda mudança de lógica deve criar ou atualizar seus testes. Use `node:test` executado por `tsx`, cubra os contratos e os caminhos de erro relevantes e mantenha `npm run typecheck` e `npm test` verdes antes de considerar uma implementação concluída.

### V. Integração Segura e Simples

O agente de gerenciamento de alertas e incidentes usa LangChain/LangGraph e integração compatível com OpenRouter. Use as dependências existentes antes de adicionar abstrações ou bibliotecas. Nunca leia, exponha ou versione `.env`, chaves ou outros secrets; carregue variáveis de ambiente usando o suporte nativo do Node.js e não use `dotenv`.

## Additional Constraints

- Runtime e linguagem: Node.js 22 LTS, TypeScript ESM e `strict: true`.
- Dependências de produção: `@langchain/core`, `@langchain/langgraph`, `@langchain/openai`, Express 5, MySQL, Sequelize 6 e Zod.
- Dependências de desenvolvimento: tipos de Express, Node.js e Sequelize, `tsx` e TypeScript.
- Persistência: MySQL com Sequelize quando o acesso ORM for necessário.
- Interfaces de execução: API Express, além dos pontos de entrada `src/index.ts`, `src/arena.ts` e `src/bench.ts`.
- Scripts oficiais: `npm run dev`, `npm run arena`, `npm run bench`, `npm test` e `npm run typecheck`.
- Não introduza `dotenv`, não acesse secrets diretamente no código e não altere contratos externos sem atualizar a especificação e os testes correspondentes.

## Development Workflow

- Siga obrigatoriamente as quatro etapas, nesta ordem: `/speckit.specify -> /speckit.plan -> /speckit.tasks -> /speckit.implement`.
- **1. Specify (`/speckit.specify`)**: registre problema, objetivo, escopo, partes afetadas, restrições, critérios de aceitação e exclusões. Esclareça ambiguidades e produza uma especificação versionada; não implemente código nesta etapa.
- Após `specify`, revise e aprove a especificação antes de iniciar o planejamento. Em caso de rejeição, corrija a especificação e repita a revisão.
- **2. Plan (`/speckit.plan`)**: analise o código existente e descreva arquivos, módulos, interfaces, dependências, decisões de arquitetura, tratamento de erros, testes e riscos. Relacione cada decisão aos critérios de aceitação.
- Após `plan`, revise e aprove o plano antes de gerar as tarefas. Em caso de rejeição, corrija o plano e repita a revisão.
- **3. Tasks (`/speckit.tasks`)**: decomponha o plano em tarefas pequenas, ordenadas e executáveis. Cada tarefa deve indicar alteração, localização, resultado esperado e validação, incluindo testes, documentação, `npm run typecheck` e `npm test` quando aplicável.
- **4. Implement (`/speckit.implement`)**: execute as tarefas na ordem, faça a menor alteração necessária e valide cada mudança relevante. Ao final, execute `npm run typecheck` e `npm test`. Se o escopo mudar, atualize a especificação e o plano antes de continuar.
- Versione as especificações junto com o projeto e mantenha a constituição sincronizada com as decisões permanentes do projeto.

## Governance

Esta constituição define as regras permanentes do opsPilot e deve ser considerada junto com `.github/copilot-instructions.md`. Em caso de conflito, a regra mais específica para a mudança deve ser aplicada sem violar estes princípios. Alterações na constituição exigem atualização do número de versão, da data de alteração e das especificações ou instruções afetadas. Toda revisão deve verificar os contratos, os testes e os gates de qualidade definidos aqui.

**Version**: 1.0.1 | **Ratified**: 2026-09-28 | **Last Amended**: 2026-09-28
