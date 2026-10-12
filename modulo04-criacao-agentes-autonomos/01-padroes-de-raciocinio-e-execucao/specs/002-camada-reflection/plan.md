# Implementation Plan: Camada de Reflection

**Branch**: `002-camada-reflection` | **Date**: 2026-10-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-camada-reflection/spec.md`

## Summary

Adicionar um decorador `withReflection(strategy, opts)` que preserva a estratégia de raciocínio existente e coordena chamadas de crítica e regeneração. O crítico usa a mesma instância de modelo da estratégia base, avalia somente o pedido original e as observações (`observation`) da tentativa, e retorna o contrato Zod com os campos `{ aproved, feedback }`. Em caso de reprovação, o feedback é anexado explicitamente ao pedido enviado à nova execução. O limite padrão permite duas regenerações. O trace intercala os eventos de crítica com os traces das tentativas; as métricas somam chamadas e latência de todas as execuções e avaliações. A arena registra `reflect:react` e `reflect:plan-and-execute`.

## Technical Context

**Language/Version**: TypeScript ESM, `strict: true`, Node.js 22 LTS

**Primary Dependencies**: Dependências já existentes: `@langchain/core`, `@langchain/langgraph`, `@langchain/openai` e Zod

**Storage**: Sem persistência nova; reutiliza o `OpsStore` injetado pelas estratégias existentes

**Testing**: `node:test` executado via `tsx`; testes unitários determinísticos para o decorador e testes de integração da seleção na arena; gates `npm run typecheck` e `npm test`

**Target Platform**: CLI Node.js e estratégias reutilizáveis pelo serviço

**Project Type**: Aplicação TypeScript com agente, CLI de arena e serviços

**Performance Goals**: Uma chamada estruturada ao crítico por tentativa avaliada; no máximo `maxReflections` regenerações; zero chamadas extras quando o limite é zero

**Constraints**: Reusar exatamente a instância do modelo resolvida/injetada pela estratégia; não introduzir dependências; não esconder erro de modelo ou validação; preservar a ordem dos eventos e os dados das estratégias base

**Scale/Scope**: Um decorador genérico, duas variantes na arena, testes e contratos do agente e da CLI

## Constitution Check

- TypeScript ESM estrito e contratos explícitos: **PASS** — usar interfaces existentes e schemas Zod, sem `any`.
- Arquitetura em camadas: **PASS** — reflection permanece no limite das estratégias em `src/agents`, sem mover regras para controllers ou models.
- Validação Zod e erros explícitos: **PASS** — validar opções e saída estruturada; deixar falhas do crítico/regeneração propagarem sem fallback de sucesso.
- Testes e quality gates: **PASS** — cobrir aprovação, regeneração, limite, erros, métricas e arena; executar os gates oficiais.
- Integração simples e segura: **PASS** — reusar modelo e dependências existentes, sem tocar em credenciais ou `.env`.

## Project Structure

### Documentation (this feature)

```text
specs/002-camada-reflection/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── reflection.md
└── tasks.md
```

### Source Code

```text
src/agents/
├── reflection.ts
├── reflection.test.ts
├── strategy.ts
├── trace.ts
├── react.ts
└── plan-and-execute.ts

src/
├── arena.ts
└── arena.test.ts

specs/001-nucleo-raciocinio-opspilot/contracts/
├── strategy.md
└── arena-cli.md
```

**Structure Decision**: Implementar o decorador e seu contrato estruturado em `src/agents/reflection.ts`, testá-lo isoladamente, e expor as variantes no registry já passado para `runArena`. Atualizar os contratos de estratégia e arena da feature-base para documentar a extensão pública.

## Design Decisions

1. **Modelo compartilhado**: resolver o modelo uma vez no wrapper (ou reutilizar o modelo já injetado) e repassá-lo à estratégia e ao crítico. Isso atende ao contrato de modelo comum e mantém os testes sem rede.
2. **Entrada da crítica**: gerar `observationsOf(trace)` filtrando exclusivamente eventos `observation`. Enviar o pedido original, observações e resposta ao crítico; orientar o prompt de sistema a julgar apenas por essas evidências e não por conhecimento externo.
3. **Regeneração genérica**: como `ReasoningStrategy.run` recebe um pedido textual, enviar uma nova entrada que preserva o pedido original e acrescenta um bloco claramente delimitado com o feedback anterior. Avaliar cada nova resposta contra o pedido original, não contra a entrada já acrescida de feedback.
4. **Limite e aprovação**: `maxReflections` conta regenerações, com default 2 e mínimo 0. Avaliar a resposta inicial e cada resposta regenerada; se a última crítica reprovar após o limite, retornar a resposta mais recente com a reprovação registrada.
5. **Trace e métricas**: concatenar os traces em ordem de execução e inserir um evento `critique` após cada tentativa avaliada. Incrementar uma chamada por crítica e somar as métricas das execuções base/regeneradas mais a duração medida de cada crítica.
6. **Arena**: registrar estratégias decoradas com seus nomes `reflect:react` e `reflect:plan-and-execute`; seleção sem prefixo permanece inalterada.

## Error Handling

- Validar o objeto estruturado retornado pelo crítico com o schema; resposta ausente ou inválida falha a execução.
- Erros do provedor, crítico ou estratégia regenerada propagam para a borda CLI/API existente, sem aprovação ou resposta sintética.
- `maxReflections` inválido falha na validação de opções antes de entrar no ciclo.
- O limite zero retorna o resultado da estratégia base sem invocar o crítico; a obtenção do modelo da estratégia base segue seu comportamento habitual.

## Test Strategy

- Usar modelo estruturado determinístico para aprovar na primeira avaliação, reprovar e aprovar após regeneração, reprovar até o limite e lançar/retornar saída inválida.
- Verificar a entrada de cada chamada base, a preservação do pedido original na crítica, a extração somente de observações, a identidade do modelo compartilhado e o feedback na regeneração.
- Verificar ordem e conteúdo dos eventos `critique`, preservação dos traces e soma exata de chamadas/latência.
- Verificar que a arena resolve os dois nomes de variante e mantém saídas independentes; nomes base seguem disponíveis.
- Executar testes direcionados, `npm run typecheck` e `npm test`.

## Risks

- Estratégias podem representar observações com texto livre; a camada passa os conteúdos de eventos `observation` sem interpretá-los e o crítico deve tratar esse conteúdo como evidência não confiável.
- A regeneração reenviará um pedido composto em texto ao agente base; os passos operacionais e chamadas a ferramentas poderão ocorrer novamente conforme a estratégia, portanto a feature não deve prometer idempotência de ferramentas.
- A chave `aproved` contém a grafia fornecida no requisito e deve ser mantida consistente entre schema, prompt e testes.

## Constitution Re-check

Os contratos usam Zod e tipos existentes, as falhas continuam explícitas, não há dependências novas nem acesso a secrets. **PASS**.
