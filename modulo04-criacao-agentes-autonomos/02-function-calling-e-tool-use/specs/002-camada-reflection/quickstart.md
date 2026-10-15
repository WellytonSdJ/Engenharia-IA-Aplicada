# Quickstart: Camada de Reflection

## Pré-requisitos

- Instalar dependências do projeto com `npm install`.
- Para execução real da arena, configurar `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` no ambiente conforme as orientações existentes do projeto. Os testes abaixo não precisam de credenciais nem rede.

## Validar a lógica de reflection sem rede

1. Executar testes focados do wrapper e da arena:

   ```powershell
   node --import tsx --test src/agents/reflection.test.ts src/arena.test.ts
   ```

2. Verificar nos testes:
   - aprovação imediata sem regeneração;
   - reprovação seguida de regeneração com feedback e aprovação;
   - limite padrão de duas regenerações, resposta mais recente e crítica reprovadora no trace;
   - `maxReflections: 0` sem invocação do crítico;
   - uso exclusivo do pedido original e das observações, e uso da mesma instância de modelo;
   - falha explícita em crítica inválida/erro e soma de métricas;
   - resolução isolada dos nomes de arena refletidos.

## Executar arena com as variantes refletidas

```powershell
npm run arena -- --strategies reflect:react,reflect:plan-and-execute --max-iterations 8 "investigue os alertas firing"
```

O resultado esperado contém um bloco `reflect:react` e outro `reflect:plan-and-execute`, cada qual com resposta, trace (incluindo eventos `critique`) e métricas próprias.

O evento `critique` registra em `content` os campos `aproved` e `feedback`. Quando o crítico reprova e ainda há ciclos, o feedback acionável acompanha o pedido original na tentativa seguinte; o crítico continua recebendo somente o pedido original, a resposta da tentativa e suas observações.

## Quality gates

```powershell
npm run typecheck
npm test
```

Contrato detalhado: [contracts/reflection.md](./contracts/reflection.md). Regras de entidades e transições: [data-model.md](./data-model.md).
