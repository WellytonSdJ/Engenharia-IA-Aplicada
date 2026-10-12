# Quickstart: Endpoint HTTP de Chat

## Pré-requisitos

- Node.js 22 e `npm install` executado.
- Para chamadas reais ao modelo: `.env` com `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` (não é necessário para os testes).

## 1. Gates automáticos (sem rede)

```bash
npm run typecheck
npm test
```

Esperado: as suítes `src/agents/index.test.ts`, `src/services/chat.test.ts` e `src/http/server.test.ts` passam, junto com as existentes. Os testes HTTP usam porta efêmera em `127.0.0.1` e estratégia fake, sem acessar a internet.

## 2. Validação manual com o servidor real

```bash
npm run dev            # sobe em http://localhost:3000 (ou PORT do ambiente)
```

| Cenário | Comando | Esperado |
| --- | --- | --- |
| Padrão `react` | `curl -s -X POST localhost:3000/chat -H "content-type: application/json" -d '{"message":"quais alertas estão firing?"}'` | 200 com `answer`, `trace`, `metrics` |
| Estratégia nomeada + reflection | `... -d '{"message":"...","strategy":"plan-and-execute","reflect":true}'` | 200; `trace` contém eventos `critique` |
| Corpo inválido | `... -d '{"message":"   "}'` | 400 com `issues[0].path = ["message"]` |
| JSON malformado | `... -d '{message'` | 400 com `"Malformed JSON body."` |
| Estratégia desconhecida | `... -d '{"message":"oi","strategy":"reflect:react"}'` | 422 com `available: ["react","plan-and-execute"]` |

No PowerShell, use `curl.exe` e aspas simples externas, ou `Invoke-RestMethod`.

O 504 não é validado manualmente (exigiria esperar 60 s); ele é coberto pelo teste de integração com `timeoutMs` reduzido.

Contrato completo: [contracts/chat-http.md](./contracts/chat-http.md). Entidades e regras: [data-model.md](./data-model.md).
