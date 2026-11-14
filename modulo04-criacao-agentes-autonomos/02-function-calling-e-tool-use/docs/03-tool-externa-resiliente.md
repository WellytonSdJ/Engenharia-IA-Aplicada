# Tool externa resiliente — `check_provider_status`

> Este doc cobre a spec 006 ([`006-status-provedores-externos`](../specs/006-status-provedores-externos/spec.md)). Antes, leia [01-function-calling-e-design-de-tools.md](./01-function-calling-e-design-de-tools.md): aqui assumimos que você já sabe como o modelo escolhe uma tool.
>
> **Em 30 segundos**
>
> - A primeira tool do OpsPilot que sai do processo e chama a **internet**: a página pública de status do GitHub ou da Cloudflare.
> - Rede falha, então a tool tem quatro defesas: **timeout de 5 s**, **1 retry** (só em erro de rede ou 5xx), **Zod na resposta** e **erro vira observação** (ela devolve uma frase, nunca lança).
> - O resultado é **uma linha** (`none — All Systems Operational`), para não inflar o contexto do modelo.
> - O `fetch` é **injetável**: os testes simulam sucesso, queda, 404 e JSON quebrado sem tocar a rede.

---

## O que é: por que essa tool existe

Pagamentos falhando. É o nosso serviço ou o provedor caiu? Essa é a primeira pergunta de qualquer plantão. Sem a tool, o agente só enxerga o lado de dentro (alertas, incidentes). Com ela, consegue olhar pela janela.

| Pergunta do usuário | O que o modelo faz |
| --- | --- |
| "o deploy travou, é o GitHub?" | `check_provider_status({ provider: "github" })` |
| "o CDN está lento" | `check_provider_status({ provider: "cloudflare" })` |
| "quantos alertas estão ativos?" | **não** chama esta tool, chama `list_alerts` |

As duas URLs são da API pública padrão de status pages (`/api/v2/status.json`) e não pedem chave:

```typescript
// src/services/tools.ts
const PROVIDER_STATUS_URLS: Record<"github" | "cloudflare", string> = {
  github: "https://www.githubstatus.com/api/v2/status.json",
  cloudflare: "https://www.cloudflarestatus.com/api/v2/status.json",
};
```

> ✅ **Checkpoint**: a tool responde "o problema é nosso ou de fora?". É isso que justifica uma tool só para ela.

---

## As 4 defesas

**Analogia:** você liga para o suporte de um fornecedor. Se ninguém atender em 5 segundos, você desliga (**timeout**). Se a linha caiu, tenta de novo **uma** vez (**retry**). Se atender alguém falando outra língua, você não finge que entendeu (**validação**). E, dando certo ou errado, você **volta e conta ao seu chefe** o que aconteceu, em vez de sumir (**erro como observação**).

```text
fetch (timeout 5 s)
  ├─ exceção (rede, timeout) ── 1ª vez? ── tenta de novo ── 2ª vez? ── "Não foi possível... : <motivo>."
  ├─ HTTP 5xx ───────────────── 1ª vez? ── tenta de novo
  ├─ HTTP 4xx ───────────────── sem retry ──────────────────────────── "Não foi possível... : HTTP 404."
  └─ HTTP 2xx ── Zod falhou? ── "resposta em formato inesperado."
                └─ Zod ok ───── "none — All Systems Operational"
```

### O código, inteiro

```typescript
// src/services/tools.ts
async function fetchProviderStatus(
  provider: "github" | "cloudflare",
  fetchFn: typeof fetch,
): Promise<string> {
  const url = PROVIDER_STATUS_URLS[provider];
  let lastError = "unknown error";

  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const response = await fetchFn(url, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) {
        lastError = `HTTP ${response.status}`;
        if (response.status >= 500 && attempt === 1) {
          continue;
        }
        return `Não foi possível consultar o status de ${provider}: ${lastError}.`;
      }
      const parsed = ProviderStatusResponseSchema.safeParse(await response.json());
      if (!parsed.success) {
        return `Não foi possível consultar o status de ${provider}: resposta em formato inesperado.`;
      }
      const { indicator, description } = parsed.data.status;
      return `${indicator} — ${description}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      if (attempt === 1) {
        continue;
      }
    }
  }

  return `Não foi possível consultar o status de ${provider}: ${lastError}.`;
}
```

| Defesa | Linha-chave | Por quê |
| --- | --- | --- |
| **Timeout** | `AbortSignal.timeout(5000)` | Sem ele, um servidor mudo segura o agente até o timeout do `/chat` (60 s) |
| **Retry seletivo** | `status >= 500 && attempt === 1` e o `continue` no `catch` | 5xx e queda de rede costumam ser passageiros; **4xx não**: tentar de novo dá o mesmo 404 |
| **Validação** | `ProviderStatusResponseSchema.safeParse(...)` | A API externa pode mudar de formato sem avisar |
| **Erro como observação** | todo caminho termina em `return "..."` | O modelo lê "não consegui consultar" e segue raciocinando |

O schema da resposta só exige o que usamos e aceita o resto (`.passthrough()`):

```typescript
// src/services/tools.ts
const ProviderStatusResponseSchema = z.object({
  status: z.object({
    indicator: z.string(),
    description: z.string(),
  }),
}).passthrough();
```

> ✅ **Checkpoint**: retry só no que é **passageiro** (rede, 5xx). Erro do cliente (4xx) não melhora tentando de novo.

---

## Erro como observação × erro de fronteira

O projeto tem **duas** políticas de erro de tool, e vale saber por quê:

| | Tools de domínio (`open_incident`, ...) | `check_provider_status` |
| --- | --- | --- |
| Quando falha | **lança** `ToolBoundaryError` ([01](../../01-padroes-de-raciocinio-e-execucao/docs/03-tools-e-store-deterministico.md)) | **devolve** uma frase de erro |
| Natureza da falha | dado nosso, previsível ("serviço não existe") | mundo externo, imprevisível (rede) |
| O que o agente faz | a estratégia decide (ex.: o ReAct mostra o erro ao modelo) | o modelo lê a frase como qualquer resultado e segue |

A internet cair **não é um bug** do OpsPilot. É um fato do plantão que o modelo precisa saber ("não consegui confirmar o status do GitHub") para responder com honestidade.

---

## Testando sem rede: `fetch` injetável

```typescript
// src/services/tools.ts
export function createOpsToolDefinitions(
  store: OpsStore,
  options: { fetchFn?: typeof fetch } = {},
): OpsToolDefinition[] {
  const fetchFn = options.fetchFn ?? fetch;
```

Em produção vale o `fetch` global. No teste, entra um fake que conta as chamadas:

```typescript
// src/services/tools.test.ts
test("check_provider_status does not retry on 4xx responses", async () => {
  let calls = 0;
  const fetchFn = (async () => {
    calls++;
    return jsonResponse({}, 404);
  }) as typeof fetch;
  const { store } = await seededTools();
  const tools = createOpsTools(store, { fetchFn });

  const result = await tools[5].invoke({});
  assert.match(result, /HTTP 404/);
  assert.equal(calls, 1);
});
```

Seis testes cobrem: sucesso (default `github`), `cloudflare`, retry após queda, falha persistente sem lançar, 4xx sem retry e payload inválido.

---

## ⚠️ Pegadinhas

- **Retry infinito.** O laço vai até `attempt <= 2`, de propósito. Mais tentativas multiplicam a latência de um turno que já é lento.
- **Devolver o JSON inteiro.** A resposta da status page tem página, componentes e datas. O modelo só precisa de `indicator — description`, e o resto é token gasto à toa.
- **`fetch` global "hardcoded".** Sem a injeção, os testes dependeriam da internet e do humor do GitHub naquele dia.
- **Normalizar o indicador.** `none`/`minor`/`major`/`critical` é o vocabulário oficial das status pages. A tool repassa como está: quem interpreta é o modelo.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| A tool | [`src/services/tools.ts`](../src/services/tools.ts) | `fetchProviderStatus`, `PROVIDER_STATUS_URLS`, `ProviderStatusResponseSchema` |
| Descrição (quando usar) | [`src/services/tools.ts`](../src/services/tools.ts) | `name: "check_provider_status"` |
| Injeção de `fetch` | [`src/services/tools.ts`](../src/services/tools.ts) | `options.fetchFn ?? fetch` |
| Testes | [`src/services/tools.test.ts`](../src/services/tools.test.ts) | Os 6 testes `check_provider_status ...` |
| Spec | [`specs/006-status-provedores-externos/spec.md`](../specs/006-status-provedores-externos/spec.md) | Política de retry e formato da resposta |

---

⬅️ Anterior: [02 · Persistência com SQLite nativo](./02-persistencia-sqlite.md) · 🗺️ [Trilha de leitura](./00-START-HERE.md#trilha-de-leitura) · **➡️ Próximo passo: [04 · `POST /chat`](./04-api-chat-http.md)**
