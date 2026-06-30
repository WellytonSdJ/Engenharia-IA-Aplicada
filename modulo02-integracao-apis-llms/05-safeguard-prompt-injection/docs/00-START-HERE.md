# Por onde começar

Este projeto demonstra ataques de prompt injection e defesa com guardrails baseados em LLM.
Se você está chegando agora, leia nesta ordem:

---

## O que estamos construindo e por quê

> Estamos construindo um sistema que **prova uma vulnerabilidade e sua solução**, não um produto.

A maioria dos tutoriais de chatbot com LLM mostra como fazer o bot responder bem. Poucos mostram como ele pode ser manipulado — e como se defender. Este projeto faz isso de forma executável.

O cenário é simples: um assistente CLI com dois tipos de usuário (admin e member) e uma ferramenta real de filesystem via MCP. O admin pode ler arquivos. O member não pode. A pergunta é:

**Se o sistema prompt proíbe o acesso, um usuário mal-intencionado consegue burlar isso?**

Resposta: **sim, com facilidade — a menos que você coloque uma camada de defesa fora do LLM.**

```
System prompts são texto. Texto pode ser ignorado.
Controle de acesso precisa ser código determinístico.
```

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [prompt-injection.md](./prompt-injection.md) | O problema: o que é, como funciona, quais são os vetores de ataque. Leia isso antes de qualquer código — sem entender o problema, a solução não faz sentido. |
| 2 | [guardrails.md](./guardrails.md) | A solução: como a camada de guardrail funciona, por que ela é determinística enquanto o system prompt não é, e como o modelo safeguard é diferente do modelo executor. |
| 3 | [mcp.md](./mcp.md) | O que torna este projeto perigoso sem guardrails: o MCP dá ferramentas reais ao LLM. Se o modelo for injetado, ele pode executar ações reais no sistema de arquivos. |
| 4 | [rbac.md](./rbac.md) | O sistema de permissões: admin vs member, como as permissões são verificadas, por que isso sozinho não protege sem guardrail. |
| 5 | [langgraph.md](./langgraph.md) | Como tudo é orquestrado: o grafo de segurança, o estado, o roteamento condicional entre os três nós. |
| 6 | [glossario.md](./glossario.md) | Referência rápida de todos os termos. Consulte quando encontrar algo que não reconhece. |

---

## Mapa do código

Depois de ler os docs, o código vai fazer mais sentido nesta ordem:

```
data/users.json                              → base de usuários: roles e permissões
prompts/system.txt                           → system prompt (IDÊNTICO nos dois modos)
prompts/guardrails.txt                       → instrução para o modelo safeguard
prompts/blocked.txt                          → template da mensagem de bloqueio
prompts/user/read-package-version.txt        → exemplo de ataque: override direto
prompts/user/read-env.txt                    → exemplo de ataque: instrução indireta

src/config.ts                                → usuários, modelos, leitura dos prompts
src/graph/state.ts                           → SafeguardStateAnnotation: user, guardrailCheck, guardrailsEnabled
src/graph/nodes/edgeConditions.ts            → routeAfterGuardrails: lógica de roteamento
src/graph/nodes/guardrailsCheckNode.ts       → chama o modelo safeguard
src/graph/nodes/chatNode.ts                  → chama o modelo executor com ferramentas MCP
src/graph/nodes/blockedNode.ts               → retorna mensagem de bloqueio (sem chamar LLM)
src/graph/graph.ts                           → StateGraph: montagem do grafo de segurança
src/services/mcpService.ts                   → configura o MCP filesystem server
src/services/openrouterService.ts            → modelo executor + modelo safeguard
src/graph/factory.ts                         → monta o grafo
src/index.ts                                 → CLI: --user, --message, --unsafe, --prompt-path
```

---

## O fluxo em uma linha

```
usuário → guardrails_check → [chat com MCP] ou [blocked]
```

Toda invocação começa no `guardrails_check`. Se estiver em modo `--unsafe`, o nó retorna `safe: true` imediatamente e vai para `chat`. Se guardrails estiverem ativos, o modelo safeguard analisa o input e decide o caminho.

---

## Como rodar e ver os dois modos

**Modo protegido — membro tentando injeção:**
```bash
npm run chat:member:safe
# → bloqueado pelo guardrail
```

**Modo vulnerável — mesma injeção, sem guardrail:**
```bash
npm run chat:member:unsafe:package
# → LLM obedece e lê o arquivo
```

**Admin (sempre funciona, com ou sem guardrails):**
```bash
npm run chat:admin
```

Rodar esses três em sequência é a forma mais rápida de entender o projeto.

---

## O insight central do projeto

O mesmo system prompt, o mesmo modelo, o mesmo código — **só o modo muda**.

Em modo unsafe o LLM lê o arquivo porque foi manipulado. Em modo safe o LLM **nunca chega a ver** o prompt malicioso — o guardrail intercepta antes.

Isso demonstra duas coisas:

1. Você não pode confiar que o LLM vai seguir suas regras quando pressionado
2. A defesa eficaz não é convencer o LLM — é impedir que o prompt chegue até ele
