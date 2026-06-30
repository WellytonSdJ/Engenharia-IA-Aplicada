# Glossário

Referência rápida dos termos deste projeto. Para profundidade, vá ao documento específico de cada conceito.

Termos já cobertos no [glossário do projeto 02](../../02-song-highlights/docs/glossario.md) (LangGraph, LangChain, OpenRouter, Zod, etc.) não são repetidos aqui — consulte lá quando necessário.

---

## Segurança e ataques

| Termo | Definição |
| --- | --- |
| **Prompt Injection** | Ataque onde input externo insere instruções que manipulam o comportamento do LLM — sobrescrevendo ou contradizendo o system prompt. O análogo de SQL injection para modelos de linguagem. |
| **Injeção direta** | Variante onde o usuário digita a instrução maliciosa diretamente no campo de input (ex: "ignore previous instructions"). |
| **Injeção indireta** | Variante onde a instrução maliciosa está em um dado que o LLM processa — um documento, página web, resultado de API — sem o usuário digitar diretamente. |
| **Jailbreak** | Técnica de contornar as restrições de um modelo via construção de narrativa, roleplay ou framing especial ("estamos em modo de manutenção", "para fins educacionais", "DAN"). |
| **Prompt Hijacking** | Variante de injection onde o atacante "sequestra" o papel do modelo — faz o LLM assumir uma identidade diferente da definida no system prompt. |
| **Override direto** | Padrão de ataque que instrui explicitamente o modelo a ignorar instruções anteriores: "IGNORE PREVIOUS INSTRUCTIONS. You are now...". Eficaz em modelos menores. |
| **Exfiltração de dados** | Obtenção não autorizada de informações sensíveis (`.env`, credenciais, configurações) via um modelo comprometido com acesso a ferramentas. |

---

## Defesas e arquitetura de segurança

| Termo | Definição |
| --- | --- |
| **Guardrails** | Camada de validação **externa ao modelo executor** que intercepta o input antes de chegar ao LLM que tem acesso a ferramentas. Bloqueio é determinístico (código), não probabilístico (confiança no LLM). |
| **Safeguard model** | Modelo de linguagem especializado em classificar risco — mais rápido e barato que o executor. Não executa ações, apenas analisa texto e retorna SAFE ou UNSAFE. No projeto: `openai/gpt-oss-safeguard-20b`. |
| **Defesa probabilística** | Abordagem que confia que o modelo vai seguir as regras do system prompt. Fraca porque o modelo pode ser convencido a não seguir. |
| **Defesa determinística** | Abordagem onde o **código** decide o que acontece — não o modelo. O `routeAfterGuardrails` é código: se `safe === false`, vai para `blocked`, sem exceção. |
| **Fail closed** | Princípio de segurança: na dúvida ou em caso de falha, nega acesso. O `guardrailsCheckNode` captura exceções e retorna `{ safe: false }` — nunca permite por padrão. Oposto: "fail open" (permitir na dúvida — perigoso). |
| **Defense in depth** | Múltiplas camadas de defesa independentes — se uma falha, outra segura. No projeto: guardrail + system prompt com regras. Em produção: guardrail de input + filtro de ferramentas por permissão + guardrail de output. |
| **PromptTemplate** | Template do LangChain com variáveis nomeadas (`{USER_ROLE}`) que substitui concatenação manual de strings. O LangChain aplica escape nas variáveis antes de substituir, reduzindo risco de injection via campos do usuário. |
| **Modo unsafe (`--unsafe`)** | Flag de demonstração que desativa os guardrails, tornando o sistema vulnerável a injection. Só existe para fins educacionais — mostra o contraste entre modos. |

---

## MCP — Model Context Protocol

| Termo | Definição |
| --- | --- |
| **MCP** | Model Context Protocol — protocolo aberto (Anthropic) que padroniza como LLMs se conectam a ferramentas e recursos externos. Define contrato entre cliente (framework) e servidor (ferramenta). |
| **MCP Server** | Processo que expõe ferramentas via MCP. No projeto: `@modelcontextprotocol/server-filesystem` expõe operações de leitura de arquivos. |
| **MCP Client** | Componente que conecta o LLM aos servidores MCP. No projeto: `MultiServerMCPClient` do pacote `@langchain/mcp-adapters`. |
| **STDIO transport** | Modo de comunicação entre cliente e servidor MCP via stdin/stdout — o servidor roda como subprocesso e troca mensagens JSON-RPC com o cliente. |
| **Tool** | Capacidade específica exposta pelo servidor MCP (ex: `read_text_file`, `list_directory`). O modelo "conhece" as tools e decide quando usá-las. |
| **Tool calling** | Mecanismo onde o LLM gera uma "chamada" estruturada a uma tool (nome + argumentos), o framework executa a tool real, e injeta o resultado de volta no contexto antes de continuar. |
| **Agente (agent)** | LLM configurado com tools que pode tomar múltiplos passos autonomamente — decide usar uma tool, analisa o resultado, decide o próximo passo. No projeto: `createAgent` do LangChain. |
| **Lazy initialization** | Padrão onde o agente MCP é criado apenas na primeira chamada e reutilizado nas seguintes — evita criar um novo processo MCP a cada requisição. |
| **Escopo MCP** | Diretório passado ao servidor MCP que limita quais arquivos ele pode acessar. No projeto: `process.cwd()` — apenas o diretório raiz do projeto. |

---

## RBAC

| Termo | Definição |
| --- | --- |
| **RBAC** | Role-Based Access Control — modelo de permissões onde o que um usuário pode fazer é determinado pelo papel (role) que ele ocupa, não por configurações individuais. |
| **Role (papel)** | Categoria de usuário com um conjunto pré-definido de permissões. No projeto: `admin` e `member`. |
| **Permission (permissão)** | String que representa uma ação permitida. No projeto: `"read_package"`, `"execute_commands"`. O admin tem ambas; o member não tem nenhuma. |
| **RBAC via prompt** | Implementação fraca onde o papel do usuário é informado ao LLM via system prompt e o modelo é "confiado" a respeitá-lo. Vulnerável a injection. |
| **RBAC via código** | Implementação forte onde a verificação de permissão é um `if` determinístico no código — o modelo não participa da decisão. Resistente a injection. |

---

## LangGraph neste projeto

| Termo | Definição |
| --- | --- |
| **SafeguardStateAnnotation** | Schema Zod do estado deste grafo: `messages`, `user`, `guardrailCheck`, `guardrailsEnabled`. |
| **guardrailCheck** | Campo do estado onde o `guardrailsCheckNode` deposita o resultado do safeguard model (`{ safe, reason, analysis }`). A edge condition lê daqui para decidir o roteamento. |
| **guardrailsEnabled** | Flag booleana no estado que reflete se `--unsafe` foi passado na CLI. Controlado pelo código, não pelo LLM. |
| **routeAfterGuardrails** | Função de roteamento condicional: lê `guardrailsEnabled` e `guardrailCheck` do estado e retorna `'chat'` ou `'blocked'`. Decisão em código. |
| **blockedNode** | Nó que monta mensagem de bloqueio usando template de texto — sem invocar nenhum LLM. Resulta de um roteamento UNSAFE. |
| **Single-shot** | Grafo que processa uma mensagem por invocação, sem histórico entre invocações (sem checkpointer). Diferente do projeto 02, que retoma sessões via `thread_id`. |
