# Por onde começar

Este projeto abre o módulo 3 (MCP na Prática). Se você está chegando agora, leia nesta ordem.

---

## O que estamos construindo e por quê

> Estamos construindo um agente que **resolve um pipeline de dados inteiro sozinho**, não um chatbot que responde perguntas.

Nos projetos do módulo 02, o código sempre controlava a sequência: primeiro extrai a intenção, depois executa uma ação específica, depois gera uma resposta — cada etapa era um nó de grafo escrito à mão. Aqui a proposta muda: em vez de escrever um nó para "converter CSV", outro para "inserir no Mongo", outro para "consultar o Mongo" e outro para "escrever o relatório", esse trabalho inteiro é entregue a **um único agente autônomo** com acesso a ferramentas reais — algumas vindas de servidores MCP (filesystem, MongoDB), uma vinda de uma tool comum do LangChain (conversão CSV→JSON).

O cenário concreto: o usuário manda uma pergunta de análise de vendas junto com um CSV embutido no texto ("qual a receita total desses dados?"). O agente decide sozinho, guiado por um prompt com passos numerados, que precisa: limpar o banco, converter o CSV, salvar o JSON, inserir no MongoDB, consultar o MongoDB, e escrever o relatório final em disco.

```
Módulo 02: você escreve o "como".        Módulo 03 (aqui): você escreve o "o quê" — e confia
Cada etapa é um nó explícito.            que o modelo descobre o "como" chamando as tools certas.
```

---

## Trilha de leitura

| Ordem | Documento | Por que ler |
| --- | --- | --- |
| 1 | [mcp-multiplos-servidores.md](./mcp-multiplos-servidores.md) | Como dois servidores MCP (filesystem + MongoDB) são combinados no mesmo cliente, e como uma tool nativa do LangChain (`csv_to_json`) entra na mesma lista. Pressupõe que você já leu o `mcp.md` de `05-safeguard-prompt-injection`. |
| 2 | [agente-autonomo-vs-orquestracao-explicita.md](./agente-autonomo-vs-orquestracao-explicita.md) | O núcleo do projeto: por que o grafo tem só 2 nós e como a responsabilidade de orquestrar o pipeline migrou do código para o loop de tool calling do agente. |
| 3 | [observabilidade-agent-loop.md](./observabilidade-agent-loop.md) | Como enxergar o que o agente está decidindo passo a passo, já que ele não tem mais um `console.log` por nó — via callbacks do LangChain. |
| 4 | [glossario.md](./glossario.md) | Referência rápida de todos os termos novos. Consulte quando encontrar algo que não reconhece. |

---

## Mapa do código

```
data/sales.csv                          → exemplo de entrada (CSV de vendas)
data/sales-complete.csv                 → variante usada por padrão em src/index.ts
data.json / products.json               → dados auxiliares de exploração/teste
reports/                                → saída: relatórios .txt escritos pelo próprio agente

src/config.ts                           → configuração do modelo (OpenRouter, modelo, temperatura)
src/prompts/v1/identifyIntent.ts        → IntentSchema (Zod) + prompt de extração de intenção
src/prompts/v1/agentNode.ts             → prompt do agente: os 6 passos (Step 0 a Step 5)

src/tools/fsTool.ts                     → configuração do servidor MCP filesystem
src/tools/mongodbTool.ts                → configuração do servidor MCP MongoDB
src/tools/csvToJSONTool.ts              → tool nativa do LangChain (não-MCP)
src/services/mcpService.ts              → combina os 2 servidores MCP + a tool nativa num único array
src/services/openRouterService.ts       → ChatOpenAI + createAgent (2 modos: schema vs. tools) + callbacks

src/graph/state.ts                      → GraphAnnotation: messages, intent, fileContent, fileName, error
src/graph/nodes/intentNode.ts           → extrai intenção estruturada (sem tools)
src/graph/nodes/agentNode.ts            → delega a execução inteira ao agente autônomo (com tools)
src/graph/graph.ts                      → StateGraph de 2 nós: intentParser → agent
src/graph/factory.ts                    → monta o grafo com o OpenRouterService

src/server.ts                           → Fastify: POST /chat invoca o grafo
src/index.ts                            → sobe o servidor e dispara uma chamada de exemplo
docker-compose.yaml                     → MongoDB + mongo-express (UI de inspeção)
langgraph.json                          → configuração do LangGraph Studio (graph `multiple_mcp_tools`)
```

---

## O fluxo em uma linha

```
POST /chat → intentParser (extrai intent + fileContent) → agent (decide e executa tudo) → resposta
```

Se `intentParser` falhar (`state.error` preenchido), o grafo pula direto para `END` sem chamar o agente — não adianta tentar executar um pipeline sem saber o que fazer.

---

## Como rodar e ver o que importa

```bash
# 1. Suba o MongoDB local
npm run docker:infra:up

# 2. Rode o pipeline de exemplo (dispara automaticamente uma pergunta sobre sales-complete.csv)
npm start
```

Acompanhe o terminal: você vai ver a sequência `🧠 LLM thinking...` → `🎯 Decided to call: ...` → `🔧 Tool called: ...` → `✅ Tool done: ...` se repetindo várias vezes antes da resposta final — essa é a evidência visual de que o agente está decidindo o pipeline sozinho, não seguindo nós fixos do grafo.

Depois, confira os artefatos gerados:

```bash
cat reports/total_revenue_report.txt
```

Esse arquivo foi escrito pelo próprio agente, no último passo do prompt — não existe nenhuma linha de código no projeto que formate esse relatório.
