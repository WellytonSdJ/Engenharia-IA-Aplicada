# Glossário

Referência rápida. Para profundidade, vá ao documento específico de cada conceito.

---

## Arquitetura e padrões

| Termo | Definição |
| --- | --- |
| **Wrapper** | Produto que "embrulha" uma API de LLM entregando uma experiência específica. Não é pejorativo — wrappers bem-feitos levantaram milhões. O diferencial está em validação, segurança e memória, não na chamada à API. |
| **Applied AI Engineer** | Engenheiro que consome modelos prontos via API e os integra em sistemas reais com qualidade, segurança e observabilidade. Diferente do pesquisador de ML — não treina modelos, usa-os. |
| **Pipe** | Composição linear de etapas: `entrada → etapa1 → etapa2 → saída`. Cada etapa recebe o output da anterior. |
| **Chain** | Encadeamento de chamadas de modelo e funções. Conceito central do LangChain. |
| **Dependency Injection** | Injetar dependências em um componente em vez de instanciá-las internamente. Usado nas factory functions dos nós: `createChatNode(llmClient, preferencesService)` — facilita testes com mocks. |
| **Fallback** | Estratégia alternativa quando a principal falha. O OpenRouter faz fallback automático entre modelos da lista. |

---

## LangChain

| Termo | Definição |
| --- | --- |
| **LangChain** | Framework que padroniza integração com LLMs. Abstrai diferenças entre providers (OpenAI, Anthropic, etc.) com interfaces uniformes. |
| **BaseChatModel** | Interface base que todo modelo LangChain implementa. Código escrito contra essa interface funciona com qualquer provider. |
| **ChatOpenAI** | Implementação do `BaseChatModel` para APIs compatíveis com OpenAI — incluindo OpenRouter. |
| **HumanMessage** | Tipo de mensagem que representa input do usuário. |
| **AIMessage** | Tipo de mensagem que representa resposta do modelo. |
| **SystemMessage** | Tipo de mensagem com instruções para o modelo (não é "fala" do usuário). |
| **RemoveMessage** | Instrução especial do LangGraph processada pelo reducer de mensagens — sinaliza que a mensagem com aquele `id` deve ser removida do state. |
| **Structured Output** | Resposta do modelo forçada a ser um JSON com schema definido. Elimina `JSON.parse` manual. Ver [langchain.md](./langchain.md). |
| **withStructuredOutput** | Método do LangChain que configura o modelo para retornar output validado por um schema Zod. Usa tool calling ou JSON mode dependendo do modelo. |

---

## LangGraph

| Termo | Definição |
| --- | --- |
| **LangGraph** | Extensão do LangChain para grafos de execução com estado persistente — permite ramificações, loops e retrocessos. |
| **StateGraph** | Classe principal do LangGraph. Você define nós e edges nela, depois compila. |
| **State** | Objeto compartilhado que todos os nós leem e atualizam. É a "ficha" que passa de nó em nó. |
| **Node (Nó)** | Função assíncrona que recebe o state e retorna um `Partial<State>` com o que mudou. |
| **Edge (Aresta)** | Conexão entre dois nós. Pode ser estática (sempre vai para o mesmo nó) ou condicional. |
| **Conditional Edge** | Aresta cujo destino é determinado por uma função de roteamento baseada no state. |
| **Reducer** | Função que define como cada campo do state é atualizado quando um nó retorna novo valor. O campo `messages` usa reducer especial configurado via `MessagesZodMeta`. |
| **MessagesZodMeta** | Metadado do LangGraph que configura o reducer de mensagens (append e RemoveMessage). Sem ele, o state substituiria o array inteiro a cada update. |
| **Checkpointer** | Salva snapshot completo do state após cada nó. Permite retomar o state entre invocações via `thread_id`. Implementado com `PostgresSaver`. |
| **Store** | Key-value persistido separado do checkpointer. Acessível pelos nós via `runtime.store`. Implementado com `PostgresStore`. |
| **thread_id** | Identificador de uma sessão de conversa. Threads diferentes = históricos isolados no checkpointer. |
| **Runtime** | Objeto injetado nos nós pelo LangGraph durante execução. Carrega `context` (passado no config do invoke). |
| **START / END** | Constantes do LangGraph que representam entrada e saída do grafo. |
| **compile()** | Transforma a definição do grafo em um workflow executável. Recebe checkpointer e store. |
| **LangSmith** | Plataforma de observabilidade do LangChain. Permite visualizar graficamente nós executados, prompts enviados e respostas recebidas. |

---

## Modelos e APIs

| Termo | Definição |
| --- | --- |
| **LLM** | Large Language Model — modelo de IA treinado em grandes volumes de texto. |
| **OpenRouter** | Plataforma que roteia chamadas para múltiplos LLMs (OpenAI, Anthropic, Google, etc.) com uma única API. Suporta fallback automático e roteamento por preço/throughput/latência. |
| **Token** | Unidade de texto que o modelo processa (≈ 0,75 palavras). Custo e limites de contexto são medidos em tokens. |
| **Temperatura** | Parâmetro entre 0 e 1+ que controla criatividade vs. consistência. 0 = determinístico. Alto = mais variado. |
| **Context window** | Limite máximo de tokens que um modelo consegue processar de uma vez (input + output). Conversas longas podem estourar esse limite. |
| **Throughput** | Velocidade de geração de tokens. Critério de roteamento no OpenRouter. |
| **modelKwargs** | Parâmetros extras enviados ao modelo além dos padrões — no projeto, passa a lista de modelos e configuração de provider para o OpenRouter. |

---

## Memória e persistência

| Termo | Definição |
| --- | --- |
| **Short-term memory** | Histórico da conversa atual — armazenado no Postgres via checkpointer do LangGraph. |
| **Long-term memory** | Preferências estáveis do usuário — armazenadas no SQLite, persistem entre sessões. |
| **Resumo incremental** | Técnica de manter um resumo que evolui: `resumo anterior + mensagens novas → novo resumo`. Evita reprocessar o histórico completo. Ver [conversation-summarization.md](./conversation-summarization.md). |
| **mergePreferences** | Método do `PreferencesService` que acumula preferências novas sem sobrescrever as existentes — usa `Set` para deduplicar gêneros e bandas. |
| **storeSummary** | Método do `PreferencesService` que sobrescreve as preferências com o resumo gerado pelo LLM (o LLM já recebeu o estado anterior como input). |
| **maxMessagesToSummary** | Configuração que define quantas mensagens no state disparam a sumarização automática. |

---

## Segurança

| Termo | Definição |
| --- | --- |
| **Prompt Injection** | Ataque onde input externo insere instruções que manipulam o comportamento do LLM. O análogo de SQL injection para modelos de linguagem. Ver [prompt-injection.md](./prompt-injection.md). |
| **Prompt Hijacking** | Variação de prompt injection onde o atacante "sequestra" o papel do modelo — faz a IA assumir uma identidade ou conjunto de regras diferente do definido no system prompt. |
| **Jailbreak** | Técnica de contornar as restrições de um modelo via narrativa, roleplay ou framing especial ("DAN", "modo de manutenção", "fins educacionais"). |
| **Guardrails** | Camada de validação externa que intercepta o prompt *antes* de chegar ao modelo executor. Um modelo validador (sem acesso a ferramentas) classifica o input como SAFE ou UNSAFE. |
| **Safeguard model** | Modelo especializado em classificar risco — mais rápido e barato que o executor. Não executa ações, só analisa texto. |
| **System Prompt** | Instrução base dada ao modelo antes da conversa. **Não é um firewall** — é texto, e pode ser ignorado ou sobrescrito por injeção. |
| **Prompt Template** | Template com variáveis (`{{nome}}`, `{{permissoes}}`) que substitui concatenação manual de strings. O LangChain aplica sanitizações internas, reduzindo superfície de ataque. |
| **MCP (Model Context Protocol)** | Protocolo que permite ao modelo executar ferramentas reais (ler arquivos, consultar banco, chamar APIs). Aumenta capacidade e superfície de ataque simultaneamente. |
| **STDIO** | Comunicação via entrada/saída padrão entre processos. Transporte usado pelo MCP Server. |
| **Variáveis de ambiente (.env)** | Configurações sensíveis (chaves de API, senhas) mantidas fora do código e não versionadas no Git. |

---

## Schema e validação

| Termo | Definição |
| --- | --- |
| **Schema** | Contrato formal de estrutura para a resposta do modelo — define campos, tipos e obrigatoriedade. |
| **Zod** | Biblioteca de validação e tipagem TypeScript. No projeto, define os schemas de ChatResponse, UserPreferences e ConversationSummary. |
| **Zod safeParse** | Variante do `parse` que retorna `{ success, data, error }` em vez de lançar exceção. Validação defensiva — ideal em nós do grafo. |
| **z.infer** | Extrai o tipo TypeScript de um schema Zod: `type GraphState = z.infer<typeof ChatStateAnnotation>`. |
| **withLangGraph** | Wrapper do LangGraph que associa metadados de reducer a um campo Zod do state. |

---

## Observabilidade

| Termo | Definição |
| --- | --- |
| **Langfuse** | Ferramenta de observabilidade para aplicações LLM. Open source, auto-hospedável. Mostra input/output, latência, tokens por requisição e tracing completo do fluxo. |
| **OpenTelemetry** | Padrão open source de coleta de telemetria (métricas, traces, logs). Langfuse integra com ele. |
| **Prompt Management** | Gerenciamento e versionamento de prompts fora do código (no Langfuse) — permite atualizar prompts sem redeploy. |
| **Evaluation Tests** | Testes que medem qualidade com scores e thresholds em vez de assert exato. Necessário para sistemas probabilísticos onde a mesma resposta pode vir com palavras diferentes. |

---

## RAG e bancos de dados

| Termo | Definição |
| --- | --- |
| **RAG** | Retrieval-Augmented Generation — técnica de buscar informações relevantes antes de gerar a resposta, enriquecendo o contexto do modelo. |
| **Neo4j** | Banco de dados orientado a grafos. Relacionamentos são cidadãos de primeira classe — eficiente para queries do tipo "quem comprou A tende a comprar o quê?". |
| **Cypher** | Linguagem de query do Neo4j. |
| **EXPLAIN** | Comando que valida a sintaxe de uma query sem executá-la. Usado pelo Cypher Executor antes de rodar a query gerada pelo LLM. |
| **Query Planner** | Nó que analisa a pergunta e decide se pode ser resolvida com uma query ou precisa ser decomposta em sub-perguntas. |
| **Multi-step** | Estratégia de decompor uma tarefa complexa em múltiplas etapas sequenciais, cada uma alimentando a próxima. |
| **Cypher Correction** | Nó que recebe uma query inválida + o erro do banco e gera uma versão corrigida. |

---

## Multimodalidade

| Termo | Definição |
| --- | --- |
| **Multimodal** | Modelo que processa múltiplos tipos de entrada: texto, imagem, áudio, PDF, vídeo. |
| **STT** | Speech-to-Text — converte áudio em texto antes de enviar ao modelo. |
| **TTS** | Text-to-Speech — converte resposta textual do modelo em áudio. |
| **Real-time audio** | Conexão contínua via WebSocket/WebRTC para conversa de voz fluida sem "turnos" explícitos. |

---

## Mercado

| Termo | Definição |
| --- | --- |
| **Equity** | Participação societária numa empresa (ações). Compensação comum para Founding Engineers. |
| **Founding Engineer** | Primeiro engenheiro de uma startup. Define stack, arquitetura e coloca a primeira versão em produção. Alto risco, potencial de equity significativo. |
