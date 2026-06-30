# Glossário

Referência rápida. Para profundidade, vá ao documento específico de cada conceito.

---

## RAG e bancos de dados

| Termo | Definição |
| --- | --- |
| **RAG** | Retrieval-Augmented Generation — buscar informações relevantes antes de gerar a resposta, enriquecendo o contexto do LLM com dados reais. |
| **Text-to-Cypher** | Técnica de converter linguagem natural em queries Cypher usando um LLM. Análogo ao Text-to-SQL para bancos relacionais. |
| **Neo4j** | Banco de dados orientado a grafos. Nós, relacionamentos e propriedades são cidadãos de primeira classe — eficiente para perguntas que envolvem conexões entre entidades. |
| **Cypher** | Linguagem de query do Neo4j. A sintaxe espelha visualmente o grafo: `(nó)-[:RELACIONAMENTO]->(nó)`. |
| **EXPLAIN** | Prefixo do Neo4j que valida sintaxe e plano de execução sem rodar a query. Usado para detectar erros antes de executar no banco real. |
| **Schema-aware prompting** | Injetar o schema real do banco no prompt de geração de query. Evita que o LLM invente nomes de propriedades ou relacionamentos que não existem. |
| **Grafo** | Estrutura de dados com nós e arestas (relacionamentos). Neo4j armazena dados nativamente nesse formato. |
| **MATCH** | Cláusula Cypher para encontrar padrões no grafo. Equivale ao FROM/JOIN do SQL. |
| **elementId()** | Função Cypher que retorna o identificador de um elemento. Substitui a função depreciada `id()` no Neo4j 5+. |
| **UNWIND** | Cláusula Cypher que expande uma lista em linhas individuais — usada no seed para inserção em lote. |

---

## Arquitetura e padrões

| Termo | Definição |
| --- | --- |
| **LangGraph** | Framework para construir fluxos de execução com LLMs onde o estado persiste entre nós. Permite ramificações, loops e retrocessos via grafos dirigidos. |
| **StateGraph** | Classe principal do LangGraph. Você define nós e edges, depois compila em um workflow executável. |
| **State** | Objeto central que flui pelo grafo. Todo nó lê e retorna um `Partial<State>`. No projeto: `GraphState`. |
| **Node (Nó)** | Função assíncrona que recebe o state e retorna um `Partial<State>` com o que mudou. |
| **Edge (Aresta)** | Conexão entre dois nós. Estática (sempre vai para o mesmo destino) ou condicional (baseada no state). |
| **Conditional Edge** | Aresta cujo destino é uma função que lê o state e retorna uma string mapeada para um nó. |
| **Self-correction** | Loop automático onde o sistema corrige seu próprio output: query falha → LLM corrige → tenta novamente. Padrão agentic. |
| **Multi-step** | Estratégia de decompor uma pergunta complexa em sub-perguntas independentes, executar cada uma e sintetizar os resultados. |
| **Dependency Injection** | Injetar serviços em factory functions em vez de instanciá-los internamente. Facilita testes com mocks: `createCypherGeneratorNode(neo4jService, llmService)`. |
| **Query Planner** | Nó que analisa a complexidade da pergunta e decide se uma única query resolve ou se precisa de decomposição multi-step. |

---

## LangChain e LLM

| Termo | Definição |
| --- | --- |
| **LangChain** | Framework que padroniza integração com LLMs. Abstrai diferenças entre providers com interfaces uniformes. |
| **ChatOpenAI** | Implementação LangChain para APIs compatíveis com OpenAI — incluindo OpenRouter. |
| **Structured Output** | Resposta do LLM forçada a ser um JSON validado por um schema Zod. Elimina parsing manual e garante tipo em tempo de execução. |
| **withStructuredOutput** | Método LangChain que configura o modelo para retornar output validado por Zod. Usa tool calling ou JSON mode dependendo do modelo. |
| **OpenRouter** | Plataforma que roteia chamadas para múltiplos LLMs com uma única API. Suporta roteamento por throughput, latência ou preço. |
| **Throughput** | Velocidade de geração de tokens. Critério de roteamento usado no projeto: `provider: { sort: { by: 'throughput' } }`. |
| **Temperature** | Parâmetro entre 0 e 2 que controla criatividade vs. consistência. No projeto: `0.7`. |
| **Few-shot** | Técnica de incluir exemplos no prompt para guiar o LLM. O `cypherGenerator` usa 5 exemplos de pergunta→Cypher. |
| **System prompt** | Instrução base dada ao LLM antes da conversa — define papel, regras e formato de saída. |

---

## Schema e validação

| Termo | Definição |
| --- | --- |
| **Zod** | Biblioteca de validação e tipagem TypeScript. Todos os schemas de saída LLM neste projeto são Zod. |
| **QueryAnalysisSchema** | Schema Zod da saída do `queryPlanner`: `complexity`, `requiresDecomposition`, `subQuestions`, `reasoning`. |
| **CypherQuerySchema** | Schema Zod da saída do `cypherGenerator`: campo `query` com a string Cypher. |
| **CypherCorrectionSchema** | Schema Zod da saída do `cypherCorrection`: `correctedQuery` + `explanation`. |
| **AnalyticalResponseSchema** | Schema Zod da saída do `analyticalResponse`: `answer` (prosa) + `followUpQuestions` (array). |
| **z.infer** | Extrai o tipo TypeScript de um schema Zod: `type Output = z.infer<typeof CypherQuerySchema>`. |

---

## Domínio (academia online)

| Termo | Definição |
| --- | --- |
| **Student** | Nó no Neo4j. Propriedades: `id` (UUID), `name`, `email`, `phone`. |
| **Course** | Nó no Neo4j. Propriedades: `name`, `url`. |
| **PURCHASED** | Relacionamento `(Student)→(Course)`. Propriedades: `status` (paid/refunded), `paymentMethod` (pix/credit_card), `paymentDate`, `amount`. |
| **PROGRESS** | Relacionamento `(Student)→(Course)`. Propriedade: `progress` (0-100). Só existe para compras com `status = "paid"`. |
| **status: "paid"** | Compra confirmada. É o filtro padrão para cálculos de receita. |
| **status: "refunded"** | Compra reembolsada. Deve ser excluída de cálculos de receita. |
| **salesContext** | Arquivo de prompts que injeta as regras de negócio do domínio nos prompts de geração e correção de Cypher. |

---

## Infraestrutura

| Termo | Definição |
| --- | --- |
| **Fastify** | Framework HTTP Node.js de alta performance. Usado para servir o endpoint `POST /sales`. |
| **Docker Compose** | Ferramenta para definir e rodar containers. Usado para subir o Neo4j localmente. |
| **APOC** | Biblioteca de procedimentos e funções para o Neo4j. Habilitada via plugin no docker-compose. |
| **Bolt** | Protocolo binário de comunicação do Neo4j. Porta 7687. Mais eficiente que HTTP para queries programáticas. |
| **neo4j-driver** | Driver oficial Node.js para Neo4j via Bolt. Usado pelo `Neo4jService` para queries diretas. |
| **Neo4jGraph** | Abstração do LangChain Community que encapsula o driver e expõe `.schema` e `.query()`. |
| **Faker** | Biblioteca para geração de dados fictícios. Usada no seed para criar alunos com nome, email e telefone realistas. |
| **LangSmith** | Plataforma de observabilidade do LangChain — mostra nós executados, prompts enviados, respostas, latência e tokens. Opcional neste projeto. |
