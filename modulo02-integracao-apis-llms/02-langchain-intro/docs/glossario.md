# Glossário — LangChain Intro

Termos do projeto anterior (`01-smart-model-router-gateway`) não são repetidos aqui.

---

## LangGraph

| Termo | Definição |
| --- | --- |
| **StateGraph** | Classe central do LangGraph. Recebe o schema do estado e permite adicionar nós e edges para montar o grafo. |
| **Estado (State)** | Objeto compartilhado que flui pelo grafo. Cada nó pode ler e retornar versões atualizadas. Definido com Zod neste projeto. |
| **Nó (Node)** | Função que recebe o estado e retorna uma versão atualizada. Cada nó tem uma responsabilidade única. |
| **Edge** | Conexão direcional entre dois nós. Diz ao LangGraph quem chama quem. |
| **Edge condicional** | Edge que usa uma função de roteamento para decidir o próximo nó com base no estado atual. |
| **START** | Constante especial do LangGraph que representa o início do grafo. Todo grafo tem uma edge `START → primeiro_nó`. |
| **END** | Constante especial que representa o fim do grafo. Quando um nó aponta para `END`, a execução para. |
| **compile()** | Método que valida e prepara o grafo para execução. Retorna o objeto que será chamado com `.invoke()`. |
| **invoke()** | Executa o grafo com um estado inicial e retorna o estado final. |
| **Reducer** | Função que define como um campo do estado é atualizado quando múltiplos nós escrevem nele. `MessagesZodMeta` implementa um reducer de acumulação para mensagens. |
| **withLangGraph()** | Wrapper do `@langchain/langgraph/zod` que associa um schema Zod a metadados de comportamento do LangGraph (como o reducer). |
| **MessagesZodMeta** | Metadados do LangGraph para o campo `messages`. Instrui o framework a acumular mensagens em vez de substituir. |
| **factory.ts** | Arquivo que exporta a função `graph()` para o servidor de desenvolvimento do LangGraph (`langgraph:serve`). |

## LangChain Messages

| Termo | Definição |
| --- | --- |
| **BaseMessage** | Tipo base para todas as mensagens do LangChain. Tem `.text` / `.content` para acessar o conteúdo. |
| **HumanMessage** | Mensagem do usuário humano. Representa o input de quem usa o sistema. |
| **AIMessage** | Mensagem gerada pelo modelo (ou pelo sistema, como neste projeto). Representa o output. |
| **SystemMessage** | Instrução de sistema enviada ao LLM para configurar seu comportamento. Não aparece neste projeto mas é comum nos seguintes. |

## Arquitetura

| Termo | Definição |
| --- | --- |
| **Grafo de estado** | Padrão arquitetural onde um estado compartilhado flui por uma rede de funções conectadas (nós). O LangGraph implementa esse padrão. |
| **Roteamento condicional** | Decisão de qual caminho seguir baseada no estado atual do grafo, em vez de if/else no código imperativo. |
| **Intent detection** | Identificação da intenção do usuário a partir do input. Neste projeto é determinístico (substring). No `03-medical-appointment`, passa a ser feito via LLM. |
