# Sistema de Recomendação para E-commerce

Aplicação web que recomenda produtos a usuários usando uma rede neural treinada no browser com [TensorFlow.js](https://www.tensorflow.org/js). O treinamento ocorre em um **Web Worker** para não bloquear a interface, e a arquitetura segue o padrão **MVC** com comunicação via barramento de eventos pub/sub.

## O que o projeto faz

- Carrega perfis de usuários e histórico de compras de arquivos JSON
- Treina automaticamente uma rede neural no browser ao abrir a página
- Exibe barra de progresso de treinamento em tempo real (via Web Worker)
- Permite selecionar um perfil de usuário e visualizar seu histórico de compras
- Gera recomendações personalizadas de produtos ordenadas por score do modelo
- Visualiza métricas de treino (loss, accuracy por época) com TF Visor

## Arquitetura — MVC com barramento de eventos

O projeto separa responsabilidades em controllers, views e services. A comunicação entre camadas usa um barramento de eventos pub/sub (`Events`) em vez de chamadas diretas — isso desacopla a UI da lógica de ML e do Worker.

```text
index.html
    │
    └── src/index.js  (bootstrap — instancia tudo e dispara treino inicial)
            │
            ├── UserController      ←→ Events ←→ ProductController
            ├── ModelController     ←→ Events ←→ WorkerController
            └── TFVisorController   ←→ Events
                                              │
                                    WorkerController
                                              │
                                    Web Worker (modelTrainingWorker.js)
                                              │
                                    TensorFlow.js (treino + inferência)
```

### Responsabilidades por camada

| Camada | Arquivo(s) | Responsabilidade |
| --- | --- | --- |
| Bootstrap | `src/index.js` | Instancia serviços, views, controllers e dispara o treino inicial |
| Controller | `UserController.js` | Seleção de usuário, exibição de detalhes e histórico de compras |
| Controller | `ProductController.js` | Listagem de produtos e registro de compras na sessão |
| Controller | `ModelTrainingController.js` | Escuta eventos de treino completo e despacha pedidos de recomendação |
| Controller | `WorkerController.js` | Ponte entre o barramento de eventos da UI e o Web Worker |
| Controller | `TFVisorController.js` | Recebe logs de treino e atualiza as visualizações do TF Visor |
| Service | `UserService.js` | CRUD de usuários via `sessionStorage` (persiste compras durante a sessão) |
| Service | `ProductService.js` | Leitura do catálogo de produtos do JSON |
| View | `UserView.js`, `ProductView.js`... | Renderização de DOM; registram callbacks que disparam eventos |
| Events | `events/constants.js` | Nomes de eventos como constantes (evita strings soltas no código) |
| Events | `events/events.js` | Barramento pub/sub — `dispatch*` e `on*` para cada evento |
| Worker | `workers/modelTrainingWorker.js` | Codificação de features, treinamento da rede neural e inferência |

### Fluxo de eventos

```text
[Página carrega]
    └── WorkerController.triggerTrain(users)
            └── Worker recebe trainModel
                    ├── postMessage(progressUpdate)  →  Events.dispatchProgressUpdate  →  UI (barra)
                    ├── postMessage(trainingLog)     →  Events.dispatchTFVisLogs       →  TFVisor
                    └── postMessage(trainingComplete)→  Events.dispatchTrainingComplete→  ModelController

[Usuário clica "Recomendar"]
    └── Events.dispatchRecommend(user)
            └── WorkerController.triggerRecommend(user)
                    └── Worker recebe recommend
                            └── postMessage(recommend) → Events.dispatchRecommendationsReady → ProductView
```

## Como o modelo funciona

### Codificação de features

O Worker converte produtos e usuários em vetores numéricos antes do treino. Cada feature recebe um peso que controla sua influência na recomendação:

| Feature | Tipo | Peso | Codificação |
| --- | --- | --- | --- |
| Preço | Contínuo [0–1] | 0.2 | Min-max normalization × peso |
| Idade média compradores | Contínuo [0–1] | 0.1 | Min-max normalization × peso |
| Categoria | Categórico | 0.4 | One-hot encoding × peso |
| Cor | Categórico | 0.3 | One-hot encoding × peso |

**Dimensão do vetor:** `2 + num_categorias + num_cores` (calculado dinamicamente a partir dos dados)

**Vetor de usuário:** média dos vetores de todos os produtos que ele comprou. Se não tiver compras, usa só a idade normalizada com zeros nas demais posições.

**Dados de treino:** para cada par `(usuário, produto)` de usuários com histórico, cria um exemplo com `[vetor_usuário, vetor_produto]` concatenados e label `1` (comprou) ou `0` (não comprou).

### Arquitetura da rede neural

```text
Entrada [inputDim]        ← vetor_usuário + vetor_produto concatenados
    │
    ▼
Dense(128, relu)          ← detecta padrões amplos de compatibilidade
    │
    ▼
Dense(64, relu)           ← comprime e refina as combinações relevantes
    │
    ▼
Dense(32, relu)           ← destila os padrões mais fortes
    │
    ▼
Dense(1, sigmoid)         ← score entre 0 e 1 (probabilidade de gostar)
```

| Parâmetro  | Valor                 | Explicação                                              |
| ---------- | --------------------- | ------------------------------------------------------- |
| Otimizador | Adam (lr = 0.01)      | Convergência eficiente com taxa de aprendizado adaptiva |
| Loss       | Binary Cross Entropy  | Problema binário: comprou (1) ou não comprou (0)        |
| Épocas     | 100                   | Passes completos pelo dataset de treino                 |
| Batch size | 32                    | Exemplos processados por atualização de pesos           |

### Fluxo de recomendação

1. Worker encoda o usuário selecionado em um vetor de features
2. Para cada produto, concatena `[vetor_usuário, vetor_produto]`
3. Roda `model.predict()` em todos os pares de uma vez (tensor 2D)
4. Ordena os produtos pelo score (0–1) em ordem decrescente
5. Envia a lista ordenada via `postMessage` para a UI exibir

## Estrutura do projeto

```text
02-Sistemas-Recomendacao/
├── index.html
├── style.css
├── src/
│   ├── index.js                          # Bootstrap — instancia e conecta todas as partes
│   ├── controller/
│   │   ├── UserController.js             # Seleção de usuário e histórico de compras
│   │   ├── ProductController.js          # Listagem de produtos e registro de compras
│   │   ├── ModelTrainingController.js    # Despacha recomendações após treino completo
│   │   ├── WorkerController.js           # Ponte UI ↔ Web Worker
│   │   └── TFVisorController.js          # Métricas de treino no TF Visor
│   ├── service/
│   │   ├── UserService.js                # CRUD de usuários via sessionStorage
│   │   └── ProductService.js             # Leitura do catálogo JSON
│   ├── view/                             # Renderização de DOM por domínio
│   ├── events/
│   │   ├── constants.js                  # Nomes de eventos (UI e Worker)
│   │   └── events.js                     # Barramento pub/sub
│   └── workers/
│       └── modelTrainingWorker.js        # Codificação de features + treino + inferência
└── data/
    ├── users.json                        # Perfis de usuário com histórico de compras
    └── products.json                     # Catálogo de produtos com categoria, cor e preço
```

## Requisitos

| Requisito    | Versão mínima | Observação                                          |
| ------------ | ------------- | --------------------------------------------------- |
| **Node.js**  | 18+           | Apenas para rodar o servidor de desenvolvimento     |
| **npm**      | 10+           | Incluído no Node.js                                 |
| **Browser**  | Moderno       | Suporte a Web Workers com `type: 'module'` e ES2020 |

Não há backend — toda a lógica de ML roda no browser via TensorFlow.js carregado por CDN.

## Como executar

```bash
npm install
npm start
# acesse http://localhost:3000
```

Ao abrir a página, o treino começa automaticamente com todos os usuários do `data/users.json`. A barra de progresso indica o andamento. Quando concluir, selecione um usuário e clique em **Recomendar**.
