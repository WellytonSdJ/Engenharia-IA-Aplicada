# Sistema de Recomendação para E-commerce

Aplicação web que recomenda produtos a usuários usando uma rede neural treinada no browser com [TensorFlow.js](https://www.tensorflow.org/js). O treinamento ocorre em um **Web Worker** para não bloquear a interface.

## Como executar

```bash
npm install
npm start
# acesse http://localhost:3000
```

## Arquitetura

O projeto segue o padrão **MVC** com comunicação via eventos:

```
index.html
└── src/
    ├── index.js                     # Bootstrap — instancia controllers, views e serviços
    ├── controller/
    │   ├── UserController.js        # Seleção e exibição de usuários
    │   ├── ProductController.js     # Listagem de produtos e registro de compras
    │   ├── ModelTrainingController.js # Dispara treino e recomendações
    │   ├── WorkerController.js      # Ponte entre eventos da UI e o Web Worker
    │   └── TFVisorController.js     # Visualizações do TF Visor
    ├── service/
    │   ├── UserService.js           # CRUD de usuários via sessionStorage
    │   └── ProductService.js        # Leitura de produtos do JSON
    ├── view/                        # Renderização de DOM
    ├── events/
    │   ├── constants.js             # Nomes dos eventos (UI e Worker)
    │   └── events.js                # Barramento de eventos (pub/sub)
    ├── workers/
    │   └── modelTrainingWorker.js   # Codificação de features + treino + inferência
    └── data/
        ├── users.json
        └── products.json
```

## Como o modelo funciona

### Codificação de features

Cada produto é representado por um vetor numérico com pesos pré-definidos:

| Feature                  | Tipo            | Peso |
| ------------------------ | --------------- | ---- |
| Preço (normalizado)      | Contínuo [0–1]  | 0.2  |
| Idade média compradores  | Contínuo [0–1]  | 0.1  |
| Categoria                | One-hot         | 0.4  |
| Cor                      | One-hot         | 0.3  |

O vetor do **usuário** é calculado como a média dos vetores dos produtos que ele comprou. Se não tiver compras, usa apenas a idade normalizada com zeros nas demais posições.

### Dados de treino

Para cada par `(usuário, produto)` de usuários com histórico de compras, é criado um exemplo:

- **Entrada:** `[vetor_usuário, vetor_produto]` (concatenados)
- **Label:** `1` se o usuário comprou o produto, `0` caso contrário

### Rede neural

| Camada       | Neurônios | Ativação |
| ------------ | --------- | -------- |
| Entrada      | 128       | ReLU     |
| Oculta 1     | 64        | ReLU     |
| Oculta 2     | 32        | ReLU     |
| Saída        | 1         | Sigmoid  |

- **Otimizador:** Adam (lr = 0.01)
- **Loss:** Binary Cross Entropy
- **Épocas:** 100 · **Batch size:** 32

A saída é um score entre 0 e 1 — quanto maior, maior a probabilidade de o usuário gostar do produto.

### Fluxo de recomendação

1. Ao carregar a página, o Worker treina o modelo automaticamente com todos os usuários
2. O usuário seleciona um perfil na UI e clica em **Recomendar**
3. O Worker codifica o usuário selecionado e roda `model.predict()` em todos os produtos
4. Os produtos são ordenados pelo score e enviados de volta para a UI

## Funcionalidades

- Seleção de perfil de usuário com histórico de compras
- Listagem de produtos com botão **Comprar** (persiste na sessão)
- Treino da rede neural no browser via Web Worker (não bloqueia a UI)
- Barra de progresso de treinamento em tempo real
- Recomendações personalizadas ordenadas por score
- Visualização de métricas de treino com TF Visor
