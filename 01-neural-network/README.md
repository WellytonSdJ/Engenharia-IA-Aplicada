# Neural Network — Classificador de Categorias de Usuário

Rede neural simples construída com [TensorFlow.js](https://www.tensorflow.org/js) que classifica pessoas em três categorias: **premium**, **medium** e **basic**.

## O que o modelo faz

O modelo recebe dados de uma pessoa (idade, cor favorita e localização) e prevê a qual categoria ela pertence, retornando a probabilidade para cada uma das três opções.

### Entrada

Vetor de 7 posições com os dados normalizados e codificados em one-hot:

| Posição | Dado               | Exemplo           |
|---------|--------------------|-------------------|
| 0       | Idade normalizada  | `0.2`             |
| 1–3     | Cor (azul/vermelho/verde) | `[0, 0, 1]` |
| 4–6     | Localização (SP/Rio/Curitiba) | `[0, 0, 1]` |

### Saída

Probabilidade para cada categoria:

```
premium (12.45%)
medium (23.10%)
basic (64.45%)
```

## Arquitetura

- **Camada de entrada:** 80 neurônios, ativação ReLU, `inputShape: [7]`
- **Camada de saída:** 3 neurônios, ativação Softmax
- **Otimizador:** Adam
- **Função de perda:** Categorical Crossentropy
- **Épocas de treino:** 100

## Como executar

```bash
npm install
node index.js
```
