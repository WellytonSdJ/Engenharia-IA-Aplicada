# Neural Network — Classificador de Categorias de Usuário

Rede neural simples construída com [TensorFlow.js](https://www.tensorflow.org/js) que classifica pessoas em três categorias: **premium**, **medium** e **basic**. O objetivo é demonstrar, de forma mínima, os conceitos fundamentais de redes neurais: codificação de features, treinamento e inferência.

## O que o modelo faz

- Recebe dados de uma pessoa (idade, cor favorita, localização)
- Converte esses dados para um vetor numérico normalizado
- Treina uma rede neural densa com esses exemplos
- Prevê a categoria mais provável para uma nova pessoa, com probabilidade para cada classe

## Pré-processamento — como os dados entram na rede

Redes neurais só entendem números. Os dados brutos precisam ser convertidos antes do treino:

### Normalização da idade

A idade é convertida para o intervalo `[0, 1]` usando min-max scaling:

```text
idade_normalizada = (idade - idade_min) / (idade_max - idade_min)
```

Exemplo: idade `28`, com mínimo `25` e máximo `40`:

```text
(28 - 25) / (40 - 25) = 0.2
```

### One-hot encoding de categorias

Cores e localizações são variáveis categóricas — não têm ordem numérica. Cada valor é representado como um vetor com `1` na posição correspondente e `0` nas demais:

| Valor    | Vetor one-hot |
| -------- | ------------- |
| azul     | `[1, 0, 0]`   |
| vermelho | `[0, 1, 0]`   |
| verde    | `[0, 0, 1]`   |

| Localização | Vetor one-hot |
| ----------- | ------------- |
| São Paulo   | `[1, 0, 0]`   |
| Rio         | `[0, 1, 0]`   |
| Curitiba    | `[0, 0, 1]`   |

### Vetor de entrada final

Cada pessoa vira um vetor de **7 posições**:

```text
[idade_norm, azul, vermelho, verde, são_paulo, rio, curitiba]
```

Exemplos do dataset de treino:

| Pessoa | Vetor | Label |
| --- | --- | --- |
| Erick | [0.33, 1, 0, 0, 1, 0, 0] | [1, 0, 0] premium |
| Ana | [0, 0, 1, 0, 0, 1, 0] | [0, 1, 0] medium |
| Carlos | [1, 0, 0, 1, 0, 0, 1] | [0, 0, 1] basic |

## Arquitetura da rede

```text
Entrada [7]
    │
    ▼
Dense(80, relu)     ← captura padrões não-lineares nos 7 features
    │
    ▼
Dense(3, softmax)   ← converte em 3 probabilidades que somam 1
    │
    ▼
[premium%, medium%, basic%]
```

| Camada  | Neurônios | Ativação | Por quê                                                                          |
| ------- | --------- | -------- | -------------------------------------------------------------------------------- |
| Entrada | 80        | ReLU     | Dataset pequeno exige mais neurônios para aprender os poucos padrões disponíveis |
| Saída   | 3         | Softmax  | Normaliza os scores em probabilidades; a soma sempre é 1                         |

### Compilação e treinamento

| Parâmetro  | Valor                     | Explicação                                                                      |
| ---------- | ------------------------- | ------------------------------------------------------------------------------- |
| Otimizador | Adam                      | Ajusta pesos levando em conta histórico de gradientes — converge mais rápido    |
| Loss       | `categoricalCrossentropy` | Mede o quão distante a previsão está do label correto (one-hot)                 |
| Épocas     | 100                       | Número de passes completos pelo dataset                                         |
| Shuffle    | `true`                    | Embaralha os dados a cada época para evitar que a ordem influence o aprendizado |

## Saída do modelo

O modelo retorna a probabilidade para cada categoria, ordenadas da maior para a menor:

```text
premium (12.45%)
medium (23.10%)
basic (64.45%)
```

## Estrutura do projeto

```text
01-neural-network/
├── index.js       # Dataset, treino e predição em um único arquivo
└── package.json   # Dependência: @tensorflow/tfjs
```

## Como executar

```bash
npm install
node index.js
```

O script treina o modelo com 3 exemplos e imprime a previsão para Wellington (28 anos, verde, Curitiba).
