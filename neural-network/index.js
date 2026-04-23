import * as tf from "@tensorflow/tfjs";

async function trainModel(inputXs, outputYs) {
  // Criamos um modelo sequencial
  const model = tf.sequential();

  //primeira camada da rede:
  // entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)

  // 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino.
  // quanto mais neuronios, mais complexidade a rede pode aprender
  // e consequentemente, mais processamento ela vai usar

  // a ReLu age como um filtro:
  // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
  // se a informação chegou nesse neuronio é positiva, passa pra frente!
  // se for zero ou negativa, pode jogar fora, não vai servir pra nada.
  model.add(
    tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" })

    // Saída: 3 neuronios
    // um para cada catetgoria (premium, medium, basic)
  );

  // activation: softmax normaliza a saída em probabilidades
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  //Compilando o modelo
  // optimizer Adam (Adaptative Moment Estimation)
  // é um treinador pessoal moderno pra redes neurais:
  // ajusta os pesos de forma eficiente, levando em conta a média dos gradientes anteriores e a variância deles.
  // aprende com historicos de erros e acertos.

  // loss: categoricalCrossentropy
  // Ele compara o que o modelo "acha" (os scroes de cada categoria)
  // com a resposta certa.
  // a categoria premium será sempre [1, 0, 0], medium [0, 1, 0] e basic [0, 0, 1]

  // quanto mais distante da previsão do modelo da resposta correta, maior o valor da perda.
  // Exemplo classico: classificação de imagens, recomendação, categorização de usuário.
  // qualquer coisa em que a resposta certa é apenas uma entre várias opções possíveis.

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Treinamento do modelo
  // verbose: desabilita o log interno (ele usa só callback)
  // epochs: número de vezes que o modelo vai passar por todo o conjunto de dados
  // embaralha os dados a cada época para evitar padrões de aprendizado
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: {
      // onEpochEnd: (epoch, logs) =>
      //   console.log(`Epoch: ${epoch}: loss: ${logs.loss}`),
    },
  });

  return model;
}

async function predict(model, pessoa) {
  // transforma o array js para o tensor (tfjs)
  const tfInput = tf.tensor2d(pessoa);

  // faz a predição (output será um vetor de 3 probabilidades)
  const pred = model.predict(tfInput);
  const predArray = await pred.array();
  return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// quanto mais dado melhor!
// assim o algoritmo consegue entender melhor os padrões complexos dos dados.
const model = await trainModel(inputXs, outputYs);

// TESTE: Prevendo a categoria de uma nova pessoa
const pessoa = {
  nome: "Wellington",
  idade: 28,
  cor: "verde",
  localizacao: "Curitiba",
};
// normalizando a idade da nova pessoa usando o mesmo padrão de treino.
// Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2
const pessoaTensorNormalizada = [
  [
    0.2, // idade normalizada
    0, // azul
    0, // vermelho
    1, // verde
    0, // São Paulo
    0, // Rio
    1, // Curitiba
  ],
];

const predictions = await predict(model, pessoaTensorNormalizada);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
  .join("\n");

console.log(results);
