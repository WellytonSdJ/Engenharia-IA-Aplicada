import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
import { workerEvents } from "../events/constants.js";

console.log("Model training worker initialized");
let _globalCtx = {};

const WEIGHTS = {
  category: 0.4,
  color: 0.3,
  price: 0.2,
  age: 0.1,
};

// normalize continuous values (price, age) to 0-1 range
// why? keeps all features balanced so no one dominates training
// formula: (val - min) / (max - min)
// example: price = 129.99, minPrice=39.99, maxPrice=199.99 -> 0.56
const normalize = (value, min, max) => (value - min) / (max - min || 1);

function makeContext(catalog, users) {
  const ages = users.map((u) => u.age);
  const prices = catalog.map((p) => p.price);

  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);

  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  const colors = [...new Set(catalog.map((p) => p.color))];
  const categories = [...new Set(catalog.map((p) => p.category))];

  const colorsIndex = Object.fromEntries(
    colors.map((color, index) => {
      return [color, index];
    }),
  );

  const categoriesIndex = Object.fromEntries(
    categories.map((category, index) => {
      return [category, index];
    }),
  );

  // computar a m´[edia de idade dos comprados por produto
  // (ajuda a personalizar as recomendações)
  const midAge = (minAge + maxAge) / 2;
  const ageSums = {};
  const ageCounts = {};

  users.forEach((user) => {
    user.purchases.forEach((p) => {
      ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
      ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
    });
  });

  const productAvgAgeNorm = Object.fromEntries(
    catalog.map((product) => {
      const avg = ageCounts[product.name]
        ? ageSums[product.name] / ageCounts[product.name]
        : midAge;

      return [product.name, normalize(avg, minAge, maxAge)];
    }),
  );

  return {
    catalog,
    users,
    colorsIndex,
    categoriesIndex,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    numCategories: categories.length,
    numColors: colors.length,
    dimentions: 2 + colors.length + categories.length,
    productAvgAgeNorm,
  };
}
const oneHotWeighted = (index, length, weight) =>
  tf.oneHot(index, length).cast("float32").mul(weight);

function encondeProduct(product, ctx) {
  const price = tf
    .tensor1d([normalize(product.price, ctx.minPrice, ctx.maxPrice)])
    .mul(WEIGHTS.price);

  const age = tf
    .tensor1d([ctx.productAvgAgeNorm[product.name] || 0.5])
    .mul(WEIGHTS.age);

  const category = oneHotWeighted(
    ctx.categoriesIndex[product.category],
    ctx.numCategories,
    WEIGHTS.category,
  );

  const color = oneHotWeighted(
    ctx.colorsIndex[product.color],
    ctx.numColors,
    WEIGHTS.color,
  );

  return tf.concat([price, age, category, color]);
}

async function trainModel({ users }) {
  console.log("Training model with users:", users);
  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 50 },
  });

  const catalog = await (await fetch("/data/products.json")).json();

  const context = makeContext(catalog, users);
  context.productVectors = catalog.map((product) => {
    return {
      name: product.name,
      meta: { ...product },
      vector: encondeProduct(product, context).dataSync(), // convertendo tensor para array normal
    };
  });

  debugger;
  _globalCtx = context;
  postMessage({
    type: workerEvents.trainingLog,
    epoch: 1,
    loss: 1,
    accuracy: 1,
  });

  setTimeout(() => {
    postMessage({
      type: workerEvents.progressUpdate,
      progress: { progress: 100 },
    });
    postMessage({ type: workerEvents.trainingComplete });
  }, 1000);
}
function recommend(user, ctx) {
  console.log("will recommend for user:", user);
  // postMessage({
  //     type: workerEvents.recommend,
  //     user,
  //     recommendations: []
  // });
}

const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx),
};

self.onmessage = (e) => {
  const { action, ...data } = e.data;
  if (handlers[action]) handlers[action](data).catch((err) => console.error("[worker error]", err));
};
