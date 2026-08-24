import { MongoClient } from 'mongodb';
import config from './config.js';

async function connect() {
        const dbClient = new MongoClient(config.dbURL);

        const db = dbClient.db(config.dbName);
        // Retorna a referência para a collection — operações de CRUD são feitas diretamente nela.
        const dbUsers = db.collection(config.collection);

        console.log('Connected to the database');

        return { collections: { dbUsers }, dbClient };

}

// Abstrai a lógica de conexão atrás de getDb() — o consumidor (index.js) não sabe como conectar,
// só recebe collections prontas para usar.
async function getDb() {
    const { collections, dbClient } = await connect();
    return { collections, dbClient };
}

export {
    getDb
}
