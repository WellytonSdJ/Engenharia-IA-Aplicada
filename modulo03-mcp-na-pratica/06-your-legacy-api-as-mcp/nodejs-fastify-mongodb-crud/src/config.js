import { randomUUID } from 'node:crypto'

// Em ambiente de teste, DB_NAME não é definido — o UUID garante um banco isolado por execução,
// evitando colisões entre testes paralelos ou reaproveitamento de dados sujos.
const randomName = randomUUID().slice(0, 4)

const dbUser = process.env.DB_USER || 'root'
const dbPassword = process.env.DB_PASSWORD || 'example'
const dbHost = process.env.DB_HOST || 'localhost'
const dbPort = process.env.DB_PORT || '27017'
const dbName = process.env.DB_NAME || `${randomName}-test`

const config = {
    dbName,
    collection: 'customers',
    dbURL: `mongodb://${dbUser}:${dbPassword}@${dbHost}:${dbPort}`
}

export default config
