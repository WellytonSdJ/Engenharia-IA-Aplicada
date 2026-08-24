import { randomUUID } from 'node:crypto'
import { REQUESTS_PER_MINUTE } from './config.js'

// Em produção estas credenciais viriam de um banco de dados ou vault.
// Hardcoded aqui para simplificar o exercício de aprender os conceitos de auth.
export const authUsers = [{
    username: 'erickwendel',
    password: '123123',
    role: 'admin',
},
{
    username: 'ananeri',
    password: '1234',
    role: 'member'
}]

// Em produção: variável de ambiente ou secret manager — NUNCA hardcoded.
export const JWT_SECRET = 'supersecret'
// Segredo adicional exigido para emitir service tokens — só quem sabe este pode gerar tokens M2M.
export const ADMIN_SUPER_SECRET = 'AM I THE BOSS?'

// Map em memória: serviceToken → { username, role }.
// Em produção usaria Redis ou banco de dados para que os tokens sobrevivam a restarts.
const issuedServiceTokens = new Map()

export const rateLimitOptions = {
    max: REQUESTS_PER_MINUTE,
    timeWindow: '1 minute',
    // keyGenerator: identifica cada "cliente" para o rate limiter.
    // Usa o token do header como chave — cada token tem sua própria janela de limite.
    // Fallback para IP quando não há token (ex: rotas públicas).
    keyGenerator: (request) => request.headers?.authorization?.replace(/bearer /i, '') ?? request.ip,
}

export function initAuthRoute(fastify) {
    // onRequest: hook global executado antes de QUALQUER rota (incluindo preHandler e o handler em si).
    // Funciona como middleware de autenticação centralizado.
    fastify.addHook('onRequest', async (request, reply) => {
        // Rotas públicas são excluídas da verificação de token.
        const publicRoutes = [
            '/v1/health',
            '/v1/auth/login',
            '/v1/auth/service-token'
        ]
        if (publicRoutes.includes(request.originalUrl)) return

        const token = request.headers?.authorization?.replace(/bearer /i, '')

        // Verifica primeiro se é um service token (M2M) — não passa pelo jwtVerify().
        // Service tokens são opacos (UUID), JWTs são decodificáveis com a chave pública.
        const serviceUser = issuedServiceTokens.get(token)
        if (serviceUser) {
            request.user = serviceUser  // injeta o user no request, igual ao jwtVerify() faria
            return
        }

        try {
            // jwtVerify() do @fastify/jwt: decodifica e valida o JWT, injeta o payload em request.user.
            await request.jwtVerify()
        } catch (error) {
            console.error('[onRequest]', error)
            return reply.code(401).send({ message: 'Unauthorized' })
        }

    })

    // Login com usuário/senha → devolve JWT de curta duração (para uso humano, no browser).
    fastify.post('/v1/auth/login',
        {
            schema: {
                body: {
                    type: 'object',
                    required: ['username', 'password'],
                    properties: {
                        username: { type: 'string' },
                        password: { type: 'string' },
                    }
                },
                response: {
                    200: {
                        type: 'object',
                        properties: {
                            token: { type: 'string' },
                        },
                    },
                    401: {
                        type: 'object',
                        properties: {
                            message: { type: 'string' },
                        },
                    },
                },
            }
        },
        async (request, reply) => {
            const { username, password } = request.body
            const user = authUsers.find(
                user =>
                    user.username.toLocaleLowerCase() === username.toLocaleLowerCase() &&
                    user.password === password
            )

            if (!user) {
                return reply.code(401).send({ message: 'Invalid credentials' })
            }

            // jwt.sign() assina o payload com JWT_SECRET → devolve token string.
            // O role fica dentro do token — sem precisar consultar banco a cada request.
            const token = fastify.jwt.sign({ username, role: user.role })

            return reply.send({ token })
        })

    // Service Token: para aplicações máquina-a-máquina (M2M), como o customers-mcp.
    // Diferente do JWT: é um UUID opaco, armazenado no servidor, sem expiração automática.
    // Requer adminSuperSecret além de username+password — segunda camada de verificação.
    fastify.post('/v1/auth/service-token',
        {
            schema: {
                body: {
                    type: 'object',
                    required: ['username', 'password', 'adminSuperSecret'],
                    properties: {
                        username: { type: 'string' },
                        password: { type: 'string' },
                        adminSuperSecret: { type: 'string' },
                    }
                },
                response: {
                    200: {
                        type: 'object',
                        properties: {
                            role: { type: 'string' },
                            serviceToken: { type: 'string' },
                        },
                    },
                    401: {
                        type: 'object',
                        properties: {
                            message: { type: 'string' },
                        },
                    },
                },
            }
        },
        async (request, reply) => {
            const { username, password, adminSuperSecret } = request.body
            if (adminSuperSecret !== ADMIN_SUPER_SECRET) {
                return reply.code(401).send({ message: 'Invalid adminSuperSecret' })
            }

            const user = authUsers.find(
                user =>
                    user.username.toLocaleLowerCase() === username.toLocaleLowerCase() &&
                    user.password === password
            )

            if (!user) {
                return reply.code(401).send({ message: 'Invalid credentials' })
            }

            // UUID como service token: opaco (não decodificável), armazenado em Map.
            const serviceToken = randomUUID()
            issuedServiceTokens.set(serviceToken, { username: user.username, role: user.role })
            return reply.send({ serviceToken, role: user.role })
        })
}

// Middleware de RBAC (Role-Based Access Control): verifica se o usuário tem o role necessário.
// Usado como preHandler em rotas específicas (POST, PUT, DELETE /v1/customers).
// Retorna uma função porque preHandler espera um handler, não uma chamada imediata.
export function requireRole(role) {
    return async function (request, reply) {
        if (request.user.role === role) return

        return reply.code(403).send({
            message: 'Forbidden: insufficient permissions'
        })
    }
}
