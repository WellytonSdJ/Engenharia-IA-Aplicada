// Hierarquia de erros de domínio: cada erro HTTP mapeado para um tipo específico.
// Isso permite que o código de infraestrutura lance um erro semântico (UnauthorizedError)
// em vez de um genérico "HTTP 401" — e o handler da tool decide como comunicar ao LLM.

export class UnauthorizedError extends Error {
    constructor(message = 'Unauthorized: service token is missing or invalid') {
        super(message);
        this.name = 'UnauthorizedError';
    }
}

// 403 é diferente de 401: o token é válido, mas o role não tem permissão para a operação.
// Ex: token com role 'member' tentando deletar um customer (que exige role 'admin').
export class ForbiddenError extends Error {
    constructor(message = 'Forbidden: token does not have sufficient permissions') {
        super(message);
        this.name = 'ForbiddenError';
    }
}

// 429 Too Many Requests — rate limit atingido. O cliente deve aguardar antes de tentar novamente.
export class RateLimitError extends Error {
    constructor(message = 'Rate limit exceeded. Please try again later.') {
        super(message);
        this.name = 'RateLimitError';
    }
}
