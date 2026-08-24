import type { Customer, CustomerMutation } from "../domain/customer.ts";
import { UnauthorizedError, ForbiddenError, RateLimitError } from "../domain/errors.ts";

export class CustomerHttpClient {
    private baseUrl: string;
    // Todas as requisições carregam o service token — armazenado uma vez no construtor.
    private authHeaders: Record<string, string>;

    constructor(baseUrl: string, serviceToken: string) {
        this.baseUrl = baseUrl;
        this.authHeaders = { Authorization: `Bearer ${serviceToken}` };
    }

    // Método privado com sintaxe # (JavaScript private field — não acessível externamente,
    // diferente de private do TypeScript que é só verificação em tempo de compilação).
    // Centraliza o tratamento de erros HTTP para todos os métodos — evita repetição.
    async #assertOk(res: Response): Promise<void> {
        if (res.status === 401) throw new UnauthorizedError();
        if (res.status === 403) throw new ForbiddenError();
        if (res.status === 429) throw new RateLimitError();
        // Para outros erros HTTP, inclui o body na mensagem para facilitar o diagnóstico.
        if (!res.ok) throw new Error(`HTTP ${res.status} - ${res.statusText} - ${await res.text()}`);
    }

    async listCustomers(): Promise<Customer[]> {
        const res = await fetch(`${this.baseUrl}/customers`, { headers: this.authHeaders });
        await this.#assertOk(res);
        return res.json() as Promise<Customer[]>;
    }

    async getCustomerById(id: string): Promise<Customer | null> {
        const res = await fetch(`${this.baseUrl}/customers/${id}`, { headers: this.authHeaders });
        // 404 e 400 são retornados como null em vez de exceção — "não encontrado" não é erro de auth.
        if (res.status === 404 || res.status === 400) return null;
        await this.#assertOk(res);
        return res.json() as Promise<Customer>;
    }

    async createCustomer(data: Omit<Customer, '_id'>): Promise<CustomerMutation> {
        const res = await fetch(`${this.baseUrl}/customers`, {
            method: "POST",
            // Spread de authHeaders junto com Content-Type — o service token vai em todas as requisições.
            headers: { "Content-Type": "application/json", ...this.authHeaders },
            body: JSON.stringify(data),
        });
        await this.#assertOk(res);
        return res.json() as Promise<CustomerMutation>;
    }

    async updateCustomer(
        id: string,
        data: Partial<Omit<Customer, '_id'>>
    ): Promise<CustomerMutation> {
        const res = await fetch(`${this.baseUrl}/customers/${id}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json", ...this.authHeaders },
            body: JSON.stringify(data),
        });
        await this.#assertOk(res);
        return res.json() as Promise<CustomerMutation>;
    }

    async deleteCustomer(id: string): Promise<CustomerMutation> {
        const res = await fetch(`${this.baseUrl}/customers/${id}`, {
            method: "DELETE",
            headers: this.authHeaders,
        });
        await this.#assertOk(res);
        return res.json() as Promise<CustomerMutation>;
    }
}
