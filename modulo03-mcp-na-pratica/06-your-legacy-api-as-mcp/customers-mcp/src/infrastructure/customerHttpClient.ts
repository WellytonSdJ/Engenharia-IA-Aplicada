import { type CustomerMutation, type Customer, type CustomerUpdate } from "../domain/customer.ts"

// Camada de infraestrutura: encapsula os detalhes de HTTP (fetch, headers, status codes).
// O resto da aplicação não sabe que a fonte de dados é uma API REST — só conhece os tipos do domínio.
export class CustomerHttpClient {
    private baseUrl: string
    constructor(baseUrl: string) {
        this.baseUrl = baseUrl
    }

    async listCustomers(): Promise<Customer[]> {
        const res = await fetch(`${this.baseUrl}/customers`)
        return res.json() as Promise<Customer[]>
    }

    async createCustomer(customer: Customer) {
        const res = await fetch(`${this.baseUrl}/customers`, {
            method: 'POST',
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(customer),
        })
        return res.json() as Promise<CustomerMutation>
    }

    // Retorna null em vez de lançar exceção para 404 — deixa o chamador decidir o que fazer com "não encontrado".
    async getCustomerById(id: string): Promise<Customer | null> {
        const res = await fetch(`${this.baseUrl}/customers/${id}`)
        if (res.status === 404) return null

        return res.json() as Promise<Customer>
    }

    async updateCustomer(customer: CustomerUpdate) {
        // Separa _id do restante do payload — a API recebe _id na URL, não no body.
        const {_id, ...remaining } = customer
        const res = await fetch(`${this.baseUrl}/customers/${_id}`, {
            method: 'PUT',
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(remaining),
        })

        return res.json() as Promise<CustomerMutation>
    }

    async deleteCustomer(id: string): Promise<CustomerMutation> {
        const response = await fetch(`${this.baseUrl}/customers/${id}`, {
            method: "DELETE"
        })

        return response.json() as Promise<CustomerMutation>
    }
}
