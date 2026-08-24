import { type CustomerQuery, type Customer, type CustomerUpdate } from "../domain/customer.ts";
import { CustomerHttpClient } from "../infrastructure/customerHttpClient.ts";

// Camada de aplicação: contém a lógica de negócio que não pertence à infraestrutura (HTTP) nem ao protocolo (MCP).
// As tools do MCP chamam esta classe — ela não sabe nada de MCP, só de customers.
export class CustomerService {
    private readonly client: CustomerHttpClient
    constructor(baseUrl: string) {
        this.client = new CustomerHttpClient(baseUrl)
    }

    async listCustomers(): Promise<Customer[]> {
        return this.client.listCustomers()
    }

    async createCustomer(customer: Omit<Customer, '_id'>) {
        return this.client.createCustomer(customer)
    }

    // findCustomer implementa busca flexível: se vier _id, vai direto; caso contrário, busca local por qualquer campo.
    // Isso evita expor na API REST um endpoint de busca genérica que a API legada talvez não tenha.
    async findCustomer(query: CustomerQuery): Promise<Customer | null> {
        if(query._id) return this.client.getCustomerById(query._id)

        const customers = await this.client.listCustomers()
        return (
            customers.find(customer => {
                const entries = Object.entries(query) as [keyof Customer, string][]

                // entries.every garante que TODOS os campos da query estejam presentes no customer.
                // .includes() em vez de === permite busca parcial ("Ana" encontra "Ana Lima").
                return entries.every(([key, value]) => {
                    const customerValue = customer[key]
                    return customerValue?.includes(value)
                })
            })
        ) ?? null
    }

    async updateCustomer(customer: CustomerUpdate) {
        return this.client.updateCustomer(customer)
    }

    async deleteCustomer(id: string) {
        return this.client.deleteCustomer(id)
    }
}
