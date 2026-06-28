import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres"
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store"
import { config } from "../config.ts"

export type MemoryService = {
    checkpointer: PostgresSaver
    store: PostgresStore
}

export async function createMemoryService(): Promise<MemoryService> {
    const dbUri = config.memory.dbUri

    // store: key-value persistido (usado para dados extras por thread/namespace)
    // checkpointer: snapshot do GraphState completo após cada nó — base do "memory" do LangGraph
    const store = PostgresStore.fromConnString(dbUri)
    const checkpointer = PostgresSaver.fromConnString(dbUri)

    // setup() cria as tabelas necessárias no Postgres se ainda não existirem
    await store.setup()
    await checkpointer.setup()

    console.log(`✅ Memória configurada: PostgreSQL`);
    return {
        checkpointer,
        store,
    }
}