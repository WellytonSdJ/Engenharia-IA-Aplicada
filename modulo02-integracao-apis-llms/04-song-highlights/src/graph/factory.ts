import { OpenRouterService } from '../services/openrouterService.ts';
import { config } from '../config.ts';
import { buildChatGraph } from './graph.ts';
import { createMemoryService } from '../services/memoryService.ts';
import { PreferencesService } from '../services/preferencesService.ts';

export async function buildGraph(dbPath: string = './preferences.db') {
  const llmClient = new OpenRouterService(config);

  // memoryService inicializa checkpointer (histórico de mensagens) e store (dados extras) no Postgres
  const memoryService = await createMemoryService()
  // preferencesService usa SQLite separado — persistência leve de preferências do usuário entre sessões
  const preferencesService = new PreferencesService(dbPath)
  const graph = buildChatGraph(
    llmClient,
    preferencesService,
    memoryService
  );

  // Expõe preferencesService para o index.ts poder carregar contexto antes da primeira mensagem
  return {
    graph,
    preferencesService,
  };
}

export const graph = async () => buildGraph();
export default graph;
