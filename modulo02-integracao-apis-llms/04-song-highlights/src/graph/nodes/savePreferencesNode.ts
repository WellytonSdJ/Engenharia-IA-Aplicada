import type { Runtime } from '@langchain/langgraph';
import type { GraphState } from '../graph.ts';
import { PreferencesService } from '../../services/preferencesService.ts';

export function createSavePreferencesNode(preferencesService: PreferencesService) {
  return async (state: GraphState, runtime?: Runtime): Promise<Partial<GraphState>> => {
    // Guard: a edge condition já garante isso, mas checagem explícita evita efeito colateral
    if(!state.extractedPreferences) return {}

    const userId = String(runtime?.context?.userId || state.userId || 'unknown')
    // merge acumula gêneros/bandas via Set — não substitui o que já existia no SQLite
    await preferencesService.mergePreferences(userId, state.extractedPreferences)

    // Limpa do state para não reprocessar em invocações seguintes
    return {
      extractedPreferences: undefined
    };
  };
}
