import type { GraphState } from '../graph.ts';

// Prioridade: salvar prefs antes de sumarizar — garante que os dados não se percam se a sumarização falhar
export const routeAfterChat = (state: GraphState): string =>
  state.extractedPreferences ? 'savePreferences' :
  state.needsSummarization ? 'summarize' : 'end';

// Após salvar, ainda verifica se precisa sumarizar (flag pode ter sido setada no mesmo invoke)
export const routeAfterSavePreferences = (state: GraphState): string =>
  state.needsSummarization ? 'summarize' : 'end';
