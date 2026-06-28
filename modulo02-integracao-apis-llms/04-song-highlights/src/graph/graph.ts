import {
  StateGraph,
  START,
  END,
  MessagesZodMeta,
} from "@langchain/langgraph";
import { withLangGraph } from "@langchain/langgraph/zod";
import { z } from "zod/v3";

import type { BaseMessage } from '@langchain/core/messages';
import { OpenRouterService } from '../services/openrouterService.ts';
import { createChatNode } from './nodes/chatNode.ts';
import { createSummarizationNode } from './nodes/summarizationNode.ts';
import { createSavePreferencesNode } from './nodes/savePreferencesNode.ts';
import { routeAfterChat, routeAfterSavePreferences } from './nodes/edgeConditions.ts';
import { PreferencesService } from "../services/preferencesService.ts";
import { type MemoryService } from "../services/memoryService.ts";

// withLangGraph + MessagesZodMeta instrui o LangGraph a usar o reducer padrão de mensagens
// (append de novas mensagens, deleção via RemoveMessage) em vez de sobrescrever o array inteiro
const ChatStateAnnotation = z.object({
  messages: withLangGraph(
    z.custom<BaseMessage[]>(),
    MessagesZodMeta),
  userContext: z.string().optional(),        // contexto textual carregado do SQLite antes da sessão
  extractedPreferences: z.any().optional(),  // preferências detectadas na última mensagem — limpo após salvar
  needsSummarization: z.boolean().optional(),// flag setada pelo chatNode quando atinge o limite de mensagens
  conversationSummary: z.any().optional(),   // último resumo gerado — passado ao próximo ciclo de sumarização
  userId: z.string().optional(),
});

export type GraphState = z.infer<typeof ChatStateAnnotation>;

export function buildChatGraph(
  llmClient: OpenRouterService,
  preferencesService: PreferencesService,
  memoryService: MemoryService,
) {
  const graph = new StateGraph(ChatStateAnnotation)
    .addNode('chat', createChatNode(llmClient, preferencesService))
    .addNode('savePreferences', createSavePreferencesNode(preferencesService))
    .addNode('summarize', createSummarizationNode(llmClient, preferencesService))

    // Todo invoke começa no chatNode
    .addEdge(START, 'chat')

    // Após chat: vai para savePreferences se houver prefs, direto para summarize se precisar, ou encerra
    .addConditionalEdges(
      'chat',
      routeAfterChat,
      {
        savePreferences: 'savePreferences',
        summarize: 'summarize',
        end: END,
      }
    )

    // Após salvar prefs: ainda pode precisar sumarizar antes de encerrar
    .addConditionalEdges(
      'savePreferences',
      routeAfterSavePreferences,
      {
        summarize: 'summarize',
        end: END,
      }
    )

    // Sumarização sempre encerra o ciclo — não há mais nós depois
    .addEdge('summarize', END);

  // checkpointer salva o state completo por thread_id no Postgres (memória entre invocações)
  // store é um key-value persistido disponível aos nós via runtime
  return graph.compile({
    checkpointer: memoryService.checkpointer,
    store: memoryService.store,
  });
}
