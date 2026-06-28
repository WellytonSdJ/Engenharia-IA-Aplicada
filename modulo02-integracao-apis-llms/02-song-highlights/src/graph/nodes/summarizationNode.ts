import { HumanMessage } from 'langchain';
import { type Runtime } from '@langchain/langgraph'
import { OpenRouterService } from '../../services/openrouterService.ts';
import type { GraphState } from '../graph.ts';
import { type ConversationSummary, getSummarizationSystemPrompt, getSummarizationUserPrompt, SummarySchema } from '../../prompts/v1/summarization.ts';
import { PreferencesService } from '../../services/preferencesService.ts';
import { RemoveMessage } from '@langchain/core/messages';

export function createSummarizationNode(llmClient: OpenRouterService, preferencesService: PreferencesService) {
    return async (state: GraphState, runtime?: Runtime): Promise<Partial<GraphState>> => {
        // Normaliza as mensagens do state para um formato legível pelo prompt
        const conversationHistory = state.messages.map(msg => ({
            role: HumanMessage.isInstance(msg) ? 'User' : 'AI',
            content: msg.text
        }))

        // Se já existe um resumo anterior, ele é passado junto para o LLM manter continuidade
        const previousSummary = state.conversationSummary as ConversationSummary | undefined
        const systemPrompt = getSummarizationSystemPrompt()
        const userPrompt = getSummarizationUserPrompt(
            conversationHistory,
            previousSummary,
        )

        // Chama o LLM com saída estruturada — o retorno já vem validado pelo SummarySchema (Zod)
        const result = await llmClient.generateStructured(
            systemPrompt,
            userPrompt,
            SummarySchema,
        )

        if (result.error || !result.data) {
            console.error('❌ Falha ao sumarizar conversa:', result.error);

            // Aborta a sumarização sem lançar exceção para não quebrar o grafo
            return {
                needsSummarization: false
            }
        }

        // userId vem do runtime (contexto de execução do LangGraph) ou do state como fallback
        const userId = String(runtime?.context?.userId || state.userId || 'unknown')

        // Persiste o resumo no banco para ser carregado nas próximas sessões
        await preferencesService.storeSummary(
            userId, result.data,
        )

        // Mantém apenas as 2 últimas mensagens no state — o restante é descartado via RemoveMessage
        // O LangGraph processa RemoveMessage como instrução de deleção no reducer de mensagens
        const deleteMessages = state.messages
            .slice(0, -2)
            .map(m => new RemoveMessage({ id: m.id as string }))

        return {
            messages: deleteMessages,
            conversationSummary: result.data,
            needsSummarization: false,
        };
    };
}
