import type { Runtime } from "@langchain/langgraph";
import { OpenRouterService } from "../../services/openrouterService.ts";
import type { GraphState } from "../graph.ts";
import {
  ChatResponseSchema,
  getSystemPrompt,
  getUserPromptTemplate,
} from "../../prompts/v1/chatResponse.ts";
import { AIMessage, HumanMessage } from "langchain";
import { PreferencesService } from "../../services/preferencesService.ts";
import { config } from "../../config.ts";

export function createChatNode(
  llmClient: OpenRouterService,
  preferencesService: PreferencesService,
) {
  return async (
    state: GraphState,
    runtime?: Runtime,
  ): Promise<Partial<GraphState>> => {
    const userId = String(
      runtime?.context?.userId || state.userId || "unknown",
    );

    // userContext vem do state se já foi carregado (primeira mensagem), senão busca do SQLite
    const userContext =
      state.userContext ?? (await preferencesService.getBasicInfo(userId));
    const systemPrompt = getSystemPrompt(userContext);

    // Serializa o histórico para texto — o LLM recebe como parte do user prompt
    const conversationHistory = state.messages
      .map(
        (msg) =>
          `${HumanMessage.isInstance(msg) ? "User" : "AI"}: ${msg.content}`,
      )
      .join("\n");

    // A última mensagem do state é sempre a do usuário atual
    const userMessage = state.messages.at(-1)?.text as string;
    const userPrompt = getUserPromptTemplate(userMessage, conversationHistory);

    // Retorno já validado pelo ChatResponseSchema — inclui message, preferences e shouldSavePreferences
    const result = await llmClient.generateStructured(
      systemPrompt,
      userPrompt,
      ChatResponseSchema,
    );

    if (!result.success || !result.data) {
      console.error("❌ Falha ao gerar resposta:", result.error);
      return {
        messages: [
          new AIMessage("Desculpe, encontrei um erro. Pode tentar novamente?"),
        ],
      };
    }

    const response = result.data;

    // Calculate if summarization is needed based on total message count
    // After summarization, we keep 2 messages (1 user + 1 AI)
    // So we trigger summarization when we have 6+ messages (3 exchanges)
    // This gives: initial 2 + 4 new messages = 6 messages total

    const totalMessages = state.messages.length;
    // O threshold é configurável em config.maxMessagesToSummary — não hardcoded aqui
    const needsSummarization = totalMessages >= config.maxMessagesToSummary;

    return {
      messages: [new AIMessage(response.message)],
      // Só propaga preferences se o LLM decidiu que vale salvar — evita ruído no state
      extractedPreferences: response.shouldSavePreferences
        ? response.preferences
        : undefined,
      needsSummarization,
    };
  };
}
