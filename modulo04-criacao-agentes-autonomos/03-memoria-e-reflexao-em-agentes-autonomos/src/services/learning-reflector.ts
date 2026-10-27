import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";
import type { MemoryStore } from "../models/memory-store.js";

// Reflection reaproveitada como LOOP DE APRENDIZADO (spec 010). O `withReflection` da spec 002
// critica a RESPOSTA antes de entregá-la; este refletor olha a MENSAGEM DO USUÁRIO depois que a
// resposta já saiu e decide se há algo que valha lembrar nas próximas sessões.
export const LEARNING_REFLECTOR_PROMPT = [
  "Você destila APRENDIZADOS DURÁVEIS da mensagem do usuário para a memória de longo prazo de um copiloto de plantão (OpsPilot).",
  'hasLearning=true SOMENTE se a mensagem contiver preferência ou fato operacional DURÁVEL (ex.: "sempre priorize o serviço checkout", "prefiro resumos curtos").',
  'hasLearning=false para pedidos PONTUAIS one-shot (ex.: "liste alertas", "abra um incidente", "resolva X agora", perguntas efêmeras de status).',
  "hasLearning=false se a mensagem contiver SEGREDOS ou dados sensíveis (senhas, tokens, API keys, credenciais).",
  "Quando hasLearning=true, fact deve ser um enunciado estável em uma frase (não copie o pedido bruto).",
  "Quando hasLearning=false, fact deve ser string vazia.",
].join(" ");

// Schema plano (só boolean + string), como os outros structured outputs do projeto: modelos
// gratuitos via OpenRouter erram bem menos com objetos simples.
export const LearningReflectionSchema = z.object({
  hasLearning: z
    .boolean()
    .describe("true somente se houver preferência ou fato operacional DURÁVEL (nunca pedido pontual, nunca segredo)"),
  fact: z
    .string()
    .describe("Enunciado estável em 1 frase; string vazia se hasLearning=false. Nunca copie pedido pontual nem segredo."),
});

export type LearningReflection = z.infer<typeof LearningReflectionSchema>;

/** Recebe a última mensagem do usuário e diz se há um fato durável. Injetável: nos testes é um fake. */
export type LearningReflectorFn = (userMessage: string) => Promise<LearningReflection>;

const NO_LEARNING: LearningReflection = { hasLearning: false, fact: "" };

export function createLLMLearningReflector(modelFactory: () => BaseChatModel): LearningReflectorFn {
  return async (userMessage) => {
    try {
      const output = await modelFactory()
        .withStructuredOutput(LearningReflectionSchema)
        .invoke([new SystemMessage(LEARNING_REFLECTOR_PROMPT), new HumanMessage(userMessage)]);
      return LearningReflectionSchema.parse(output);
    } catch {
      // Modelo fora do ar, JSON inválido, chave ausente: na dúvida, não aprende nada.
      return NO_LEARNING;
    }
  };
}

/**
 * Pergunta ao refletor e, se houver fato durável, grava com `remember` (que já deduplica).
 * Nunca lança: roda depois da resposta HTTP e não pode derrubar nada.
 */
export async function scheduleLearning(args: {
  reflector: LearningReflectorFn;
  memoryStore: MemoryStore;
  userId: string;
  userMessage: string;
}): Promise<void> {
  try {
    const reflection = await args.reflector(args.userMessage);
    const fact = reflection.fact.trim();
    // `hasLearning: true` com fato vazio é contraditório → trata como "nada a aprender".
    if (!reflection.hasLearning || !fact) {
      return;
    }
    await args.memoryStore.remember(args.userId, fact);
  } catch (error) {
    // Best-effort, mas observável: registra no stderr em vez de engolir em silêncio.
    console.error("learning reflector failed:", error instanceof Error ? error.message : String(error));
  }
}
