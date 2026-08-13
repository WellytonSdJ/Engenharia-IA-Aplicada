import { MessagesZodMeta } from '@langchain/langgraph';
import { withLangGraph } from '@langchain/langgraph/zod';
import type { BaseMessage } from '@langchain/core/messages';
import { z } from 'zod/v3';

export const GraphAnnotation =  z.object({
    // withLangGraph + MessagesZodMeta instrui o LangGraph a usar o reducer padrão de mensagens
    // (append de novas mensagens) em vez de sobrescrever o array inteiro a cada nó
    messages: withLangGraph(
        z.custom<BaseMessage[]>(),
        MessagesZodMeta),
    trendsData: z.string().optional(),   // texto gerado pelo researcher, consumido pelo responder
    question: z.string().optional(),     // pergunta original, propagada para o responder sem depender de messages
    keywords: z.array(z.string()).optional(),   // reservado para uma futura extração estruturada; nenhum nó escreve aqui hoje
});

export type GraphState = z.infer<typeof GraphAnnotation>;
