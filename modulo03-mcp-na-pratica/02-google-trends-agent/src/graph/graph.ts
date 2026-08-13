import { StateGraph, START, END } from '@langchain/langgraph';
import { createResearcherNode } from './nodes/researcherNode.ts';
import { createResponderNode } from './nodes/responderNode.ts';
import { OpenRouterService } from '../services/openRouterService.ts';
import { GraphAnnotation } from './state.ts';

// Prompt chaining linear: sem addConditionalEdges e sem loop de qualidade — a ordem
// researcher → responder é sempre a mesma, diferente do agente autônomo de 01-multiple-mcp-tools
export function buildTrendsGraph(openRouterService: OpenRouterService) {
  return new StateGraph(GraphAnnotation)
    .addNode('researcher', createResearcherNode(openRouterService))
    .addNode('responder', createResponderNode(openRouterService))

    .addEdge(START, 'researcher')
    .addEdge('researcher', 'responder')
    .addEdge('responder', END)

    .compile();
}
