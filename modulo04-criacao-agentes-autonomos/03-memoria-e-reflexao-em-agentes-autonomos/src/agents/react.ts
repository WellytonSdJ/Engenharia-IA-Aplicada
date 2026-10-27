import { AIMessage, HumanMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { AgentConfigurationError, IterationLimitError, ValidationError } from "../services/errors.js";
import { defaultOpsStore, createForgetPreferenceTool, createOpsTools } from "../services/tools.js";
import { resolveModel } from "./model.js";
import { createExecutionContext, ReasoningOptionsSchema, type ReasoningOptions, type ReasoningStrategy } from "./strategy.js";
import { OPSPILOT_SYSTEM_PROMPT } from "./system-prompt.js";
import type { TraceEvent } from "./trace.js";

function messageText(message: BaseMessage): string {
  return typeof message.content === "string" ? message.content : JSON.stringify(message.content);
}

function traceFromMessages(messages: readonly BaseMessage[]): TraceEvent[] {
  const trace: TraceEvent[] = [];
  for (const message of messages) {
    if (message instanceof AIMessage) {
      const toolCalls = message.tool_calls ?? [];
      if (toolCalls.length > 0) {
        if (message.content) {
          trace.push({ type: "thought", content: messageText(message) });
        }
        for (const call of toolCalls) {
          trace.push({
            type: "action",
            content: `Calling ${call.name}`,
            tool: call.name,
            args: call.args,
          });
        }
      } else if (message.content) {
        trace.push({ type: "answer", content: messageText(message) });
      }
    } else if (message instanceof ToolMessage) {
      trace.push({ type: "observation", content: messageText(message) });
    }
  }
  return trace;
}

export interface ReactAgentRunner {
  invoke(
    input: { messages: BaseMessage[] },
    options: { recursionLimit: number },
  ): Promise<{ messages: BaseMessage[] }>;
}

export type ReactAgentFactory = (
  model: BaseChatModel,
  tools: StructuredToolInterface[],
  recursionLimit: number,
) => ReactAgentRunner;

const langGraphAgentFactory: ReactAgentFactory = (model, tools, recursionLimit) => {
  const agent = createReactAgent({ llm: model, tools, prompt: OPSPILOT_SYSTEM_PROMPT });
  return {
    async invoke(input) {
      const result = await agent.invoke(input, { recursionLimit });
      return { messages: result.messages };
    },
  };
};

export function createReactStrategy(agentFactory: ReactAgentFactory = langGraphAgentFactory): ReasoningStrategy {
  return {
    name: "react",
    async run(input: string, options?: ReasoningOptions) {
    const { maxIterations } = ReasoningOptionsSchema.parse({ maxIterations: options?.maxIterations });
    if (!input.trim()) {
      throw new ValidationError("Input must not be empty.");
    }
    const model = resolveModel(options?.model);
    const store = options?.store ?? defaultOpsStore;
    const context = createExecutionContext(maxIterations);
    const tools: StructuredToolInterface[] = [...createOpsTools(store)];
    if (options?.memory) {
      // Spec 010: só existe quando há um usuário no turno (/chat). Arena e bench não passam memory.
      tools.push(createForgetPreferenceTool(options.memory.store, options.memory.userId));
    }
    const agent = agentFactory(model, tools, maxIterations * 2 + 1);
    let messages: BaseMessage[];
    try {
      const result = await agent.invoke(
        { messages: [new HumanMessage(input)] },
        { recursionLimit: maxIterations * 2 + 1 },
      );
      messages = result.messages;
    } catch (error) {
      if (error instanceof Error && error.name === "GraphRecursionError") {
        throw new IterationLimitError(maxIterations);
      }
      throw error;
    }
    const trace = traceFromMessages(messages);
    const answer = [...trace].reverse().find((event) => event.type === "answer")?.content;
    if (!answer) {
      throw new AgentConfigurationError("ReAct agent completed without a final answer.");
    }
    context.trace.push(...trace);
    context.llmCalls = messages.filter((message) => message instanceof AIMessage).length;
    context.iterations = Math.max(0, context.llmCalls - (answer ? 1 : 0));
    if (context.iterations > maxIterations) {
      throw new IterationLimitError(maxIterations);
    }
    return context.result(answer);
    },
  };
}

export const reactStrategy = createReactStrategy();

export { traceFromMessages };
